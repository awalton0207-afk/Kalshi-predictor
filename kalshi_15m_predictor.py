"""
Kalshi BTC 15-minute predictor
------------------------------
For every Kalshi "BTC price up in next 15 mins?" window (series KXBTC15M), this
sends an UP or DOWN call to your phone 10 minutes before the window closes.

What goes into each call:
  1. Price vs target   - where BTC is now compared with the window's target price,
                         and how far BTC usually moves in the time that's left
  2. Kalshi's own odds - the live YES/NO prices (Kalshi's odds track real results closely)
  3. Technicals        - the signals from the Grok bot: RSI, EMA 9/21, 3-candle momentum,
                         last-5-minute trend, volume spike, 1-hour trend.
                         Grok's Kalshi research said these are weak on their own,
                         so they only nudge the odds a few percent.
  4. Warnings          - too close to call, quiet market, bot vs Kalshi disagreeing
  5. Scorecard         - every call is saved to kalshi_predictions.csv and checked
                         against the real result, so you can see how accurate it is.
  6. Paper trades      - each call is paper-traded (no real money) using rules borrowed
                         from Pondletter's Ping tool: start at 1 contract, buy at the real
                         ask only if there are contracts for sale there, count Kalshi's fee,
                         and stop for the day after a set loss (daily loss limit).
                         Like Ping, paper results still leave out partial fills, your place
                         in line and outages, so real results can be worse.

This only sends predictions. It never places bets. Not financial advice.

Setup: pip install requests     Run: python kalshi_15m_predictor.py
Test:  python kalshi_15m_predictor.py --now    (sends one call right away)
"""

import csv
import math
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

# ======================= SETTINGS =======================
# Phone alerts: install the free "ntfy" app and subscribe to this topic name.
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")   # set as a GitHub secret

# Optional Telegram alerts as well (leave blank to skip)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

LEAD_MINUTES = 10                 # send the call this many minutes before the window closes
SERIES = "KXBTC15M"               # Kalshi BTC 15-minute markets
COINBASE_PRODUCT = "BTC-USD"
BINANCE_SYMBOL = "BTCUSDT"        # backup price source

# Indicator settings (same as the Grok bot)
RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT = 14, 30, 70
EMA_FAST, EMA_SLOW = 9, 21
ATR_PERIOD = 14
VOLUME_SPIKE = 1.5                # 1.5x normal volume counts as a spike
MOMENTUM_PCT = 0.15               # 3-candle move bigger than 0.15%
SHORT_TREND_PCT = 0.05            # last-5-minute move bigger than 0.05%

# How the pieces are combined
MODEL_WEIGHT = 0.5                # price-vs-target odds
MARKET_WEIGHT = 0.5               # Kalshi's odds
TILT_PER_POINT = 0.01             # each technicals point moves the odds 1%
MAX_TILT = 0.05                   # technicals can move the odds 5% at most
NEAR_STRIKE_DOLLARS = 20          # within $20 of the target = too close to call
KALSHI_FEE_RATE = 0.07            # Kalshi taker fee is about 0.07 x price x (1 - price)
# Paper trading (rules borrowed from Pondletter's Ping)
PAPER_CONTRACTS = 1               # start small: 1 contract per call
DAILY_LOSS_LIMIT = 3.00           # stop paper trading for the day after losing this many dollars
VALUE_EDGE = 0.01                 # a "value" call: bot odds beat the price + fee by over 1 cent

# How long one cloud run lasts before it hands off to the next one (GitHub allows 6 hours)
MAX_RUNTIME_MINUTES = float(os.environ.get("MAX_RUNTIME_MINUTES", "0"))   # 0 = run forever
# ========================================================

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(HERE, "kalshi_predictions.csv")
LOG_FIELDS = ["ticker", "close_time_utc", "sent_at_utc", "minutes_left", "call", "bot_odds",
              "confidence", "kalshi_up_odds", "kalshi_favorite", "price_model_up", "tech_score",
              "target", "btc_price", "too_close", "result",
              "paper_price", "paper_fee", "paper_note", "value_call", "paper_pnl"]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

session = requests.Session()
session.headers["User-Agent"] = "kalshi-15m-predictor/1.0"


# ------------------------- data -------------------------
def get_json(url, params=None, tries=3):
    last_error = None
    for attempt in range(tries):
        try:
            r = session.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
    raise last_error


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def kalshi_price(m, name):
    """Kalshi prices as 0-1 (handles both the new *_dollars fields and old cents fields)."""
    v = m.get(f"{name}_dollars")
    if v not in (None, ""):
        return float(v)
    v = m.get(name)
    if v not in (None, ""):
        return float(v) / 100
    return None


def current_market():
    data = get_json(f"{KALSHI}/markets", {"series_ticker": SERIES, "status": "open", "limit": 50})
    now = datetime.now(timezone.utc)
    markets = [m for m in data.get("markets", [])
               if m.get("close_time") and parse_time(m["close_time"]) > now]
    return min(markets, key=lambda m: parse_time(m["close_time"])) if markets else None


def market_by_ticker(ticker):
    return get_json(f"{KALSHI}/markets/{ticker}").get("market", {})


def target_price(m):
    if m.get("floor_strike") not in (None, ""):
        return float(m["floor_strike"])
    text = (m.get("yes_sub_title") or "").replace(",", "")
    if "$" in text:
        return float(text.split("$")[-1].strip())
    return None


def kalshi_up_odds(m):
    bid, ask = kalshi_price(m, "yes_bid"), kalshi_price(m, "yes_ask")
    if bid and ask:
        return (bid + ask) / 2
    return kalshi_price(m, "last_price")


def coinbase_candles(seconds):
    rows = get_json(f"https://api.exchange.coinbase.com/products/{COINBASE_PRODUCT}/candles",
                    {"granularity": seconds})
    rows = sorted(rows, key=lambda r: r[0])
    return [{"time": r[0], "low": float(r[1]), "high": float(r[2]), "open": float(r[3]),
             "close": float(r[4]), "volume": float(r[5])} for r in rows]


def binance_candles(seconds):
    interval = {60: "1m", 900: "15m", 3600: "1h"}[seconds]
    rows = get_json("https://data-api.binance.vision/api/v3/klines",
                    {"symbol": BINANCE_SYMBOL, "interval": interval, "limit": 300})
    return [{"time": r[0] // 1000, "open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
             "close": float(r[4]), "volume": float(r[5])} for r in rows]


def candles(seconds):
    try:
        return coinbase_candles(seconds)
    except Exception as e:
        print("Coinbase candles failed, using Binance:", e)
        return binance_candles(seconds)


def btc_price():
    try:
        return float(get_json(f"https://api.exchange.coinbase.com/products/{COINBASE_PRODUCT}/ticker")["price"])
    except Exception as e:
        print("Coinbase price failed, using Binance:", e)
        return float(get_json("https://data-api.binance.vision/api/v3/ticker/price",
                              {"symbol": BINANCE_SYMBOL})["price"])


# ---------------------- indicators ----------------------
def ema_series(values, period):
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    out = [sum(values[:period]) / period]
    for v in values[period:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def rsi(closes, period=14):
    """Standard (Wilder) RSI."""
    if len(closes) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0)
        losses += max(-d, 0)
    avg_gain, avg_loss = gains / period, losses / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(d, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0)) / period
    if avg_loss == 0:
        return 100.0
    return 100 - 100 / (1 + avg_gain / avg_loss)


def atr(c, period=14):
    if len(c) < period + 1:
        return None
    trs = [max(c[i]["high"] - c[i]["low"], abs(c[i]["high"] - c[i - 1]["close"]),
               abs(c[i]["low"] - c[i - 1]["close"])) for i in range(1, len(c))]
    return sum(trs[-period:]) / period


def pct(a, b):
    return (a - b) / b * 100


def kalshi_fee(price, contracts=1):
    """Kalshi taker fee: 0.07 x contracts x price x (1 - price), rounded up to the next cent."""
    return math.ceil(round(KALSHI_FEE_RATE * contracts * price * (1 - price) * 100, 6)) / 100


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ----------------------- signals ------------------------
def technicals(c15, c1, c60, price):
    """Grok's signals. Positive score leans UP, negative leans DOWN."""
    closes = [x["close"] for x in c15]
    closes[-1] = price                      # use the live price for the candle in progress
    score, lines = 0, []

    # 1. RSI (Grok: use it as a secondary filter only, so it counts 1 point)
    r = rsi(closes, RSI_PERIOD)
    if r is not None:
        if r < RSI_OVERSOLD:
            score += 1
            lines.append(f"RSI {r:.0f}: oversold (leans UP)")
        elif r > RSI_OVERBOUGHT:
            score -= 1
            lines.append(f"RSI {r:.0f}: overbought (leans DOWN)")
        else:
            lines.append(f"RSI {r:.0f}: neutral")

    # 2. EMA 9/21 trend and real crossover check
    fast, slow = ema_series(closes, EMA_FAST), ema_series(closes, EMA_SLOW)
    if len(fast) >= 2 and len(slow) >= 2:
        f_now, f_prev, s_now, s_prev = fast[-1], fast[-2], slow[-1], slow[-2]
        if f_prev <= s_prev and f_now > s_now:
            ema_text = "just crossed UP"
        elif f_prev >= s_prev and f_now < s_now:
            ema_text = "just crossed DOWN"
        else:
            ema_text = "bullish" if f_now > s_now else "bearish"
        score += 1 if f_now > s_now else -1
        lines.append(f"EMA 9/21: {ema_text}")

    # 3. 3-candle (45 min) momentum
    if len(closes) >= 4:
        mom = pct(closes[-1], closes[-4])
        if mom > MOMENTUM_PCT:
            score += 1
        elif mom < -MOMENTUM_PCT:
            score -= 1
        lines.append(f"Last 45 min: {mom:+.2f}%")

    # 4. Last 5 minutes (Grok: late-window momentum is what matters most)
    if len(c1) >= 6:
        short = pct(price, c1[-6]["close"])
        if short > SHORT_TREND_PCT:
            score += 1
        elif short < -SHORT_TREND_PCT:
            score -= 1
        lines.append(f"Last 5 min: {short:+.2f}%")

    # 5. Volume spike (last 5 finished minutes vs the hour before) amplifies the lean
    vols = [x["volume"] for x in c1]
    rel_vol = None
    if len(vols) >= 66:
        base = statistics.mean(vols[-66:-6])
        rel_vol = statistics.mean(vols[-6:-1]) / base if base else None
    if rel_vol is not None:
        if rel_vol >= VOLUME_SPIKE and score != 0:
            score += 1 if score > 0 else -1
            lines.append(f"Volume {rel_vol:.1f}x normal: spike, backs the lean")
        else:
            lines.append(f"Volume {rel_vol:.1f}x normal")

    # 6. 1-hour trend (higher timeframe bias)
    htf = "neutral"
    hour_ema = ema_series([x["close"] for x in c60], 50)
    if hour_ema:
        if price > hour_ema[-1] * 1.002:
            htf = "bullish"
        elif price < hour_ema[-1] * 0.998:
            htf = "bearish"
        if htf == "bullish" and score > 0:
            score += 1
        elif htf == "bearish" and score < 0:
            score -= 1
        lines.append(f"1-hour trend: {htf}")

    atr_val = atr(c15, ATR_PERIOD)
    if atr_val:
        lines.append(f"ATR(15m): ${atr_val:,.0f}")
    return score, lines, rel_vol


def price_vs_target_odds(price, target, c1, minutes_left):
    """Chance BTC ends at or above the target, from how much BTC has been moving per minute."""
    closes = [x["close"] for x in c1[-61:]]
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    sigma = statistics.pstdev(rets) if len(rets) > 10 else 0.0008
    sigma = max(sigma, 1e-5)
    t = max(minutes_left - 0.5, 0.25)       # settles on a 60-second average, so ~half a minute less
    z = math.log(price / target) / (sigma * math.sqrt(t))
    p = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    typical_move = price * sigma * math.sqrt(t)
    return min(max(p, 0.02), 0.98), z, typical_move


# ---------------------- scorecard -----------------------
def read_log():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_log(rows):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def update_results():
    rows = read_log()
    now = datetime.now(timezone.utc)
    changed = False
    for row in rows:
        if row.get("result") or not row.get("close_time_utc"):
            continue
        if parse_time(row["close_time_utc"]) > now - timedelta(minutes=1):
            continue
        try:
            res = market_by_ticker(row["ticker"]).get("result")
        except Exception:
            continue
        if res in ("yes", "no"):
            row["result"] = "UP" if res == "yes" else "DOWN"
            price, fee = to_float(row.get("paper_price")), to_float(row.get("paper_fee"))
            if price is not None:
                won = row["call"] == row["result"]
                pnl = ((1 - price) if won else -price) * PAPER_CONTRACTS - (fee or 0)
                row["paper_pnl"] = f"{pnl:.2f}"
            changed = True
    if changed:
        write_log(rows)
    return rows


def scorecard(rows):
    done = [r for r in rows if r.get("result")]
    if not done:
        return "Scorecard: no finished windows yet"
    last = done[-1]
    mark = "✅" if last["call"] == last["result"] else "❌"
    hit = sum(r["call"] == r["result"] for r in done)
    lines = [f"Last window: called {last['call']} → {last['result']} {mark}",
             f"Bot record: {hit}/{len(done)} ({hit / len(done):.0%})"]
    high = [r for r in done if r.get("confidence") == "High"]
    if high:
        h = sum(r["call"] == r["result"] for r in high)
        lines.append(f"High-confidence calls: {h}/{len(high)} ({h / len(high):.0%})")
    fav = [r for r in done if r.get("kalshi_favorite")]
    if fav:
        f = sum(r["kalshi_favorite"] == r["result"] for r in fav)
        lines.append(f"Just following Kalshi's favorite: {f}/{len(fav)} ({f / len(fav):.0%})")
    lines += paper_summary(rows)
    return "\n".join(lines)


def et_date(iso):
    dt = parse_time(iso)
    try:
        from zoneinfo import ZoneInfo
        return dt.astimezone(ZoneInfo("America/New_York")).date()
    except Exception:
        return dt.astimezone().date()


def paper_today(rows, today):
    return sum(to_float(r.get("paper_pnl")) or 0 for r in rows
               if r.get("paper_pnl") and r.get("sent_at_utc") and et_date(r["sent_at_utc"]) == today)


def paper_summary(rows):
    traded = [r for r in rows if r.get("paper_pnl")]
    if not traded:
        return []
    today = et_date(datetime.now(timezone.utc).isoformat())

    def line(label, group):
        if not group:
            return f"• {label}: no trades yet"
        total = sum(float(r["paper_pnl"]) for r in group)
        day = sum(float(r["paper_pnl"]) for r in group if et_date(r["sent_at_utc"]) == today)
        wins = sum(float(r["paper_pnl"]) > 0 for r in group)
        return (f"• {label}: today {day:+.2f} | all-time {total:+.2f} "
                f"({wins}/{len(group)} won)")

    return ["", f"Paper trades ({PAPER_CONTRACTS} contract, fees included, $):",
            line("Every call", traded),
            line("Only value calls", [r for r in traded if r.get("value_call") == "yes"])]


# ----------------------- alerts -------------------------
def notify(title, body, tag="bar_chart"):
    print("\n" + title + "\n" + body + "\n" + "-" * 40)
    if NTFY_TOPIC:
        try:
            session.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=body.encode("utf-8"),
                         headers={"Title": title, "Tags": tag}, timeout=10)
        except Exception as e:
            print("ntfy error:", e)
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            session.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                         data={"chat_id": TELEGRAM_CHAT_ID, "text": f"{title}\n{body}"}, timeout=10)
        except Exception as e:
            print("Telegram error:", e)


def local_time(dt):
    try:
        from zoneinfo import ZoneInfo
        dt = dt.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        dt = dt.astimezone()
    return dt.strftime("%I:%M %p ET").lstrip("0")


# ------------------------ main --------------------------
def make_call():
    rows = update_results()
    m = current_market()
    if not m:
        notify("Kalshi BTC: no open market", "No open BTC 15-minute market found right now.")
        return
    now = datetime.now(timezone.utc)
    close = parse_time(m["close_time"])
    minutes_left = (close - now).total_seconds() / 60
    target = target_price(m)
    if target is None:
        notify("Kalshi BTC: no target yet", f"{m['ticker']} has no target price yet.")
        return

    price = btc_price()
    c1, c15, c60 = candles(60), candles(900), candles(3600)

    p_model, z, typical_move = price_vs_target_odds(price, target, c1, minutes_left)
    p_market = kalshi_up_odds(m)
    score, tech_lines, rel_vol = technicals(c15, c1, c60, price)
    tilt = max(-MAX_TILT, min(MAX_TILT, score * TILT_PER_POINT))

    base = p_model if p_market is None else MODEL_WEIGHT * p_model + MARKET_WEIGHT * p_market
    p_up = min(max(base + tilt, 0.01), 0.99)
    if p_up > 0.5:
        call = "UP"
    elif p_up < 0.5:
        call = "DOWN"
    else:
        call = "UP" if price >= target else "DOWN"
    odds = p_up if call == "UP" else 1 - p_up
    confidence = "High" if odds >= 0.85 else "Medium" if odds >= 0.65 else "Low (coin flip)"
    favorite = None if p_market is None else ("UP" if p_market >= 0.5 else "DOWN")
    diff = price - target
    too_close = abs(diff) < NEAR_STRIKE_DOLLARS or abs(z) < 0.3

    # Is the called side cheap or expensive on Kalshi?
    ask = kalshi_price(m, "yes_ask" if call == "UP" else "no_ask")
    value_line = "Kalshi price: not available"
    if ask:
        fee = kalshi_fee(ask)
        edge = odds - ask - fee
        verdict = (f"bot sees about +{edge * 100:.0f}¢ value" if edge > VALUE_EDGE
                   else "no real value at this price")
        value_line = (f"{call} costs {ask * 100:.0f}¢ + {fee * 100:.0f}¢ fee; "
                      f"bot odds {odds:.0%} → {verdict}")

    # Paper trade (Ping-style rules: 1 contract, real ask, fee counted, daily loss limit)
    today_pnl = paper_today(rows, et_date(now.isoformat()))
    size = to_float(m.get("yes_ask_size_fp" if call == "UP" else "yes_bid_size_fp"))
    paper_price = paper_fee = None
    if today_pnl <= -DAILY_LOSS_LIMIT:
        paper_note = f"skipped: daily loss limit hit ({today_pnl:+.2f} today)"
    elif not ask or ask >= 1:
        paper_note = "skipped: no price to buy at"
    elif size is not None and size < PAPER_CONTRACTS:
        paper_note = "skipped: no contracts for sale at the ask"
    else:
        paper_price, paper_fee = ask, kalshi_fee(ask, PAPER_CONTRACTS)
        paper_note = (f"bought {PAPER_CONTRACTS} {call} at {ask * 100:.0f}¢ + {paper_fee * 100:.0f}¢ fee "
                      f"(win +{(1 - ask) * PAPER_CONTRACTS - paper_fee:.2f} / lose "
                      f"-{ask * PAPER_CONTRACTS + paper_fee:.2f})")
    value_call = bool(ask) and odds - ask - kalshi_fee(ask) > VALUE_EDGE

    warnings = []
    if today_pnl <= -DAILY_LOSS_LIMIT:
        warnings.append(f"🛑 Daily loss limit hit ({today_pnl:+.2f} today): no more paper trades today.")
    if too_close:
        warnings.append(f"⚠️ Too close to call: BTC is only ${abs(diff):,.0f} from the target. "
                        "Small moves (and Coinbase vs Kalshi's price index) can flip it.")
    if favorite and favorite != call:
        warnings.append("⚠️ The bot disagrees with Kalshi's favorite.")
    if rel_vol is not None and rel_vol < 0.7:
        warnings.append("⚠️ Quiet market: low volume, choppy conditions.")
    if confidence.startswith("Low"):
        warnings.append("⚠️ Low confidence: close to 50/50.")

    body = "\n".join(
        [f"Window closes {local_time(close)} ({minutes_left:.0f} min left)",
         f"Target: ${target:,.2f} | BTC now: ${price:,.2f} ({diff:+,.0f})",
         f"Confidence: {confidence}",
         "",
         "Why:",
         f"• Price vs target: {p_model:.0%} UP (BTC usually moves ~${typical_move:,.0f} in {minutes_left:.0f} min)",
         f"• Kalshi odds: {p_market:.0%} UP" if p_market is not None else "• Kalshi odds: not available",
         f"• Technicals: score {score:+d} → {tilt * 100:+.0f}% nudge"]
        + [f"   - {t}" for t in tech_lines]
        + ["", value_line, f"Paper trade: {paper_note}"]
        + ([""] + warnings if warnings else [])
        + ["", scorecard(rows)]
    )
    title = f"BTC 15m: {call} ({odds:.0%})"
    notify(title, body, "chart_with_upwards_trend" if call == "UP" else "chart_with_downwards_trend")

    rows = read_log()
    if not any(r["ticker"] == m["ticker"] for r in rows):
        rows.append({"ticker": m["ticker"], "close_time_utc": m["close_time"],
                     "sent_at_utc": now.isoformat(timespec="seconds"),
                     "minutes_left": f"{minutes_left:.1f}", "call": call, "bot_odds": f"{odds:.3f}",
                     "confidence": confidence.split(" ")[0], "kalshi_up_odds":
                     "" if p_market is None else f"{p_market:.3f}", "kalshi_favorite": favorite or "",
                     "price_model_up": f"{p_model:.3f}", "tech_score": score, "target": target,
                     "btc_price": price, "too_close": "yes" if too_close else "", "result": "",
                     "paper_price": "" if paper_price is None else f"{paper_price:.4f}",
                     "paper_fee": "" if paper_fee is None else f"{paper_fee:.2f}",
                     "paper_note": paper_note, "value_call": "yes" if value_call else ""})
        write_log(rows)


def next_send_time(now):
    """Windows close at :00, :15, :30 and :45. Send LEAD_MINUTES before the next close."""
    start = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    close = start + timedelta(minutes=15)
    send = close - timedelta(minutes=LEAD_MINUTES)
    while send <= now:
        send += timedelta(minutes=15)
    return send


def main():
    if "--now" in sys.argv:
        make_call()
        return
    started = datetime.now(timezone.utc)
    stop_at = started + timedelta(minutes=MAX_RUNTIME_MINUTES) if MAX_RUNTIME_MINUTES else None
    print(f"Kalshi BTC 15-min predictor running. Calls go out {LEAD_MINUTES} min before each close.")
    automatic = (os.environ.get("GITHUB_EVENT_NAME") == "schedule"
                 or os.environ.get("HANDOFF") == "true")
    if not automatic:                                        # stay quiet on automatic restarts
        notify("Kalshi BTC predictor started",
               f"You'll get an UP/DOWN call {LEAD_MINUTES} minutes before every 15-minute window closes.")
    while True:
        send_at = next_send_time(datetime.now(timezone.utc))
        if stop_at and send_at > stop_at:
            print("Run time is up; the next run takes over.")
            return
        print(f"Next call at {local_time(send_at)}")
        while True:
            left = (send_at - datetime.now(timezone.utc)).total_seconds()
            if left <= 0:
                break
            time.sleep(min(left, 30))
        late = (datetime.now(timezone.utc) - send_at).total_seconds()
        if late > 120:            # the computer was asleep; skip to the next window
            continue
        try:
            make_call()
        except Exception as e:
            notify("Kalshi BTC: call failed", f"Couldn't make this call: {e}")
        time.sleep(5)


if __name__ == "__main__":
    main()
