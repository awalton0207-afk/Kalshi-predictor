"""
Kalshi BTC 15-minute predictor
------------------------------
For every Kalshi "BTC price up in next 15 mins?" window (series KXBTC15M), this
sends an UP or DOWN call to your phone 3 minutes into the window (12 minutes before it closes).

What goes into each call:
  1. Price vs target   - where BTC is now compared with the window's target price,
                         and how far BTC usually moves in the time that's left
  2. Kalshi's own odds - the live YES/NO prices (Kalshi's odds track real results closely)
  3. Technicals        - the signals from the Grok bot: RSI, EMA 9/21, 3-candle momentum,
                         last-5-minute trend, volume spike, 1-hour trend.
                         Added from the kalshi-mcp-bot (rebuilt, not copied): Bollinger Band
                         position, buying vs. selling pressure from live Coinbase trades,
                         and engulfing-candle patterns on the 1-minute chart.
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

  7. Real bets (optional) - when REAL_TRADING is "on" and a Kalshi API key is set,
                         it places a real $5 bet on HIGH-confidence calls only, priced
                         92¢ or less. It stops for the day after losing $10 and stops
                         completely after losing $25 in total. After 3 real losses in a row
                         it skips real bets for the next 2 windows (30 min). Bets are fill-or-kill at the
                         current ask, so it never pays more than the price it saw.

Without a Kalshi API key it only sends predictions. Not financial advice.

Setup: pip install requests cryptography     Run: python kalshi_15m_predictor.py
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

LEAD_MINUTES = 12                 # send the call this many minutes before the window closes (12 = 3 min in)
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
BB_PERIOD, BB_WIDTH = 20, 2       # Bollinger Bands on 1-minute candles
PRESSURE_MINUTES = 5              # buying vs. selling pressure over the last 5 minutes
PRESSURE_LEAN = 0.15              # 15% more buying than selling (or the reverse) counts

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

# Real bets (only when REAL_TRADING is "on" AND both Kalshi key secrets are set)
REAL_TRADING = os.environ.get("REAL_TRADING", "").strip().lower() == "on"
KALSHI_KEY_ID = os.environ.get("KALSHI_KEY_ID", "").strip()
KALSHI_PRIVATE_KEY = os.environ.get("KALSHI_PRIVATE_KEY", "").strip()
REAL_BET_DOLLARS = 5.00           # most a single bet can cost, fee included
REAL_MAX_PRICE = 0.92             # skip calls priced above 92¢ (payout too small)
REAL_DAILY_LOSS_LIMIT = 10.00     # no more real bets today after losing this much
REAL_TOTAL_LOSS_LIMIT = 25.00     # no more real bets at all after losing this much in total
REAL_LOSS_STREAK = 3              # after this many real losses in a row...
REAL_STREAK_PAUSE_MIN = 30        # ...skip real bets for this many minutes (2 windows)

# How long one cloud run lasts before it hands off to the next one (GitHub allows 6 hours)
MAX_RUNTIME_MINUTES = float(os.environ.get("MAX_RUNTIME_MINUTES", "0"))   # 0 = run forever
# ========================================================

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(HERE, "kalshi_predictions.csv")
LOG_FIELDS = ["ticker", "close_time_utc", "sent_at_utc", "minutes_left", "call", "bot_odds",
              "confidence", "kalshi_up_odds", "kalshi_favorite", "price_model_up", "tech_score",
              "target", "btc_price", "too_close", "result",
              "paper_price", "paper_fee", "paper_note", "value_call", "paper_pnl",
              "real_count", "real_cost", "real_note", "real_pnl"]

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


def bollinger_position(closes, period=BB_PERIOD, width=BB_WIDTH):
    """0 = on the lower band, 0.5 = middle, 1 = on the upper band (can go past 0 or 1)."""
    if len(closes) < period:
        return None
    w = closes[-period:]
    mid = sum(w) / period
    sd = statistics.pstdev(w)
    if sd == 0:
        return 0.5
    return (closes[-1] - (mid - width * sd)) / (2 * width * sd)


def trade_pressure(minutes=PRESSURE_MINUTES):
    """-1 = all selling, +1 = all buying, from Coinbase's recent trades (by size).
    Coinbase's 'side' is the resting order's side, so side 'sell' = a buyer took the ask."""
    try:
        trades = get_json(f"https://api.exchange.coinbase.com/products/{COINBASE_PRODUCT}/trades",
                          {"limit": 1000})
    except Exception as e:
        print("Coinbase trades failed:", e)
        return None
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    buy = sell = 0.0
    for t in trades:
        try:
            if parse_time(t["time"]) < cutoff:
                continue
            size = float(t["size"])
        except Exception:
            continue
        if t.get("side") == "sell":
            buy += size
        elif t.get("side") == "buy":
            sell += size
    total = buy + sell
    return None if total == 0 else (buy - sell) / total


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

    # 7. Bollinger Band position (1-minute candles, live price)
    c1_closes = [x["close"] for x in c1]
    if c1_closes:
        c1_closes[-1] = price
    bp = bollinger_position(c1_closes)
    if bp is not None:
        if bp > 1.05:
            score -= 1
            lines.append(f"Bollinger: above the top band ({bp:.2f}), stretched (leans DOWN)")
        elif bp < -0.05:
            score += 1
            lines.append(f"Bollinger: below the bottom band ({bp:.2f}), stretched (leans UP)")
        elif bp > 0.65:
            score += 1
            lines.append(f"Bollinger: riding the upper half ({bp:.2f}) (leans UP)")
        elif bp < 0.35:
            score -= 1
            lines.append(f"Bollinger: riding the lower half ({bp:.2f}) (leans DOWN)")
        else:
            lines.append(f"Bollinger: middle ({bp:.2f})")

    # 8. Buying vs. selling pressure (live Coinbase trades, last 5 minutes)
    pr = trade_pressure()
    if pr is not None:
        if pr > PRESSURE_LEAN:
            score += 1
            lines.append(f"Trade pressure {pr:+.2f}: more buying (leans UP)")
        elif pr < -PRESSURE_LEAN:
            score -= 1
            lines.append(f"Trade pressure {pr:+.2f}: more selling (leans DOWN)")
        else:
            lines.append(f"Trade pressure {pr:+.2f}: balanced")

    # 9. Engulfing candle on the last two finished 1-minute candles
    if len(c1) >= 3:
        a, b = c1[-3], c1[-2]
        if a["close"] < a["open"] and b["close"] > b["open"] and b["close"] >= a["open"] and b["open"] <= a["close"]:
            score += 1
            lines.append("1-min candles: bullish engulfing (leans UP)")
        elif a["close"] > a["open"] and b["close"] < b["open"] and b["close"] <= a["open"] and b["open"] >= a["close"]:
            score -= 1
            lines.append("1-min candles: bearish engulfing (leans DOWN)")

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
            count, cost = to_float(row.get("real_count")), to_float(row.get("real_cost"))
            if count and cost is not None:
                won = row["call"] == row["result"]
                row["real_pnl"] = f"{(count if won else 0) - cost:.2f}"
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

    out = ["", f"Paper trades ({PAPER_CONTRACTS} contract, fees included, $):",
           line("Every call", traded),
           line("Only value calls", [r for r in traded if r.get("value_call") == "yes"])]
    real = [r for r in rows if r.get("real_pnl")]
    if real:
        out += ["", "REAL money ($5 high-confidence bets, fees included):", line("Real bets", real)]
    return out


# --------------------- real bets ------------------------
def real_pnl_since(rows, today=None):
    return sum(to_float(r.get("real_pnl")) or 0 for r in rows if r.get("real_pnl")
               and (today is None or et_date(r["sent_at_utc"]) == today))


def kalshi_signed(method, path, body=None):
    """Signed request to Kalshi's trading API (RSA-PSS signature, as Kalshi requires)."""
    import base64
    import json
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    key = serialization.load_pem_private_key(KALSHI_PRIVATE_KEY.replace("\\n", "\n").encode(),
                                             password=None)
    ts = str(int(time.time() * 1000))
    full_path = "/trade-api/v2" + path
    sig = key.sign((ts + method + full_path).encode(),
                   padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                               salt_length=padding.PSS.DIGEST_LENGTH),
                   hashes.SHA256())
    headers = {"KALSHI-ACCESS-KEY": KALSHI_KEY_ID, "KALSHI-ACCESS-TIMESTAMP": ts,
               "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
               "Content-Type": "application/json"}
    r = session.request(method, KALSHI + path, headers=headers,
                        data=json.dumps(body) if body is not None else None, timeout=15)
    if r.status_code >= 400:
        raise RuntimeError(f"Kalshi said {r.status_code}: {r.text[:200]}")
    return r.json()


def real_ready():
    return REAL_TRADING and KALSHI_KEY_ID and KALSHI_PRIVATE_KEY


def kalshi_balance():
    b = kalshi_signed("GET", "/portfolio/balance")
    if b.get("balance_dollars") not in (None, ""):
        return float(b["balance_dollars"])
    return float(b.get("balance", 0)) / 100


def order_number(o, name, cents=False):
    for k in (f"{name}_dollars", f"{name}_fp", name):
        v = o.get(k)
        if v not in (None, ""):
            v = float(v)
            return v / 100 if cents and k == name else v
    return 0.0


def place_real_bet(rows, m, call, confidence, ask, now):
    """Returns (count, cost, note). count/cost are None when no bet was placed."""
    if not real_ready():
        return None, None, ""
    today_real = real_pnl_since(rows, et_date(now.isoformat()))
    total_real = real_pnl_since(rows)
    if total_real <= -REAL_TOTAL_LOSS_LIMIT:
        return None, None, (f"🛑 stopped: total real loss limit hit ({total_real:+.2f}). "
                            "No more real bets until the limit is reset.")
    if today_real <= -REAL_DAILY_LOSS_LIMIT:
        return None, None, f"🛑 skipped: daily real loss limit hit ({today_real:+.2f} today)"
    settled = [r for r in rows if r.get("real_pnl")]
    last = settled[-REAL_LOSS_STREAK:]
    if len(last) == REAL_LOSS_STREAK and all(float(r["real_pnl"]) < 0 for r in last):
        resume = parse_time(last[-1]["close_time_utc"]) + timedelta(minutes=REAL_STREAK_PAUSE_MIN)
        if now < resume:
            return None, None, (f"⏸️ skipped: {REAL_LOSS_STREAK} real losses in a row, "
                                f"real bets resume after {local_time(resume)}")
    if confidence != "High":
        return None, None, "no bet: only high-confidence calls are bet"
    if not ask or ask >= 1:
        return None, None, "no bet: no price to buy at"
    if ask > REAL_MAX_PRICE:
        return None, None, f"no bet: {ask * 100:.0f}¢ is above the {REAL_MAX_PRICE * 100:.0f}¢ cap"
    count = 0
    while kalshi_fee(ask, count + 1) + ask * (count + 1) <= REAL_BET_DOLLARS:
        count += 1
    if count < 1:
        return None, None, "no bet: one contract costs more than the bet size"
    cents = int(round(ask * 100))
    side = "yes" if call == "UP" else "no"
    body = {"ticker": m["ticker"], "action": "buy", "side": side, "count": count,
            "type": "limit", f"{side}_price": cents, "time_in_force": "fill_or_kill",
            "client_order_id": f"pred-{m['ticker']}-{int(time.time())}"}
    try:
        o = kalshi_signed("POST", "/portfolio/orders", body).get("order", {})
    except Exception as e:
        return None, None, f"⚠️ real bet failed: {e}"
    filled = order_number(o, "fill_count")
    if filled < 1:
        return None, None, f"real bet not filled (price moved above {cents}¢)"
    cost = (order_number(o, "taker_fill_cost", cents=True) + order_number(o, "maker_fill_cost", cents=True)
            + order_number(o, "taker_fees", cents=True) + order_number(o, "maker_fees", cents=True))
    if cost <= 0:                  # Kalshi didn't report the cost; estimate it
        cost = ask * filled + kalshi_fee(ask, int(filled))
    return int(filled), round(cost, 2), (f"💵 REAL bet: bought {int(filled)} {call} at {cents}¢ "
                                       f"for ${cost:.2f} (win +{filled - cost:.2f} / lose -{cost:.2f})")


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

    # Real bet ($5, high-confidence only) - only if REAL_TRADING is on and keys are set
    already = any(r["ticker"] == m["ticker"] for r in rows)
    real_count, real_cost, real_note = (None, None, "") if already else \
        place_real_bet(rows, m, call, confidence.split(" ")[0], ask, now)

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
        + ([real_note] if real_note else [])
        + ([""] + warnings if warnings else [])
        + ["", scorecard(rows)]
    )
    title = f"BTC 15m: {call} ({odds:.0%})" + (" 💵 REAL $ bet placed" if real_count else "")
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
                     "paper_note": paper_note, "value_call": "yes" if value_call else "",
                     "real_count": real_count or "", "real_cost": "" if real_cost is None else f"{real_cost:.2f}",
                     "real_note": real_note})
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
    real_status = "Real betting: OFF (predictions and paper trades only)."
    if REAL_TRADING and not (KALSHI_KEY_ID and KALSHI_PRIVATE_KEY):
        real_status = "Real betting: OFF. REAL_TRADING is on, but the Kalshi key secrets are missing."
    elif real_ready():
        try:
            real_status = (f"💵 Real betting: ON. Kalshi balance ${kalshi_balance():,.2f}. "
                           f"${REAL_BET_DOLLARS:.0f} bets on high-confidence calls up to "
                           f"{REAL_MAX_PRICE * 100:.0f}¢; stops after -${REAL_DAILY_LOSS_LIMIT:.0f} in a day "
                           f"or -${REAL_TOTAL_LOSS_LIMIT:.0f} in total.")
        except Exception as e:
            real_status = f"⚠️ Real betting is on, but Kalshi rejected the API key: {e}"
    print(real_status)
    if not automatic:                                        # stay quiet on automatic restarts
        notify("Kalshi BTC predictor started",
               f"You'll get an UP/DOWN call {15 - LEAD_MINUTES} minutes into every 15-minute window ({LEAD_MINUTES} min before it closes).\n"
               + real_status)
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
