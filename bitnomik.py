"""
 bitNOMIK live signal monitor for Bybit through CCXT Pro.

Install:
    pip install "ccxt[pro]" requests

Required secrets/environment:
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHANNEL_ID              # @channel_name or numeric chat id

Optional configuration:
    BITNOMIK_PAIRS=BTC/USDT,ETH/USDT,LTC/USDT
    BITNOMIK_SIGNAL_COOLDOWN=1800
    BITNOMIK_CHANNEL_GAP=480
    BITNOMIK_SETUP_WINDOW=75
    BITNOMIK_MIN_SCORE=85
    BITNOMIK_REFERRAL_LINK=https://...
    BITNOMIK_HTTP_TIMEOUT_MS=30000
    BITNOMIK_PROXY=https://user:pass@host:port
    BITNOMIK_FUNDING_SYMBOL_SUFFIX=:USDT   # appended to base pair to watch the linear swap market
    BITNOMIK_TA_PERIOD_1=20                # short moving-average / support-resistance lookback
    BITNOMIK_TA_PERIOD_2=50                # long moving-average / support-resistance lookback
    BITNOMIK_TA_INTERVAL_SECONDS=3600      # min seconds between TA snapshots for the same pair
    BITNOMIK_TA_CANDLE_LIMIT=200           # candles kept in memory per pair/timeframe

This script publishes educational market alerts only. It does not place trades.
Nothing in this output is financial advice.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import ccxt.pro as ccxtpro
import requests
from ccxt.base.errors import NetworkError


def load_env_file(path: str | None = None) -> None:
    """Load KEY=VALUE pairs from a local .env file, for local development only.

    Never overrides a variable that's already set in the real environment (so
    Railway's injected env vars always win), and this file never contains any
    hardcoded secret itself — it only reads whatever the .env file on disk has.
    Silently does nothing if no .env file is present (e.g. on Railway).
    """
    env_path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()


DEFAULT_PAIRS = os.getenv("DEFAULT_PAIRS","BTC/USDT,ETH/USDT")

ORDER_BOOK_LEVELS = 10
IMBALANCE_THRESHOLD = 2.50
WALL_MULTIPLIER = 8.0
SPREAD_WIDENING_MULTIPLIER = 2.50
WHALE_MIN_NOTIONAL = 400_000.0
WHALE_STANDALONE_NOTIONAL = 1_000_000.0
VOLUME_SPIKE_MULTIPLIER = 6.0
RAPID_MOVE_PERCENT = 1.25
RAPID_MOVE_STANDALONE_PERCENT = 2.00
EXTREME_FUNDING_RATE = 0.0008  # 0.08% per funding interval
PRESSURE_SHARE = 0.78
BULL = "bull"
BEAR = "bear"
NEUTRAL = "neutral"
FUNDING_WARNED: set[str] = set()
DEFAULT_REFERRAL_LINK = "https://www.binance.com/register?ref=BITNOMIK"
SEPARATOR = "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _"
DISCLAIMER = "Educational market analysis only. Not financial advice. Always manage your risk."


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


WALL_MIN_NOTIONAL = env_float("BITNOMIK_WALL_MIN_NOTIONAL", 100_000)
PRESSURE_MIN_NOTIONAL = env_float("BITNOMIK_PRESSURE_MIN_NOTIONAL", 80_000)
SETUP_WINDOW_SECONDS = env_int("BITNOMIK_SETUP_WINDOW", 75)
CHANNEL_GAP_SECONDS = env_int("BITNOMIK_CHANNEL_GAP", 480)
MIN_SETUP_SCORE = env_float("BITNOMIK_MIN_SCORE", 85)
STANDALONE_SCORE = env_float("BITNOMIK_STANDALONE_SCORE", 100)
FUNDING_SYMBOL_SUFFIX = os.getenv("BITNOMIK_FUNDING_SYMBOL_SUFFIX", ":USDT").strip()
TA_PERIOD_1 = env_int("BITNOMIK_TA_PERIOD_1", 20)
TA_PERIOD_2 = env_int("BITNOMIK_TA_PERIOD_2", 50)
TA_INTERVAL_SECONDS = env_int("BITNOMIK_TA_INTERVAL_SECONDS", 3600)
TA_CANDLE_LIMIT = env_int("BITNOMIK_TA_CANDLE_LIMIT", 200)
TA_PIVOT_SPAN = env_int("BITNOMIK_TA_PIVOT_SPAN", 2)
TA_CLUSTER_TOLERANCE_PCT = env_float("BITNOMIK_TA_CLUSTER_TOLERANCE_PCT", 0.25) / 100


def usd_notional(level_price: float, amount: float) -> float:
    return abs(level_price * amount)


def two_decimal_places(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 2)


def moving_average(candles: list[list[float]], period: int) -> float | None:
    """Simple moving average of close prices over the last `period` candles.

    Each candle is a CCXT OHLCV row: [timestamp, open, high, low, close, volume].
    """
    if len(candles) < period or period <= 0:
        return None
    closes = [candle[4] for candle in candles[-period:]]
    return statistics.mean(closes)


def find_pivots(
    candles: list[list[float]], span: int = 2
) -> tuple[list[float], list[float]]:
    """Detect swing highs and swing lows within `candles`.

    A candle at index i is a pivot high if its high is the maximum among the
    `span` candles on either side of it (and similarly a pivot low for the minimum
    low). This is the standard fractal/swing-point definition used in most charting
    tools, rather than just taking the raw high/low of the whole window.
    """
    pivot_highs: list[float] = []
    pivot_lows: list[float] = []
    count = len(candles)
    for i in range(span, count - span):
        window = candles[i - span : i + span + 1]
        high = candles[i][2]
        low = candles[i][3]
        if high == max(candle[2] for candle in window):
            pivot_highs.append(high)
        if low == min(candle[3] for candle in window):
            pivot_lows.append(low)
    return pivot_highs, pivot_lows


def cluster_levels(levels: list[float], tolerance_pct: float) -> list[tuple[float, int]]:
    """Group nearby price levels into zones.

    Real support/resistance is rarely a single exact price — price tends to react
    within a band. This merges pivots that fall within `tolerance_pct` of each other
    into one zone, averaging their prices and counting how many times price pivoted
    there (more touches = a more significant zone). Returns (zone_price, touch_count).
    """
    if not levels:
        return []
    ordered = sorted(levels)
    clusters: list[list[float]] = [[ordered[0]]]
    for level in ordered[1:]:
        if abs(level - clusters[-1][-1]) / clusters[-1][-1] <= tolerance_pct:
            clusters[-1].append(level)
        else:
            clusters.append([level])
    return [(statistics.mean(cluster), len(cluster)) for cluster in clusters]


def support_resistance(
    candles: list[list[float]],
    period: int,
    live_price: float | None = None,
    pivot_span: int = 2,
    cluster_tolerance_pct: float = 0.0025,
) -> tuple[float | None, float | None]:
    """Pivot-based support/resistance.

    1. Restrict to the last `period` candles.
    2. Detect swing highs/lows (fractal pivots).
    3. Cluster nearby pivots into zones, weighted by how many times price touched them.
    4. Return the nearest resistance zone above the current price and nearest support
       zone below it, preferring zones with more touches when several are similarly close.

    Falls back to a simple rolling low/high if there isn't enough candle history or
    pivot structure to work with (e.g. a very flat or very short window).
    """
    if len(candles) < period or period <= 0:
        return None, None
    window = candles[-period:]
    reference_price = live_price if live_price is not None else window[-1][4]

    pivot_highs, pivot_lows = find_pivots(window, span=pivot_span)
    lows = [candle[3] for candle in window]
    highs = [candle[2] for candle in window]

    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return min(lows), max(highs)

    resistance_zones = cluster_levels(pivot_highs, cluster_tolerance_pct)
    support_zones = cluster_levels(pivot_lows, cluster_tolerance_pct)

    def nearest_zone(zones: list[tuple[float, int]], predicate) -> float | None:
        candidates = [zone for zone in zones if predicate(zone[0])]
        if not candidates:
            return None
        candidates.sort(key=lambda zone: (abs(zone[0] - reference_price), -zone[1]))
        return candidates[0][0]

    resistance = nearest_zone(resistance_zones, lambda price_level: price_level > reference_price)
    support = nearest_zone(support_zones, lambda price_level: price_level < reference_price)

    if resistance is None:
        resistance = max(highs)
    if support is None:
        support = min(lows)

    return support, resistance


def is_qualifying_wall(level_price: float, amount: float, average_level_size: float) -> bool:
    if average_level_size <= 0 or amount <= 0:
        return False
    return (
        amount >= average_level_size * WALL_MULTIPLIER
        and usd_notional(level_price, amount) >= WALL_MIN_NOTIONAL
    )


def now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def number(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.{decimals}f}"


def price(value: Any) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    decimals = 2 if value >= 1 else 6
    return f"{value:,.{decimals}f}"


def extract_timestamp(value: dict[str, Any]) -> str:
    timestamp = value.get("datetime")
    if timestamp:
        return str(timestamp)
    return now_text()


def parse_clock(observed_at: str | None) -> tuple[str, str]:
    raw = observed_at or now_text()
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.now(timezone.utc)
    return parsed.strftime("%Y-%m-%d"), parsed.strftime("%H:%M")


def referral_footer(referral_link: str = "") -> str:
    link = referral_link or DEFAULT_REFERRAL_LINK
    if random.randint(1, 100) > 65:
        return f"🔗 **GET EXCLUSIVE Binance DISCOUNT ON YOUR FIRST TRANSACTION**\n\n{link}"
    return ""


def compose_desk_message(
    title: str,
    pair: str,
    observed_at: str | None,
    live_price: Any,
    body_lines: list[str],
    referral_link: str,
) -> str:
    date_str, time_str = parse_clock(observed_at)
    footer = referral_footer(referral_link)
    lines = [
        "Welcome to bitNOMIK Signal Monitor\n",
        f"🔥 Crypto MARKET DESK | {title} for {pair} 📊",
        SEPARATOR,
        f"Asset: {pair}",
        f"Time: {date_str} at {time_str}",
        "",
        f"💰 **Live Price:** ${price(live_price)}",
        "",
        *body_lines,
    ]
    if footer:
        lines += ["", footer]
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def conviction_label(score: float) -> str:
    if score >= 130:
        return "EXTREME"
    if score >= 100:
        return "HIGH"
    return "ELEVATED"


def build_conviction_message(
    pair: str,
    live_price: Any,
    direction: str,
    score: float,
    summaries: list[str],
    observed_at: str | None,
    referral_link: str,
) -> str:
    if direction == BULL:
        bias = "🟢 BULLISH"
        title = "HIGH CONVICTION LONG SETUP"
    elif direction == BEAR:
        bias = "🔴 BEARISH"
        title = "HIGH CONVICTION SHORT SETUP"
    else:
        bias = "🟡 MIXED"
        title = "HIGH CONVICTION SETUP"
    body = [
        f"📌 **Conviction:** {conviction_label(score)}",
        f"📌 **Bias:** {bias}",
        f"📌 **Confirming signals:** {len(summaries)}",
        "",
        "Why it matters:",
        *[f"   • {line}" for line in summaries],
    ]
    return compose_desk_message(title, pair, observed_at, live_price, body, referral_link)


def build_imbalance_message(
    pair: str,
    live_price: Any,
    bid_volume: float,
    ask_volume: float,
    ratio: float,
    observed_at: str | None,
    referral_link: str,
) -> str:
    if ratio >= IMBALANCE_THRESHOLD:
        signal = f"🟢 Signal: Buyers outweigh sellers (buyers / sellers: {ratio:.2f}x)"
    elif ratio <= 1 / IMBALANCE_THRESHOLD:
        signal = f"🔴 Signal: Sellers outweigh buyers (buyers / sellers: {ratio:.2f}x)"
    else:
        signal = f"🟡 Signal: Market is balanced (buyers / sellers: {ratio:.2f}x)"

    return compose_desk_message(
        "ORDER BOOK BREAKDOWN",
        pair,
        observed_at,
        live_price,
        [
            f"🎯 **Top {ORDER_BOOK_LEVELS} Bids and Asks:**",
            f"   • Bid Volume: {number(bid_volume, 2)}",
            f"   • Ask Volume: {number(ask_volume, 2)}",
            f"   • Ratio: {ratio:.2f}x",
            "",
            signal,
        ],
        referral_link,
    )


def build_wall_message(
    pair: str,
    live_price: Any,
    side: str,
    wall_price: float,
    wall_size: float,
    average_level_size: float,
    observed_at: str | None,
    referral_link: str,
) -> str:
    multiple = wall_size / average_level_size if average_level_size > 0 else 0
    if side == "bid":
        signal = "🟢 Signal: Support wall in the book"
        label = "BID (support)"
    else:
        signal = "🔴 Signal: Resistance wall in the book"
        label = "ASK (resistance)"

    return compose_desk_message(
        "ORDER BOOK WALL",
        pair,
        observed_at,
        live_price,
        [
            "🧱 **Order Book Wall:**",
            f"   • Side: {label}",
            f"   • Price: ${price(wall_price)}",
            f"   • Size: {number(wall_size, 4)} base units",
            f"   • vs typical level: {multiple:.1f}x",
            "",
            signal,
        ],
        referral_link,
    )


def build_spread_message(
    pair: str,
    live_price: Any,
    spread_bps: float,
    baseline_bps: float,
    observed_at: str | None,
    referral_link: str,
) -> str:
    expansion = spread_bps / baseline_bps if baseline_bps > 0 else 0
    return compose_desk_message(
        "SPREAD WIDENING",
        pair,
        observed_at,
        live_price,
        [
            "⚠️ **Spread Alert:**",
            f"   • Current spread: {spread_bps:.2f} bps",
            f"   • Baseline: {baseline_bps:.2f} bps",
            f"   • Expansion: {expansion:.2f}x",
            "",
            "🟡 Signal: Liquidity is thinning — trade with caution",
        ],
        referral_link,
    )


def build_wall_removal_message(
    pair: str,
    live_price: Any,
    side: str,
    wall_price: float,
    previous_amount: float,
    remaining: float,
    observed_at: str | None,
    referral_link: str,
) -> str:
    if side == "bid":
        signal = "🔴 Signal: Support removed"
        label = "BID (support)"
    else:
        signal = "🟢 Signal: Resistance removed"
        label = "ASK (resistance)"

    return compose_desk_message(
        "WALL REMOVAL",
        pair,
        observed_at,
        live_price,
        [
            "🧱 **Wall Removal:**",
            f"   • Side: {label}",
            f"   • Level: ${price(wall_price)}",
            f"   • Size fell: {number(previous_amount, 4)} → {number(remaining, 4)} base units",
            "",
            signal,
        ],
        referral_link,
    )


def build_whale_message(
    pair: str,
    trade_price: float,
    notional: float,
    side: str | None,
    observed_at: str | None,
    referral_link: str,
) -> str:
    if side == "buy":
        signal = "🟢 Signal: Whale buying"
        side_label = "BUY"
    elif side == "sell":
        signal = "🔴 Signal: Whale selling"
        side_label = "SELL"
    else:
        signal = "🟡 Signal: Large trade printed"
        side_label = "UNKNOWN"

    return compose_desk_message(
        "WHALE TRADE",
        pair,
        observed_at,
        trade_price,
        [
            "🐋 **Large Trade:**",
            f"   • Size: ${number(notional, 0)} USDT",
            f"   • Fill price: ${price(trade_price)}",
            f"   • Side: {side_label}",
            "",
            signal,
        ],
        referral_link,
    )


def build_pressure_message(
    pair: str,
    live_price: Any,
    buy_share: float,
    observed_at: str | None,
    referral_link: str,
) -> str:
    sell_share = 1 - buy_share
    signal = "🟢 Signal: Buyers dominating the tape" if buy_share >= 0.65 else "🔴 Signal: Sellers dominating the tape"

    return compose_desk_message(
        "BUY/SELL PRESSURE",
        pair,
        observed_at,
        live_price,
        [
            "⚖️ **Flow:**",
            f"   • Buy share: {buy_share:.0%}",
            f"   • Sell share: {sell_share:.0%}",
            "",
            signal,
        ],
        referral_link,
    )


def build_volume_spike_message(
    pair: str,
    live_price: Any,
    batch_volume: float,
    baseline: float,
    observed_at: str | None,
    referral_link: str,
) -> str:
    multiple = batch_volume / baseline if baseline > 0 else 0
    return compose_desk_message(
        "VOLUME SPIKE",
        pair,
        observed_at,
        live_price,
        [
            "📈 **Volume Spike:**",
            f"   • Batch volume: {number(batch_volume, 4)}",
            f"   • Average: {number(baseline, 4)}",
            f"   • Multiple: {multiple:.1f}x",
            "",
            "🟡 Signal: Unusual activity",
        ],
        referral_link,
    )


def build_rapid_move_message(
    pair: str,
    previous_price: float,
    last_price: float,
    move_percent: float,
    observed_at: str | None,
    referral_link: str,
) -> str:
    signal = "🟢 Signal: Sharp move UP" if move_percent > 0 else "🔴 Signal: Sharp move DOWN"

    return compose_desk_message(
        "RAPID PRICE MOVE",
        pair,
        observed_at,
        last_price,
        [
            "⚡ **Rapid Price Move:**",
            f"   • Change: {move_percent:+.2f}%",
            f"   • From: ${price(previous_price)}",
            f"   • To: ${price(last_price)}",
            "",
            signal,
        ],
        referral_link,
    )


def build_high_low_break_message(
    pair: str,
    last_price: float,
    high: Any,
    low: Any,
    break_side: str,
    observed_at: str | None,
    referral_link: str,
) -> str:
    if break_side == "high":
        signal = "🟢 Signal: Breaking the 24h high"
        title = "24H HIGH BREAK"
    else:
        signal = "🔴 Signal: Breaking the 24h low"
        title = "24H LOW BREAK"

    return compose_desk_message(
        title,
        pair,
        observed_at,
        last_price,
        [
            "🏁 **24h High/Low Break:**",
            f"   • Live price: ${price(last_price)}",
            f"   • 24h high: ${price(high)}",
            f"   • 24h low: ${price(low)}",
            "",
            signal,
        ],
        referral_link,
    )


def build_funding_message(
    pair: str,
    live_price: Any,
    rate: float,
    next_funding: Any,
    observed_at: str | None,
    referral_link: str,
) -> str:
    signal = "🔴 Signal: Crowded longs" if rate > 0 else "🟢 Signal: Crowded shorts"

    return compose_desk_message(
        "EXTREME FUNDING RATE",
        pair,
        observed_at,
        live_price,
        [
            "💸 **Funding:**",
            f"   • Funding rate: {rate:.4%}",
            f"   • Next funding: {next_funding}",
            "",
            signal,
        ],
        referral_link,
    )


def build_message_for_pair_ticker(
    pair: str,
    live_price: Any,
    observed_at: str | None,
    candles_4h: list[list[float]],
    candles_1h: list[list[float]],
    period_1: int,
    period_2: int,
    referral_link: str,
) -> str | None:
    """Build a periodic technical-analysis snapshot (moving averages + support/resistance).

    Returns None when there isn't enough candle history yet for either period, so the
    caller can simply skip posting rather than sending a near-empty message.
    """
    needed = max(period_1, period_2)
    if len(candles_4h) < needed or len(candles_1h) < needed:
        return None

    date_str, time_str = parse_clock(observed_at)

    ma_4h_p1 = two_decimal_places(moving_average(candles_4h, period_1))
    ma_4h_p2 = two_decimal_places(moving_average(candles_4h, period_2))
    ma_1h_p1 = two_decimal_places(moving_average(candles_1h, period_1))
    ma_1h_p2 = two_decimal_places(moving_average(candles_1h, period_2))

    support_4h_p1, resistance_4h_p1 = support_resistance(
        candles_4h, period_1, live_price, TA_PIVOT_SPAN, TA_CLUSTER_TOLERANCE_PCT
    )
    support_4h_p2, resistance_4h_p2 = support_resistance(
        candles_4h, period_2, live_price, TA_PIVOT_SPAN, TA_CLUSTER_TOLERANCE_PCT
    )
    support_1h_p1, resistance_1h_p1 = support_resistance(
        candles_1h, period_1, live_price, TA_PIVOT_SPAN, TA_CLUSTER_TOLERANCE_PCT
    )
    support_1h_p2, resistance_1h_p2 = support_resistance(
        candles_1h, period_2, live_price, TA_PIVOT_SPAN, TA_CLUSTER_TOLERANCE_PCT
    )

    footer = referral_footer(referral_link)
    lines = [
        f"🔥 Crypto MARKET DESK | TICKER & INDICATORS for {pair} 📊",
        SEPARATOR,
        f"Time: {date_str} at {time_str}",
        "",
        f"💰 **Live Price:** ${price(live_price)}",
        "",
        "📈 **4H Moving Averages:**",
        f"   • MA({period_1}): ${number(ma_4h_p1)}",
        f"   • MA({period_2}): ${number(ma_4h_p2)}",
        "",
        "📉 **1H Moving Averages:**",
        f"   • MA({period_1}): ${number(ma_1h_p1)}",
        f"   • MA({period_2}): ${number(ma_1h_p2)}",
        "",
        "📊 **4H Support/Resistance:**",
        f"   • Support {period_1} candles: ${number(support_4h_p1)}",
        f"   • Resistance {period_1} candles: ${number(resistance_4h_p1)}",
        f"   • Support {period_2} candles: ${number(support_4h_p2)}",
        f"   • Resistance {period_2} candles: ${number(resistance_4h_p2)}",
        "",
        "📊 **1H Support/Resistance:**",
        f"   • Support {period_1} candles: ${number(support_1h_p1)}",
        f"   • Resistance {period_1} candles: ${number(resistance_1h_p1)}",
        f"   • Support {period_2} candles: ${number(support_1h_p2)}",
        f"   • Resistance {period_2} candles: ${number(resistance_1h_p2)}",
    ]
    if footer:
        lines += ["", footer]
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


@dataclass
class SignalEvent:
    signal_id: int
    direction: str
    score: float
    summary: str
    message: str
    standalone: bool = False


@dataclass
class PairState:
    last_price: float | None = None
    previous_24h_high: float | None = None
    previous_24h_low: float | None = None
    spreads_bps: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    trade_volumes: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    qualifying_bid_wall: tuple[float, float] | None = None
    qualifying_ask_wall: tuple[float, float] | None = None
    setup_events: list[SignalEvent] = field(default_factory=list)
    setup_opened_at: float = 0.0
    last_pair_post_at: float = 0.0
    candles_4h: deque[list[float]] = field(default_factory=lambda: deque(maxlen=TA_CANDLE_LIMIT))
    candles_1h: deque[list[float]] = field(default_factory=lambda: deque(maxlen=TA_CANDLE_LIMIT))
    last_ta_post_at: float = 0.0


class TelegramPublisher:
    def __init__(self, token: str, channel_id: str, referral_link: str = ""):
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.channel_id = channel_id
        self.referral_link = referral_link

    async def send(self, message: str) -> None:
        payload = {
            "chat_id": self.channel_id,
            "text": message,
            "disable_web_page_preview": True,
        }

        def post() -> None:
            last_response: requests.Response | None = None
            for _ in range(4):
                last_response = requests.post(self.url, data=payload, timeout=15)
                if last_response.status_code == 429:
                    retry_after = last_response.headers.get("Retry-After", "5")
                    try:
                        wait = min(max(float(retry_after), 1), 30)
                    except ValueError:
                        wait = 5
                    time.sleep(wait)
                    continue
                if last_response.ok:
                    return
                raise RuntimeError(
                    f"Telegram HTTP {last_response.status_code}: {last_response.reason}"
                )
            raise RuntimeError("Telegram rate limit: too many messages")

        await asyncio.to_thread(post)


class SignalMonitor:
    def __init__(
        self,
        publisher: TelegramPublisher | None,
        cooldown_seconds: int,
        referral_link: str = "",
        channel_gap_seconds: int = CHANNEL_GAP_SECONDS,
        setup_window_seconds: int = SETUP_WINDOW_SECONDS,
    ):
        self.publisher = publisher
        self.cooldown_seconds = cooldown_seconds
        self.channel_gap_seconds = channel_gap_seconds
        self.setup_window_seconds = setup_window_seconds
        self.referral_link = referral_link or DEFAULT_REFERRAL_LINK
        self.states: dict[str, PairState] = defaultdict(PairState)
        self.lock = asyncio.Lock()
        self.last_channel_post_at = 0.0

    def live_price(self, pair: str, fallback: Any = None) -> Any:
        state = self.states[pair]
        if state.last_price is not None:
            return state.last_price
        return fallback

    def _combined_setup(self, events: list[SignalEvent]) -> tuple[float, str, list[SignalEvent]]:
        bull = [event for event in events if event.direction == BULL]
        bear = [event for event in events if event.direction == BEAR]
        neutral = [event for event in events if event.direction == NEUTRAL]
        bull_score = sum(event.score for event in bull)
        bear_score = sum(event.score for event in bear)
        if bull_score == 0 and bear_score == 0:
            return 0.0, NEUTRAL, events
        if bull_score >= bear_score:
            chosen, other, direction = bull, bear, BULL
        else:
            chosen, other, direction = bear, bull, BEAR
        score = max(event.score for event in chosen)
        if len(chosen) >= 2:
            score += 28 * (len(chosen) - 1)
        score += 8 * len(neutral)
        score -= 0.4 * sum(event.score for event in other)
        return score, direction, chosen + neutral

    def _publishable(self, score: float, chosen: list[SignalEvent]) -> bool:
        if any(event.standalone for event in chosen):
            return score >= 70
        if score < MIN_SETUP_SCORE:
            return False
        directed = [event for event in chosen if event.direction in {BULL, BEAR}]
        return len(directed) >= 2

    async def consider(
        self,
        pair: str,
        signal_id: int,
        score: float,
        direction: str,
        summary: str,
        message: str,
        standalone: bool = False,
    ) -> None:
        if score < 35:
            return
        await self.flush()
        async with self.lock:
            state = self.states[pair]
            now = time.monotonic()
            window_open = (
                not state.setup_events
                or now - state.setup_opened_at < self.setup_window_seconds
            )
            if not window_open:
                return
            if not state.setup_events:
                state.setup_opened_at = now
            state.setup_events = [
                event for event in state.setup_events if event.signal_id != signal_id
            ]
            state.setup_events.append(
                SignalEvent(
                    signal_id=signal_id,
                    direction=direction,
                    score=score,
                    summary=summary,
                    message=message,
                    standalone=standalone,
                )
            )
        await self.flush()

    async def flush(self) -> None:
        async with self.lock:
            now = time.monotonic()
            channel_ready = now - self.last_channel_post_at >= self.channel_gap_seconds

            best: tuple[float, str, str, list[SignalEvent], PairState] | None = None
            for pair, state in self.states.items():
                if not state.setup_events:
                    continue
                age = now - state.setup_opened_at
                score, direction, chosen = self._combined_setup(state.setup_events)
                expired = age >= self.setup_window_seconds
                ready = expired or any(event.standalone for event in state.setup_events)
                if not self._publishable(score, chosen):
                    if expired or age > self.setup_window_seconds * 3:
                        state.setup_events.clear()
                        state.setup_opened_at = 0.0
                    continue
                if not ready:
                    continue
                if now - state.last_pair_post_at < self.cooldown_seconds:
                    if age > self.setup_window_seconds * 4:
                        state.setup_events.clear()
                        state.setup_opened_at = 0.0
                    continue
                if not channel_ready:
                    continue
                if best is None or score > best[0]:
                    best = (score, pair, direction, chosen, state)

            if best is None:
                return

            score, pair, direction, chosen, state = best
            if len(chosen) == 1:
                message = chosen[0].message
            else:
                message = build_conviction_message(
                    pair,
                    self.live_price(pair),
                    direction,
                    score,
                    [event.summary for event in chosen],
                    now_text(),
                    self.referral_link,
                )
            state.setup_events.clear()
            state.setup_opened_at = 0.0
            state.last_pair_post_at = now
            self.last_channel_post_at = now

        print("\n" + message + "\n")
        if self.publisher:
            await self.publisher.send(message)

    async def publisher_loop(self) -> None:
        while True:
            await asyncio.sleep(2)
            await self.flush()

    def merge_candle(self, pair: str, timeframe: str, candles: list[list[float]]) -> None:
        state = self.states[pair]
        target = state.candles_4h if timeframe == "4h" else state.candles_1h
        for candle in candles:
            if target and target[-1][0] == candle[0]:
                target[-1] = candle
            else:
                target.append(candle)

    async def maybe_post_ta(self, pair: str) -> None:
        """Post a periodic technical-analysis snapshot for `pair` if enough time and
        candle history have accumulated. Respects the same channel-wide quiet spacing
        used for scored signals, so TA snapshots don't crowd out or get crowded by them.
        """
        async with self.lock:
            state = self.states[pair]
            now = time.monotonic()
            if state.last_price is None:
                return
            if now - state.last_ta_post_at < TA_INTERVAL_SECONDS:
                return
            if now - self.last_channel_post_at < self.channel_gap_seconds:
                return
            message = build_message_for_pair_ticker(
                pair,
                state.last_price,
                now_text(),
                list(state.candles_4h),
                list(state.candles_1h),
                TA_PERIOD_1,
                TA_PERIOD_2,
                self.referral_link,
            )
            if message is None:
                return
            state.last_ta_post_at = now
            self.last_channel_post_at = now

        print("\n" + message + "\n")
        if self.publisher:
            await self.publisher.send(message)

    async def order_book(self, pair: str, book: dict[str, Any]) -> None:
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        if not bids or not asks:
            return

        state = self.states[pair]
        top_bids = [(float(level[0]), float(level[1])) for level in bids[:ORDER_BOOK_LEVELS]]
        top_asks = [(float(level[0]), float(level[1])) for level in asks[:ORDER_BOOK_LEVELS]]
        bid_volume = sum(amount for _, amount in top_bids)
        ask_volume = sum(amount for _, amount in top_asks)
        if bid_volume <= 0 or ask_volume <= 0:
            return

        ratio = bid_volume / ask_volume
        timestamp = extract_timestamp(book)
        best_bid, best_ask = top_bids[0][0], top_asks[0][0]
        midpoint = (best_bid + best_ask) / 2
        live = self.live_price(pair, midpoint)

        # 1. Bid/Ask Imbalance
        extremity = max(ratio, 1 / ratio)
        if extremity >= IMBALANCE_THRESHOLD:
            direction = BULL if ratio > 1 else BEAR
            score = min(90.0, 40 + (extremity - IMBALANCE_THRESHOLD) * 18)
            await self.consider(
                pair,
                1,
                score,
                direction,
                f"{'Buyers outweigh sellers' if direction == BULL else 'Sellers outweigh buyers'} ({ratio:.2f}x)",
                build_imbalance_message(pair, live, bid_volume, ask_volume, ratio, timestamp, self.referral_link),
            )

        # 2. Order Book Wall
        bid_wall = max(top_bids, key=lambda level: level[1])
        ask_wall = max(top_asks, key=lambda level: level[1])
        average_level_size = (bid_volume + ask_volume) / (2 * ORDER_BOOK_LEVELS)
        if is_qualifying_wall(bid_wall[0], bid_wall[1], average_level_size):
            multiple = bid_wall[1] / average_level_size
            usd = usd_notional(bid_wall[0], bid_wall[1])
            score = min(95.0, 42 + (multiple - WALL_MULTIPLIER) * 4 + min(20.0, usd / 80_000))
            await self.consider(
                pair,
                2,
                score,
                BULL,
                f"Support wall at ${price(bid_wall[0])} ({multiple:.1f}x, ${number(usd, 0)})",
                build_wall_message(pair, live, "bid", bid_wall[0], bid_wall[1], average_level_size, timestamp, self.referral_link),
            )
        if is_qualifying_wall(ask_wall[0], ask_wall[1], average_level_size):
            multiple = ask_wall[1] / average_level_size
            usd = usd_notional(ask_wall[0], ask_wall[1])
            score = min(95.0, 42 + (multiple - WALL_MULTIPLIER) * 4 + min(20.0, usd / 80_000))
            await self.consider(
                pair,
                12,
                score,
                BEAR,
                f"Resistance wall at ${price(ask_wall[0])} ({multiple:.1f}x, ${number(usd, 0)})",
                build_wall_message(pair, live, "ask", ask_wall[0], ask_wall[1], average_level_size, timestamp, self.referral_link),
            )

        # 3. Spread Widening
        spread_bps = ((best_ask - best_bid) / midpoint) * 10_000
        if len(state.spreads_bps) >= 8:
            baseline = statistics.mean(state.spreads_bps)
            if baseline > 0 and spread_bps >= baseline * SPREAD_WIDENING_MULTIPLIER and spread_bps >= 4:
                expansion = spread_bps / baseline
                score = min(70.0, 35 + (expansion - SPREAD_WIDENING_MULTIPLIER) * 12)
                await self.consider(
                    pair,
                    3,
                    score,
                    NEUTRAL,
                    f"Spread widened {expansion:.2f}x to {spread_bps:.2f} bps",
                    build_spread_message(pair, live, spread_bps, baseline, timestamp, self.referral_link),
                )
        state.spreads_bps.append(spread_bps)

        # 4. Wall Removal
        await self._check_wall_removal(pair, "bid", state.qualifying_bid_wall, top_bids, timestamp)
        await self._check_wall_removal(pair, "ask", state.qualifying_ask_wall, top_asks, timestamp)
        state.qualifying_bid_wall = bid_wall if is_qualifying_wall(bid_wall[0], bid_wall[1], average_level_size) else None
        state.qualifying_ask_wall = ask_wall if is_qualifying_wall(ask_wall[0], ask_wall[1], average_level_size) else None

    async def _check_wall_removal(
        self,
        pair: str,
        side: str,
        previous: tuple[float, float] | None,
        levels: list[tuple[float, float]],
        timestamp: str,
    ) -> None:
        if previous is None:
            return
        previous_price, previous_amount = previous
        remaining = next((amount for level_price, amount in levels if level_price == previous_price), 0.0)
        removed = remaining <= previous_amount * 0.40
        previous_usd = usd_notional(previous_price, previous_amount)
        if not removed or previous_usd < WALL_MIN_NOTIONAL:
            return
        direction = BEAR if side == "bid" else BULL
        drop_ratio = 1 - (remaining / previous_amount) if previous_amount else 1
        score = min(90.0, 48 + drop_ratio * 25 + min(15.0, previous_usd / 100_000))
        label = "Support" if side == "bid" else "Resistance"
        await self.consider(
            pair,
            4 if side == "bid" else 14,
            score,
            direction,
            f"{label} wall pulled at ${price(previous_price)} (${number(previous_usd, 0)} → ${number(usd_notional(previous_price, remaining), 0)})",
            build_wall_removal_message(
                pair, self.live_price(pair, previous_price), side, previous_price, previous_amount, remaining, timestamp, self.referral_link
            ),
        )

    async def trades(self, pair: str, trades: list[dict[str, Any]]) -> None:
        if not trades:
            return
        state = self.states[pair]
        batch_volume = 0.0
        batch_buy = 0.0
        batch_sell = 0.0

        for trade in trades:
            amount = float(trade.get("amount") or 0)
            trade_price = float(trade.get("price") or 0)
            notional = amount * trade_price
            side = trade.get("side")
            batch_volume += amount
            if side == "buy":
                batch_buy += amount
            elif side == "sell":
                batch_sell += amount

            # 5. Whale Trade
            if notional >= WHALE_MIN_NOTIONAL:
                if side == "buy":
                    direction = BULL
                elif side == "sell":
                    direction = BEAR
                else:
                    direction = NEUTRAL
                standalone = notional >= WHALE_STANDALONE_NOTIONAL
                score = min(120.0, 55 + min(50.0, notional / 40_000))
                await self.consider(
                    pair,
                    5,
                    score,
                    direction,
                    f"Whale {side or 'print'} ${number(notional, 0)}",
                    build_whale_message(pair, trade_price, notional, side, extract_timestamp(trade), self.referral_link),
                    standalone=standalone,
                )

        # 6. Buy/Sell Pressure
        state.buy_volume += batch_buy
        state.sell_volume += batch_sell
        pressure_total = batch_buy + batch_sell
        last_trade_price = float(trades[-1].get("price") or 0)
        tape_time = extract_timestamp(trades[-1])
        batch_notional = pressure_total * last_trade_price if last_trade_price else 0
        if pressure_total > 0 and batch_notional >= PRESSURE_MIN_NOTIONAL:
            buy_share = batch_buy / pressure_total
            if buy_share >= PRESSURE_SHARE or buy_share <= 1 - PRESSURE_SHARE:
                direction = BULL if buy_share >= PRESSURE_SHARE else BEAR
                extremity = max(buy_share, 1 - buy_share)
                score = min(80.0, 40 + (extremity - PRESSURE_SHARE) * 120)
                await self.consider(
                    pair,
                    6,
                    score,
                    direction,
                    f"{'Buyers' if direction == BULL else 'Sellers'} dominating the tape ({buy_share:.0%} buy)",
                    build_pressure_message(pair, self.live_price(pair, last_trade_price), buy_share, tape_time, self.referral_link),
                )

        # 7. Volume Spike
        if len(state.trade_volumes) >= 8:
            baseline = statistics.mean(state.trade_volumes)
            if baseline > 0 and batch_volume >= baseline * VOLUME_SPIKE_MULTIPLIER:
                multiple = batch_volume / baseline
                if batch_buy > batch_sell * 1.4:
                    direction = BULL
                elif batch_sell > batch_buy * 1.4:
                    direction = BEAR
                else:
                    direction = NEUTRAL
                score = min(70.0, 28 + (multiple - VOLUME_SPIKE_MULTIPLIER) * 6)
                await self.consider(
                    pair,
                    7,
                    score,
                    direction,
                    f"Volume spike {multiple:.1f}x average",
                    build_volume_spike_message(pair, self.live_price(pair, last_trade_price), batch_volume, baseline, tape_time, self.referral_link),
                )
        state.trade_volumes.append(batch_volume)

    async def ticker(self, pair: str, ticker: dict[str, Any]) -> None:
        last = ticker.get("last")
        if last is None:
            return
        last = float(last)
        state = self.states[pair]
        timestamp = extract_timestamp(ticker)

        # 8. Rapid Price Move
        if state.last_price and state.last_price > 0:
            move_percent = ((last - state.last_price) / state.last_price) * 100
            if abs(move_percent) >= RAPID_MOVE_PERCENT:
                direction = BULL if move_percent > 0 else BEAR
                standalone = abs(move_percent) >= RAPID_MOVE_STANDALONE_PERCENT
                score = min(110.0, 50 + abs(move_percent) * 18)
                await self.consider(
                    pair,
                    8,
                    score,
                    direction,
                    f"Sharp move {move_percent:+.2f}%",
                    build_rapid_move_message(pair, state.last_price, last, move_percent, timestamp, self.referral_link),
                    standalone=standalone,
                )

        high = ticker.get("high")
        low = ticker.get("low")

        # 9. 24h High/Low Break
        if high is not None and state.previous_24h_high is not None:
            if last >= float(high) and float(high) > state.previous_24h_high:
                await self.consider(
                    pair,
                    9,
                    105,
                    BULL,
                    f"Breaking the 24h high (${price(high)})",
                    build_high_low_break_message(pair, last, high, low, "high", timestamp, self.referral_link),
                    standalone=True,
                )
        if low is not None and state.previous_24h_low is not None:
            if last <= float(low) and float(low) < state.previous_24h_low:
                await self.consider(
                    pair,
                    19,
                    105,
                    BEAR,
                    f"Breaking the 24h low (${price(low)})",
                    build_high_low_break_message(pair, last, high, low, "low", timestamp, self.referral_link),
                    standalone=True,
                )

        state.last_price = last
        if high is not None:
            state.previous_24h_high = float(high)
        if low is not None:
            state.previous_24h_low = float(low)

    async def funding(self, pair: str, funding: dict[str, Any]) -> None:
        rate = funding.get("fundingRate")
        if rate is None:
            return
        rate = float(rate)
        if abs(rate) < EXTREME_FUNDING_RATE:
            return
        direction = BEAR if rate > 0 else BULL
        standalone = abs(rate) >= 0.0015
        score = min(100.0, 60 + abs(rate) * 20_000)
        await self.consider(
            pair,
            10,
            score,
            direction,
            f"{'Crowded longs' if rate > 0 else 'Crowded shorts'} ({rate:.4%})",
            build_funding_message(pair, self.live_price(pair), rate, funding.get("fundingTimestamp", "n/a"), extract_timestamp(funding), self.referral_link),
            standalone=standalone,
        )


def funding_symbol(pair: str) -> str:
    """Map a spot symbol like BTC/USDT to its linear perpetual swap symbol.

    CCXT's unified symbol for a USDT-margined perpetual is typically
    'BTC/USDT:USDT'. Spot symbols have no funding rate at all, so watching
    the spot symbol for funding data silently never fires.
    """
    if ":" in pair:
        return pair
    return f"{pair}{FUNDING_SYMBOL_SUFFIX}"


async def seed_candles(exchange: Any, monitor: SignalMonitor, pair: str) -> None:
    """Preload recent 4h/1h candle history via REST so TA snapshots don't have to wait
    for enough live candles to stream in before the first one can be built."""
    needed = max(TA_PERIOD_1, TA_PERIOD_2)
    for timeframe in ("4h", "1h"):
        try:
            candles = await exchange.fetch_ohlcv(pair, timeframe, limit=max(needed, TA_CANDLE_LIMIT))
            monitor.merge_candle(pair, timeframe, candles)
        except Exception as error:
            print(f"{pair} {timeframe} candle seed skipped: {error}")


async def watch_pair(exchange: Any, monitor: SignalMonitor, pair: str, swap_pair: str | None) -> None:
    async def watch_order_book() -> None:
        while True:
            try:
                await monitor.order_book(pair, await exchange.watch_order_book(pair))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                print(f"{pair} order-book error: {error}")
                await asyncio.sleep(3)

    async def watch_trades() -> None:
        while True:
            try:
                await monitor.trades(pair, await exchange.watch_trades(pair))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                print(f"{pair} trades error: {error}")
                await asyncio.sleep(3)

    async def watch_ticker() -> None:
        while True:
            try:
                await monitor.ticker(pair, await exchange.watch_ticker(pair))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                print(f"{pair} ticker error: {error}")
                await asyncio.sleep(3)

    async def watch_funding() -> None:
        if not swap_pair:
            if pair not in FUNDING_WARNED:
                print(f"{pair} has no linear swap market available; skipping funding-rate signal.")
                FUNDING_WARNED.add(pair)
            return
        while True:
            try:
                funding = await exchange.watch_funding_rate(swap_pair)
                await monitor.funding(pair, funding)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if pair not in FUNDING_WARNED:
                    print(f"{pair} funding stream unavailable ({swap_pair}): {error}")
                    FUNDING_WARNED.add(pair)
                await asyncio.sleep(60)

    async def watch_candles(timeframe: str) -> None:
        while True:
            try:
                candles = await exchange.watch_ohlcv(pair, timeframe)
                monitor.merge_candle(pair, timeframe, candles)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                print(f"{pair} {timeframe} candles error: {error}")
                await asyncio.sleep(5)

    async def watch_ta_snapshot() -> None:
        while True:
            await asyncio.sleep(30)
            await monitor.maybe_post_ta(pair)

    await asyncio.gather(
        watch_order_book(),
        watch_trades(),
        watch_ticker(),
        watch_funding(),
        watch_candles("4h"),
        watch_candles("1h"),
        watch_ta_snapshot(),
    )


def create_bybit_exchange(market_types: list[str]) -> Any:
    timeout_ms = int(os.getenv("BITNOMIK_HTTP_TIMEOUT_MS", "30000"))
    proxy = os.getenv("BITNOMIK_PROXY", "").strip() or os.getenv("HTTPS_PROXY", "").strip()
    config: dict[str, Any] = {
        "enableRateLimit": True,
        "timeout": timeout_ms,
        "options": {
            "defaultType": "spot",
            "fetchMarkets": market_types,
        },
    }
    if proxy:
        config["httpsProxy"] = proxy
    return ccxtpro.bybit(config)


async def load_markets_with_retry(exchange: Any, attempts: int = 4) -> None:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            await exchange.load_markets()
            return
        except NetworkError as error:
            last_error = error
            print(f"Bybit market load failed (attempt {attempt}/{attempts}): {error}")
            if attempt < attempts:
                await asyncio.sleep(2 * attempt)
    raise last_error or NetworkError("Bybit market load failed")


async def run(pairs: list[str], cooldown_seconds: int, dry_run: bool, channel_gap_seconds: int = CHANNEL_GAP_SECONDS) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    channel_id = os.getenv("TELEGRAM_CHANNEL_ID", "").strip()
    referral_link = os.getenv("BITNOMIK_REFERRAL_LINK", "").strip() or DEFAULT_REFERRAL_LINK

    publisher = None
    if dry_run:
        print("Dry run enabled: messages will be printed but not sent to Telegram.")
    elif not token or not channel_id:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHANNEL_ID environment variables. "
            "Set them in your shell/host, or use --dry-run to test without Telegram."
        )
    else:
        publisher = TelegramPublisher(token, channel_id, referral_link)

    exchange = create_bybit_exchange(["spot"])
    monitor = SignalMonitor(publisher, cooldown_seconds, referral_link, channel_gap_seconds=channel_gap_seconds)
    try:
        await load_markets_with_retry(exchange)

        swap_markets_loaded = False
        try:
            exchange.options["fetchMarkets"] = ["spot", "linear"]
            await exchange.load_markets(reload=True)
            swap_markets_loaded = True
        except NetworkError as error:
            print(f"Bybit linear markets skipped: {error}")
            exchange.options["fetchMarkets"] = ["spot"]

        available_pairs = [pair for pair in pairs if pair in exchange.markets]
        missing_pairs = sorted(set(pairs) - set(available_pairs))
        if missing_pairs:
            raise ValueError(f"Unknown Bybit pairs: {', '.join(missing_pairs)}")

        swap_pairs: dict[str, str | None] = {}
        for pair in available_pairs:
            candidate = funding_symbol(pair)
            if swap_markets_loaded and candidate in exchange.markets:
                swap_pairs[pair] = candidate
            else:
                swap_pairs[pair] = None

        print(f"Watching {', '.join(available_pairs)}")
        print(
            "Quiet channel mode: only high-conviction setups are posted "
            f"(min score {MIN_SETUP_SCORE:.0f}, {monitor.channel_gap_seconds}s between posts)."
        )
        await asyncio.gather(*(seed_candles(exchange, monitor, pair) for pair in available_pairs))
        await asyncio.gather(
            monitor.publisher_loop(),
            *(watch_pair(exchange, monitor, pair, swap_pairs[pair]) for pair in available_pairs),
        )
    finally:
        await exchange.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="bitNOMIK ten-signal crypto monitor")
    parser.add_argument(
        "--pairs",
        default=os.getenv("BITNOMIK_PAIRS", DEFAULT_PAIRS),
        help="Comma-separated Bybit symbols, e.g. BTC/USDT,ETH/USDT",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=int(os.getenv("BITNOMIK_SIGNAL_COOLDOWN", "1800")),
        help="Minimum seconds between posts for the same pair",
    )
    parser.add_argument(
        "--channel-gap",
        type=int,
        default=CHANNEL_GAP_SECONDS,
        help="Minimum seconds between any two channel posts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print signals without sending Telegram messages",
    )
    args = parser.parse_args()
    pairs = [pair.strip().upper() for pair in args.pairs.split(",") if pair.strip()]
    if not pairs:
        raise SystemExit("At least one pair is required.")
    try:
        asyncio.run(run(pairs, args.cooldown, args.dry_run, args.channel_gap))
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
