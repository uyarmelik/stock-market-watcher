"""
Stock Market Watcher — AI-powered exchange tracking assistant.
Uses a debate loop (Gemini vs OpenAI) for high-confidence signals only.
Scheduler: Opportunity Hunter (weekdays 12:15) + Drop Detector (weekdays 10–17, hourly).
"""

import argparse
import json
import os
import re
import smtplib
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import schedule
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types as genai_types

load_dotenv()

ENV = os.environ
KEY_GEMINI = ENV.get("GEMINI_API_KEY")
KEY_OPENAI = ENV.get("OPENAI_API_KEY")
MAIL_USER = ENV.get("EMAIL_USER")
MAIL_PASS = ENV.get("EMAIL_PASS")
MAIL_RCVR = ENV.get("RECEIVER_EMAIL")

GEMINI_MODEL = ENV.get("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = ENV.get("OPENAI_MODEL", "gpt-4o")

BIST_EQUITY_CSV_URL = ENV.get(
    "BIST_EQUITY_CSV_URL",
    "https://www.borsaistanbul.com/datum/hisse_endeks_ds.csv",
)

CONFIG_PATH = "config.json"
STOCK_STATES_PATH = "stock_states.json"
ALERT_STATES_PATH = "alert_states.json"

# Verdicts that require both models to agree; anything else is filtered out.
STRONG_VERDICTS = ("STRONG BUY", "STRONG SELL")


@dataclass
class StockSnapshot:
    ticker: str
    price: float
    sma20: Optional[float]
    sma50: Optional[float]
    rsi14: Optional[float]
    ret_5d_pct: Optional[float]
    ret_21d_pct: Optional[float]
    baseline: str
    purchase_price: Optional[float] = None
    target_low: Optional[float] = None
    target_high: Optional[float] = None


def read_json(path: str) -> Dict[str, Any]:
    """Load JSON file; return empty dict if missing or invalid."""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
    return {}


def write_json(path: str, obj: Any) -> None:
    """Write JSON to file with graceful failure."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: could not write {path}: {e}")


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\[\s*\{.*\}\s*\]", text, flags=re.DOTALL)
    if m:
        return m.group(0)
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, flags=re.DOTALL)
    if m:
        return m.group(0)
    return None


def normalize_verdict(s: str) -> str:
    """Normalize model verdict to STRONG BUY | STRONG SELL | HOLD."""
    t = (s or "").strip().upper()
    if "STRONG BUY" in t or t == "STRONG_BUY":
        return "STRONG BUY"
    if "STRONG SELL" in t or t == "STRONG_SELL":
        return "STRONG SELL"
    return "HOLD"


def compute_rsi(close: pd.Series, period: int = 14) -> Optional[float]:
    if close is None or close.empty or len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    v = rsi.iloc[-1]
    return None if pd.isna(v) else float(v)


def pct_return(close: pd.Series, days: int) -> Optional[float]:
    if close is None or close.empty or len(close) < days + 1:
        return None
    cur = close.iloc[-1]
    prev = close.iloc[-(days + 1)]
    if pd.isna(cur) or pd.isna(prev) or prev == 0:
        return None
    return float((cur / prev - 1) * 100)


def baseline_action(
    price: float,
    sma20: Optional[float],
    sma50: Optional[float],
    rsi14: Optional[float],
) -> str:
    if sma20 is None or sma50 is None or rsi14 is None:
        return "HOLD"
    if price > sma20 > sma50 and 55 <= rsi14 <= 70:
        return "BUY"
    if price < sma20 < sma50 and rsi14 <= 45:
        return "SELL"
    return "HOLD"


def format_snapshot_line(s: StockSnapshot) -> str:
    parts = [
        f"{s.ticker}",
        f"P={s.price:.2f}",
        f"RSI={'' if s.rsi14 is None else f'{s.rsi14:.1f}'}",
        f"SMA20={'' if s.sma20 is None else f'{s.sma20:.2f}'}",
        f"SMA50={'' if s.sma50 is None else f'{s.sma50:.2f}'}",
        f"R5={'' if s.ret_5d_pct is None else f'{s.ret_5d_pct:.1f}%'}",
        f"R21={'' if s.ret_21d_pct is None else f'{s.ret_21d_pct:.1f}%'}",
        f"BASE={s.baseline}",
    ]
    if s.purchase_price is not None:
        parts.append(f"MAL={s.purchase_price:.2f}")
    return " | ".join(parts)


def format_position_line(s: StockSnapshot) -> str:
    pp = s.purchase_price or 0
    chg = ((s.price - pp) / pp * 100) if pp else 0
    parts = [
        f"{s.ticker}",
        f"bought_at={pp:.2f}",
        f"current={s.price:.2f}",
        f"chg_vs_buy={chg:+.1f}%",
        f"RSI={'' if s.rsi14 is None else f'{s.rsi14:.1f}'}",
        f"SMA20={'' if s.sma20 is None else f'{s.sma20:.2f}'}",
    ]
    return " | ".join(parts)


def fetch_bist_tickers_from_borsaistanbul() -> List[str]:
    try:
        raw = urllib.request.urlopen(BIST_EQUITY_CSV_URL, timeout=30).read()
        text = raw.decode("utf-8", errors="ignore")
        tickers: List[str] = []
        for line in text.splitlines():
            first = line.split(";")[0].strip()
            if not first.endswith(".E"):
                continue
            sym = first[:-2]
            if not sym or len(sym) < 2:
                continue
            tickers.append(f"{sym}.IS")
        seen = set()
        out: List[str] = []
        for t in tickers:
            if t not in seen:
                out.append(t)
                seen.add(t)
        return out
    except Exception as e:
        print(f"Failed to download BIST ticker list: {e}")
        return []


def load_universe(config: Dict[str, Any], universe: str) -> List[str]:
    if universe == "config":
        return list(config.get("tickers", []))
    if universe == "bist":
        return fetch_bist_tickers_from_borsaistanbul()
    cfg = list(config.get("tickers", []))
    return cfg if cfg else fetch_bist_tickers_from_borsaistanbul()


def download_history(
    tickers: List[str], period: str = "3mo"
) -> Optional[pd.DataFrame]:
    if not tickers:
        return pd.DataFrame()
    try:
        df = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            group_by="ticker",
            threads=True,
            auto_adjust=False,
            progress=False,
        )
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception as e:
        print(f"Download history error: {e}")
        return pd.DataFrame()


def get_close_series(df: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if ticker not in df.columns.get_level_values(0):
            return None
        return df[ticker]["Close"].dropna()
    if "Close" not in df.columns:
        return None
    return df["Close"].dropna()


def build_snapshots(
    tickers: List[str],
    target_prices: Dict[str, Any],
    period: str = "3mo",
) -> List[StockSnapshot]:
    df = download_history(tickers, period=period)
    if df is None or df.empty:
        return []

    snapshots: List[StockSnapshot] = []
    for t in tickers:
        try:
            close = get_close_series(df, t)
            if close is None or close.empty:
                continue
            price = float(close.iloc[-1])
            sma20 = (
                safe_float(close.rolling(20).mean().iloc[-1])
                if len(close) >= 20
                else None
            )
            sma50 = (
                safe_float(close.rolling(50).mean().iloc[-1])
                if len(close) >= 50
                else None
            )
            rsi14 = compute_rsi(close, 14)
            r5 = pct_return(close, 5)
            r21 = pct_return(close, 21)
            tp = target_prices.get(t, {}) if isinstance(target_prices, dict) else {}
            purchase_price = safe_float(tp.get("purchase_price"))
            # Auto low/high from purchase_price ±5% when not set in config
            low = safe_float(tp.get("low"))
            high = safe_float(tp.get("high"))
            if purchase_price is not None:
                if low is None:
                    low = round(purchase_price * 0.95, 2)
                if high is None:
                    high = round(purchase_price * 1.05, 2)
            base = baseline_action(price, sma20, sma50, rsi14)
            snapshots.append(
                StockSnapshot(
                    ticker=t,
                    price=round(price, 2),
                    sma20=None if sma20 is None else round(sma20, 2),
                    sma50=None if sma50 is None else round(sma50, 2),
                    rsi14=None if rsi14 is None else round(rsi14, 1),
                    ret_5d_pct=None if r5 is None else round(r5, 1),
                    ret_21d_pct=None if r21 is None else round(r21, 1),
                    baseline=base,
                    purchase_price=(
                        None if purchase_price is None else round(purchase_price, 2)
                    ),
                    target_low=None if low is None else round(low, 2),
                    target_high=None if high is None else round(high, 2),
                )
            )
        except Exception as e:
            print(f"Warning: skip snapshot for {t}: {e}")
            continue
    return snapshots


# ---------- Debate loop: Gemini (Round 1 & 3) <-> OpenAI (Round 2 & Final) ----------

SYSTEM_ENGLISH = (
    "You must respond only in English. Use plain language suitable for someone without financial expertise. "
    "Avoid jargon; use phrases like 'Price is too high, risk of drop' or 'Price is below average, buying opportunity.'"
)

def debate_round1_gemini_thesis(
    snapshots: List[StockSnapshot], intent: str
) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """Round 1: Gemini presents BUY or SELL thesis. intent is 'BUY' or 'SELL'."""
    if not KEY_GEMINI or not snapshots:
        return None
    lines = [format_snapshot_line(s) for s in snapshots]
    prompt = (
        f"You are a technical analyst for BIST (Istanbul Stock Exchange). Your task is to present a clear {intent} thesis ONLY for stocks that support a {intent}.\n"
        "Rules:\n"
        "- Respond ONLY in English. Use plain language (e.g. 'Price is below average, buying opportunity' not 'RSI oversold').\n"
        "- For each ticker you recommend {intent}, provide: ticker, action (exactly '{intent}' or 'HOLD'), reason (short, plain English), entry_price (number), target_exit_price (number).\n"
        "- If a stock does not support {intent}, set action to 'HOLD' and give a brief reason.\n"
        "- Return ONLY a JSON array. No other text.\n"
        "Format: [{\"ticker\":\"XXX.IS\",\"action\":\"BUY|SELL|HOLD\",\"reason\":\"...\",\"entry_price\":number,\"target_exit_price\":number}]\n\n"
        "Data:\n" + "\n".join(lines)
    ).replace("{intent}", intent)

    try:
        client = genai.Client(api_key=KEY_GEMINI)
        config = genai_types.GenerateContentConfig(
            tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
        )
        resp = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt, config=config
        )
        text = getattr(resp, "text", "") or ""
        block = extract_json_block(text)
        if not block:
            return None
        arr = json.loads(block)
        if not isinstance(arr, list):
            return None
        return text, arr
    except Exception as e:
        print(f"Debate Round 1 (Gemini) error: {e}")
        return None


def debate_round2_openai_critique(
    thesis_text: str, snapshots: List[StockSnapshot], intent: str
) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """Round 2: OpenAI acts as Risk Manager and criticizes the thesis."""
    if not KEY_OPENAI:
        return None
    compact = [
        {
            "ticker": s.ticker,
            "price": s.price,
            "rsi": s.rsi14,
            "sma20": s.sma20,
            "sma50": s.sma50,
        }
        for s in snapshots
    ]
    prompt = (
        f"You are a strict Risk Manager. Gemini has proposed a {intent} thesis. Your job is to criticize it: bear market scenarios, technical traps, overbought/oversold false signals.\n"
        "Rules:\n"
        "- Respond ONLY in English. Use plain language.\n"
        "- For each ticker, output: ticker, critique (short), verdict_after_critique: exactly one of 'STRONG BUY', 'STRONG SELL', 'HOLD'. If you disagree with {intent}, use HOLD or the opposite.\n"
        "- Return ONLY a JSON array. No other text.\n"
        "Format: [{\"ticker\":\"XXX.IS\",\"critique\":\"...\",\"verdict_after_critique\":\"STRONG BUY|STRONG SELL|HOLD\"}]\n\n"
        f"Technical data: {json.dumps(compact)}\n\n"
        f"Gemini thesis (Round 1):\n{thesis_text}"
    ).replace("{intent}", intent)

    try:
        client = OpenAI(api_key=KEY_OPENAI)
        res = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_ENGLISH + " Output only JSON array."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        text = res.choices[0].message.content or ""
        block = extract_json_block(text)
        if not block:
            return None
        arr = json.loads(block)
        if not isinstance(arr, list):
            return None
        return text, arr
    except Exception as e:
        print(f"Debate Round 2 (OpenAI) error: {e}")
        return None


def debate_round3_gemini_reply(
    thesis_text: str,
    critique_text: str,
    snapshots: List[StockSnapshot],
    intent: str,
) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """Round 3: Gemini replies to critique and gives final stance (STRONG BUY / STRONG SELL / HOLD)."""
    if not KEY_GEMINI:
        return None
    lines = [format_snapshot_line(s) for s in snapshots]
    prompt = (
        f"Your {intent} thesis was criticized by a Risk Manager. Reply to the critique and give your FINAL stance per ticker.\n"
        "Rules:\n"
        "- Respond ONLY in English. Plain language.\n"
        "- For each ticker set final_action to exactly one of: 'STRONG BUY', 'STRONG SELL', 'HOLD'. Only use STRONG BUY or STRONG SELL if you remain confident after the critique.\n"
        "- Include: ticker, reply (short), final_action, reason, entry_price, target_exit_price.\n"
        "- Return ONLY a JSON array.\n"
        "Format: [{\"ticker\":\"XXX.IS\",\"reply\":\"...\",\"final_action\":\"STRONG BUY|STRONG SELL|HOLD\",\"reason\":\"...\",\"entry_price\":number,\"target_exit_price\":number}]\n\n"
        "Data:\n" + "\n".join(lines) + "\n\n--- Your thesis (Round 1) ---\n" + thesis_text + "\n\n--- Risk Manager critique (Round 2) ---\n" + critique_text
    ).replace("{intent}", intent)

    try:
        client = genai.Client(api_key=KEY_GEMINI)
        config = genai_types.GenerateContentConfig(
            tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
        )
        resp = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt, config=config
        )
        text = getattr(resp, "text", "") or ""
        block = extract_json_block(text)
        if not block:
            return None
        arr = json.loads(block)
        if not isinstance(arr, list):
            return None
        return text, arr
    except Exception as e:
        print(f"Debate Round 3 (Gemini) error: {e}")
        return None


def debate_final_openai_verdict(
    round1_text: str,
    round2_text: str,
    round3_text: str,
    snapshots: List[StockSnapshot],
) -> Optional[List[Dict[str, Any]]]:
    """Final: OpenAI summarizes the debate and outputs FINAL VERDICT per ticker."""
    if not KEY_OPENAI:
        return None
    compact = [{"ticker": s.ticker, "price": s.price} for s in snapshots]
    prompt = (
        "Summarize the debate and give your FINAL VERDICT for each ticker. Only STRONG BUY or STRONG SELL if you agree with the final thesis; otherwise HOLD.\n"
        "Rules:\n"
        "- Respond ONLY in English. Plain language.\n"
        "- For each ticker output: ticker, verdict (exactly 'STRONG BUY', 'STRONG SELL', or 'HOLD'), entry_price, target_exit_price, summary_reason (plain English, no jargon).\n"
        "- Return ONLY a JSON array. No other text.\n"
        "Format: [{\"ticker\":\"XXX.IS\",\"verdict\":\"STRONG BUY|STRONG SELL|HOLD\",\"entry_price\":number,\"target_exit_price\":number,\"summary_reason\":\"...\"}]\n\n"
        f"Data: {json.dumps(compact)}\n\n--- Round 1 (Gemini thesis) ---\n{round1_text}\n\n--- Round 2 (Risk Manager critique) ---\n{round2_text}\n\n--- Round 3 (Gemini reply) ---\n{round3_text}"
    )

    try:
        client = OpenAI(api_key=KEY_OPENAI)
        res = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_ENGLISH + " Output only JSON array."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        text = res.choices[0].message.content or ""
        block = extract_json_block(text)
        if not block:
            return None
        arr = json.loads(block)
        if not isinstance(arr, list):
            return None
        return arr
    except Exception as e:
        print(f"Debate Final (OpenAI) error: {e}")
        return None


def run_debate_loop(
    snapshots: List[StockSnapshot], intent: str
) -> List[Dict[str, Any]]:
    """
    Run full debate (Gemini -> OpenAI -> Gemini -> OpenAI). Return only items where
    BOTH models agree on STRONG BUY or STRONG SELL; filter out HOLD or disagreement.
    """
    if not snapshots:
        return []

    r1 = debate_round1_gemini_thesis(snapshots, intent)
    if not r1:
        return []
    thesis_text, thesis_arr = r1

    r2 = debate_round2_openai_critique(thesis_text, snapshots, intent)
    if not r2:
        return []
    critique_text, critique_arr = r2

    r3 = debate_round3_gemini_reply(
        thesis_text, critique_text, snapshots, intent
    )
    if not r3:
        return []
    reply_text, reply_arr = r3

    final_arr = debate_final_openai_verdict(
        thesis_text, critique_text, reply_text, snapshots
    )
    if not final_arr:
        return []

    # Build maps by ticker
    reply_by_ticker = {}
    for r in reply_arr:
        t = (r.get("ticker") or "").strip()
        if t:
            reply_by_ticker[t] = r
    final_by_ticker = {}
    for f in final_arr:
        t = (f.get("ticker") or "").strip()
        if t:
            final_by_ticker[t] = f

    # Only report when BOTH say STRONG BUY or BOTH say STRONG SELL (same verdict)
    approved: List[Dict[str, Any]] = []
    for s in snapshots:
        t = s.ticker
        gemini_verdict = normalize_verdict(
            (reply_by_ticker.get(t) or {}).get("final_action", "HOLD")
        )
        openai_verdict = normalize_verdict(
            (final_by_ticker.get(t) or {}).get("verdict", "HOLD")
        )
        if gemini_verdict not in STRONG_VERDICTS or openai_verdict not in STRONG_VERDICTS:
            continue
        if gemini_verdict != openai_verdict:
            continue
        rec = final_by_ticker.get(t) or reply_by_ticker.get(t) or {}
        entry = safe_float(rec.get("entry_price")) or s.price
        target = safe_float(rec.get("target_exit_price")) or (s.target_high or s.price * 1.1)
        approved.append({
            "ticker": t,
            "verdict": gemini_verdict,
            "entry_price": round(entry, 2),
            "target_exit_price": round(target, 2),
            "reason": (rec.get("summary_reason") or rec.get("reason") or "").strip() or "Agreed after debate.",
            "current_price": s.price,
        })
    return approved


def send_email(subject: str, body: str) -> bool:
    if not (MAIL_USER and MAIL_PASS and MAIL_RCVR):
        print("Mail env variables missing (EMAIL_USER/EMAIL_PASS/RECEIVER_EMAIL).")
        return False
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = MAIL_USER
    msg["To"] = MAIL_RCVR
    msg.attach(MIMEText(body, "plain", "utf-8"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(MAIL_USER, MAIL_PASS)
            s.send_message(msg)
        return True
    except Exception as e:
        print(f"Mail error: {e}")
        return False


def format_opportunity_email_plain_english(
    opportunities: List[Dict[str, Any]],
    title: str,
    target_prices: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Mentorship-style email: plain English, Entry Price, Target Exit Price. Subject is always 'Daily Recommendation - Date'. Main list excludes target_prices tickers; those are listed at the end with Your purchase price. Strong Sell then Strong Buy, each sorted alphabetically."""
    today = datetime.now().strftime("%d-%m-%Y")
    subject = f"Daily Recommendation - {today}"
    target_prices = target_prices or {}

    # Main list: only tickers NOT in target_prices (watchlist opportunities)
    main_opps = [o for o in opportunities if o.get("ticker", "") not in target_prices]
    # Owned positions: tickers IN target_prices (listed at the end)
    owned_opps = [o for o in opportunities if o.get("ticker", "") in target_prices]

    strong_sell_main = sorted(
        [o for o in main_opps if normalize_verdict(o.get("verdict", "")) == "STRONG SELL"],
        key=lambda o: (o.get("ticker") or ""),
    )
    strong_buy_main = sorted(
        [o for o in main_opps if normalize_verdict(o.get("verdict", "")) == "STRONG BUY"],
        key=lambda o: (o.get("ticker") or ""),
    )
    n_sell, n_buy = len(strong_sell_main), len(strong_buy_main)

    lines = [
        f"Strong Sell: {n_sell}  |  Strong Buy: {n_buy}",
        "",
    ]
    num = 1
    for o in strong_sell_main:
        ticker = o.get("ticker", "")
        entry = o.get("entry_price")
        target = o.get("target_exit_price")
        reason = (o.get("reason") or "").strip() or "No additional comment."
        entry_s = f"{entry:.2f}" if entry is not None else "N/A"
        target_s = f"{target:.2f}" if target is not None else "N/A"
        lines.append(f"{num}. {ticker}: STRONG SELL")
        lines.append(f"   Entry price: {entry_s}  |  Target exit price: {target_s}")
        lines.append(f"   Reason: {reason}")
        lines.append("")
        num += 1
    for o in strong_buy_main:
        ticker = o.get("ticker", "")
        entry = o.get("entry_price")
        target = o.get("target_exit_price")
        reason = (o.get("reason") or "").strip() or "No additional comment."
        entry_s = f"{entry:.2f}" if entry is not None else "N/A"
        target_s = f"{target:.2f}" if target is not None else "N/A"
        lines.append(f"{num}. {ticker}: STRONG BUY")
        lines.append(f"   Entry price: {entry_s}  |  Target exit price: {target_s}")
        lines.append(f"   Reason: {reason}")
        lines.append("")
        num += 1
    lines.append("This is not investment advice; for informational purposes only.")

    # At the end: owned positions (target_prices), STRONG SELL then STRONG BUY, alphabetically; show Your purchase price above Entry price
    if owned_opps and target_prices:
        strong_sell_owned = sorted(
            [o for o in owned_opps if normalize_verdict(o.get("verdict", "")) == "STRONG SELL"],
            key=lambda o: (o.get("ticker") or ""),
        )
        strong_buy_owned = sorted(
            [o for o in owned_opps if normalize_verdict(o.get("verdict", "")) == "STRONG BUY"],
            key=lambda o: (o.get("ticker") or ""),
        )
        lines.append("")
        lines.append("MY POSITIONS")
        lines.append("")
        num_owned = 1
        for o in strong_sell_owned:
            ticker = o.get("ticker", "")
            tp = target_prices.get(ticker, {}) if isinstance(target_prices, dict) else {}
            purchase = safe_float(tp.get("purchase_price"))
            entry = o.get("entry_price")
            target = o.get("target_exit_price")
            reason = (o.get("reason") or "").strip() or "No additional comment."
            purchase_s = f"{purchase:.2f}" if purchase is not None else "N/A"
            entry_s = f"{entry:.2f}" if entry is not None else "N/A"
            target_s = f"{target:.2f}" if target is not None else "N/A"
            lines.append(f"{num_owned}. {ticker}: STRONG SELL")
            lines.append(f"   Your purchase price: {purchase_s}")
            lines.append(f"   Entry price: {entry_s}  |  Target exit price: {target_s}")
            lines.append(f"   Reason: {reason}")
            lines.append("")
            num_owned += 1
        for o in strong_buy_owned:
            ticker = o.get("ticker", "")
            tp = target_prices.get(ticker, {}) if isinstance(target_prices, dict) else {}
            purchase = safe_float(tp.get("purchase_price"))
            entry = o.get("entry_price")
            target = o.get("target_exit_price")
            reason = (o.get("reason") or "").strip() or "No additional comment."
            purchase_s = f"{purchase:.2f}" if purchase is not None else "N/A"
            entry_s = f"{entry:.2f}" if entry is not None else "N/A"
            target_s = f"{target:.2f}" if target is not None else "N/A"
            lines.append(f"{num_owned}. {ticker}: STRONG BUY")
            lines.append(f"   Your purchase price: {purchase_s}")
            lines.append(f"   Entry price: {entry_s}  |  Target exit price: {target_s}")
            lines.append(f"   Reason: {reason}")
            lines.append("")
            num_owned += 1

    return subject, "\n".join(lines)


# ---------- Scenario A: Opportunity Hunter (weekdays 12:15) ----------

def run_opportunity_hunter() -> None:
    """Run BUY scan for tickers list and SELL scan for target_prices; send email only if debate approves any."""
    config = read_json(CONFIG_PATH)
    tickers_watch = list(config.get("tickers", []))
    target_prices = config.get("target_prices", {})
    owned_tickers = list(target_prices.keys()) if isinstance(target_prices, dict) else []

    all_opportunities: List[Dict[str, Any]] = []

    # BUY opportunities for watchlist
    if tickers_watch:
        try:
            snap_buy = build_snapshots(
                tickers=tickers_watch,
                target_prices=config.get("target_prices", {}),
                period="3mo",
            )
            if snap_buy:
                approved_buy = run_debate_loop(snap_buy, "BUY")
                all_opportunities.extend(approved_buy)
        except Exception as e:
            print(f"Opportunity Hunter (BUY) error: {e}")

    # SELL opportunities for owned positions
    if owned_tickers:
        try:
            snap_sell = build_snapshots(
                tickers=owned_tickers,
                target_prices=target_prices,
                period="3mo",
            )
            if snap_sell:
                approved_sell = run_debate_loop(snap_sell, "SELL")
                all_opportunities.extend(approved_sell)
        except Exception as e:
            print(f"Opportunity Hunter (SELL) error: {e}")

    if not all_opportunities:
        return  # Do not send email when no approved opportunities

    subject, body = format_opportunity_email_plain_english(
        all_opportunities,
        "Stock Opportunity Hunter — Approved Signals",
        target_prices=target_prices,
    )
    send_email(subject, body)


# ---------- Scenario B: Drop Detector (weekdays 10–17, hourly) ----------

def get_yearly_metrics(ticker: str) -> Optional[Dict[str, float]]:
    """Fetch 1-year data; return price, sma252, high_52w. None on error."""
    try:
        df = download_history([ticker], period="1y")
        if df is None or df.empty:
            return None
        close = get_close_series(df, ticker)
        if close is None or len(close) < 2:
            return None
        price = float(close.iloc[-1])
        sma252 = (
            float(close.rolling(252).mean().iloc[-1])
            if len(close) >= 252
            else float(close.mean())
        )
        high_52w = float(close.max())
        return {"price": price, "sma252": sma252, "high_52w": high_52w}
    except Exception as e:
        print(f"Drop Detector data error for {ticker}: {e}")
        return None


def run_drop_detector() -> None:
    """Check: (1) 20% below SMA252 or 30% below 52w high; (2) target_prices: price vs purchase ±5% band. Send one email only when there is at least one alert."""
    config = read_json(CONFIG_PATH)
    tickers = list(config.get("tickers", []))
    target_prices = config.get("target_prices", {}) or {}
    if isinstance(target_prices, dict):
        for t in target_prices:
            if t not in tickers:
                tickers.append(t)
    if not tickers:
        return

    states = read_json(ALERT_STATES_PATH)
    alerts_sent_this_run: List[Dict[str, Any]] = []
    band_alerts: List[Dict[str, Any]] = []

    for ticker in tickers:
        try:
            m = get_yearly_metrics(ticker)
            if not m:
                continue
            price = m["price"]
            sma252 = m["sma252"]
            high_52w = m["high_52w"]
            st = dict(states.get(ticker, {}))

            # --- Drop detector (SMA252 / 52w high) ---
            below_sma20 = sma252 > 0 and price <= sma252 * 0.80
            below_high30 = high_52w > 0 and price <= high_52w * 0.70
            if not (below_sma20 or below_high30):
                if ticker in states:
                    last_price = safe_float(st.get("last_alert_price"))
                    if last_price and price >= max(sma252 * 0.81, high_52w * 0.71):
                        st["sent"] = False
                states[ticker] = st
                # Band check still runs below for target_prices; do not continue here
            else:
                already_sent = st.get("sent", False)
                last_alert_price = safe_float(st.get("last_alert_price"))
                if last_alert_price is not None and price <= last_alert_price * 0.95:
                    already_sent = False
                if not already_sent:
                    alerts_sent_this_run.append({
                        "ticker": ticker,
                        "price": price,
                        "sma252": sma252,
                        "high_52w": high_52w,
                        "below_sma20_pct": below_sma20,
                        "below_high30_pct": below_high30,
                    })
                    st["sent"] = True
                    st["last_alert_price"] = price
                    st["last_alert_at"] = datetime.now().isoformat(timespec="seconds")

            # --- Band alert: purchase_price ±5% (only for target_prices) ---
            if ticker in target_prices:
                tp = target_prices.get(ticker, {}) or {}
                purchase_price = safe_float(tp.get("purchase_price"))
                if purchase_price is not None and purchase_price > 0:
                    band_low = round(purchase_price * 0.95, 2)
                    band_high = round(purchase_price * 1.05, 2)
                    if price <= band_low:
                        if not st.get("band_below_sent", False):
                            band_alerts.append({
                                "ticker": ticker,
                                "price": price,
                                "purchase_price": purchase_price,
                                "kind": "below",
                            })
                            st["band_below_sent"] = True
                    elif price >= band_high:
                        if not st.get("band_above_sent", False):
                            band_alerts.append({
                                "ticker": ticker,
                                "price": price,
                                "purchase_price": purchase_price,
                                "kind": "above",
                            })
                            st["band_above_sent"] = True
                    else:
                        st["band_below_sent"] = False
                        st["band_above_sent"] = False

            states[ticker] = st
        except Exception as e:
            print(f"Drop Detector ticker {ticker} error: {e}")
            continue

    write_json(ALERT_STATES_PATH, states)

    if not alerts_sent_this_run and not band_alerts:
        return

    lines = []
    if alerts_sent_this_run:
        lines.extend([
            "Urgent Opportunity / Alert — Price drop detected",
            "",
            "The following ticker(s) are significantly below their yearly average or 52-week high (plain English: price is low relative to the past year).",
            "",
        ])
        for a in alerts_sent_this_run:
            t = a.get("ticker", "")
            p = a.get("price")
            lines.append(f"• {t}: current price {p:.2f}")
            if a.get("below_sma20_pct"):
                lines.append("  — Price is 20% or more below the 1-year average.")
            if a.get("below_high30_pct"):
                lines.append("  — Price is 30% or more below the 52-week high.")
            lines.append("")
    if band_alerts:
        if lines:
            lines.append("")
        lines.extend([
            "Price band alert (vs your purchase price ±5%)",
            "",
        ])
        for a in band_alerts:
            t = a.get("ticker", "")
            p = a.get("price")
            pp = a.get("purchase_price")
            if a.get("kind") == "below":
                lines.append(f"• {t}: Price is 5% or more below your purchase price (current {p:.2f}, purchase {pp:.2f}).")
            else:
                lines.append(f"• {t}: Price is 5% or more above your purchase price (current {p:.2f}, purchase {pp:.2f}).")
            lines.append("")
    lines.append("This is not investment advice; for informational purposes only.")
    subject = "Urgent Opportunity / Alert — Drop detected"
    send_email(subject, "\n".join(lines))


# ---------- Scheduler ----------

def run_scheduler() -> None:
    """Run scheduled jobs: Opportunity Hunter weekdays 12:15, Drop Detector weekdays 10–17 every hour."""
    schedule.clear()
    schedule.every().monday.at("12:15").do(run_opportunity_hunter)
    schedule.every().tuesday.at("12:15").do(run_opportunity_hunter)
    schedule.every().wednesday.at("12:15").do(run_opportunity_hunter)
    schedule.every().thursday.at("12:15").do(run_opportunity_hunter)
    schedule.every().friday.at("12:15").do(run_opportunity_hunter)

    for hour in range(10, 18):  # 10:00 to 17:00
        schedule.every().monday.at(f"{hour:02d}:00").do(run_drop_detector)
        schedule.every().tuesday.at(f"{hour:02d}:00").do(run_drop_detector)
        schedule.every().wednesday.at(f"{hour:02d}:00").do(run_drop_detector)
        schedule.every().thursday.at(f"{hour:02d}:00").do(run_drop_detector)
        schedule.every().friday.at(f"{hour:02d}:00").do(run_drop_detector)

    print("Scheduler started. Opportunity Hunter: weekdays 12:15. Drop Detector: weekdays 10:00–17:00 hourly.")
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            print(f"Scheduler job error: {e}")
        time.sleep(60)


# ---------- Legacy: track only & daily analysis (kept for compatibility) ----------

def run_track_only(config: Dict[str, Any]) -> None:
    target_prices = config.get("target_prices", {})
    if not target_prices:
        print("target_prices is empty.")
        return
    tickers = list(target_prices.keys())
    try:
        snapshots = build_snapshots(
            tickers=tickers, target_prices=target_prices, period="1mo"
        )
    except Exception as e:
        print(f"Track only error: {e}")
        return
    states = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "targets": {},
    }
    for s in snapshots:
        status = "IN_RANGE"
        if s.target_low is not None and s.price <= s.target_low:
            status = "BELOW_LOW"
        if s.target_high is not None and s.price >= s.target_high:
            status = "ABOVE_HIGH"
        states["targets"][s.ticker] = {
            "price": s.price,
            "status": status,
            "purchase_price": s.purchase_price,
            "low": s.target_low,
            "high": s.target_high,
        }
    write_json(STOCK_STATES_PATH, states)
    print("stock_states.json updated.")


def run_daily_analysis(config: Dict[str, Any], universe: str) -> None:
    """Legacy daily analysis: runs debate for all tickers and positions, saves recommendations and sends one email (only STRONG BUY/SELL after debate)."""
    target_prices = config.get("target_prices", {})
    tickers = load_universe(config, universe=universe)
    if not tickers:
        print("No ticker list found (config or BIST source).")
        return
    try:
        snapshots = build_snapshots(
            tickers=tickers, target_prices=target_prices, period="3mo"
        )
    except Exception as e:
        print(f"Daily analysis data error: {e}")
        return
    if not snapshots:
        print("Market data could not be fetched.")
        return

    # Single debate for BUY opportunities on full universe; then filter owned for SELL
    approved_buy = run_debate_loop(snapshots, "BUY")
    owned = [s for s in snapshots if s.purchase_price is not None]
    approved_sell = run_debate_loop(owned, "SELL") if owned else []

    all_recs = approved_buy + approved_sell
    write_json(
        "daily_recommendations.json",
        {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "gemini_model": GEMINI_MODEL,
            "openai_model": OPENAI_MODEL,
            "recommendations": all_recs,
        },
    )
    if all_recs:
        subject, body = format_opportunity_email_plain_english(
            all_recs,
            "Daily Recommendation — Approved Signals",
            target_prices=target_prices,
        )
        send_email(subject, body)
    else:
        print("No STRONG BUY/STRONG SELL agreements; no email sent.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stock Market Watcher — AI debate-based BUY/SELL + Scheduler"
    )
    p.add_argument(
        "mode",
        nargs="?",
        default="track",
        choices=["track", "daily_analysis", "scheduler", "opportunity_hunter", "drop_detector"],
        help="track: update stock_states only. daily_analysis: one-shot debate + email. scheduler: run daemon. opportunity_hunter: one-shot. drop_detector: one-shot.",
    )
    p.add_argument(
        "--universe",
        default="config",
        choices=["config", "bist", "auto"],
        help="Universe for daily_analysis: config, bist, or auto.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(CONFIG_PATH)

    if args.mode == "scheduler":
        run_scheduler()
        return
    if args.mode == "opportunity_hunter":
        run_opportunity_hunter()
        return
    if args.mode == "drop_detector":
        run_drop_detector()
        return
    if args.mode == "track":
        run_track_only(config)
    else:
        run_daily_analysis(config, universe=args.universe)


if __name__ == "__main__":
    main()
