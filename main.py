import argparse
import json
import os
import re
import smtplib
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

load_dotenv()

ENV = os.environ
KEY_GEMINI = ENV.get("GEMINI_API_KEY")
KEY_OPENAI = ENV.get("OPENAI_API_KEY")
MAIL_USER = ENV.get("EMAIL_USER")
MAIL_PASS = ENV.get("EMAIL_PASS")
MAIL_RCVR = ENV.get("RECEIVER_EMAIL")

GEMINI_MODEL = ENV.get("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = ENV.get("OPENAI_MODEL", "gpt-4o-mini")

BIST_EQUITY_CSV_URL = ENV.get(
    "BIST_EQUITY_CSV_URL", "https://www.borsaistanbul.com/datum/hisse_endeks_ds.csv"
)


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
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return m.group(0)

    return None


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
    target_low: Optional[float],
    target_high: Optional[float],
) -> str:
    if target_low is not None and price <= target_low:
        return "AL"
    if target_high is not None and price >= target_high:
        return "SAT"

    if sma20 is None or sma50 is None or rsi14 is None:
        return "TUT"

    if price > sma20 > sma50 and 55 <= rsi14 <= 70:
        return "AL"
    if price < sma20 < sma50 and rsi14 <= 45:
        return "SAT"
    return "TUT"


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
    if s.target_low is not None:
        parts.append(f"LOW={s.target_low:.2f}")
    if s.target_high is not None:
        parts.append(f"HIGH={s.target_high:.2f}")

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


def download_history(tickers: List[str], period: str = "3mo") -> Optional[pd.DataFrame]:
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        threads=True,
        auto_adjust=False,
        progress=False,
    )
    if df is None:
        return pd.DataFrame()
    return df


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
    if df is None:
        return []

    snapshots: List[StockSnapshot] = []
    for t in tickers:
        close = get_close_series(df, t)
        if close is None or close.empty:
            continue

        price = float(close.iloc[-1])
        sma20 = (
            safe_float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
        )
        sma50 = (
            safe_float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        )
        rsi14 = compute_rsi(close, 14)
        r5 = pct_return(close, 5)
        r21 = pct_return(close, 21)

        tp = target_prices.get(t, {}) if isinstance(target_prices, dict) else {}
        purchase_price = safe_float(tp.get("purchase_price"))
        low = safe_float(tp.get("low"))
        high = safe_float(tp.get("high"))

        base = baseline_action(price, sma20, sma50, rsi14, low, high)

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

    return snapshots


def gemini_json_recommendations(lines: List[str]) -> Optional[List[Dict[str, Any]]]:
    if not KEY_GEMINI:
        print("GEMINI_API_KEY is missing.")
        return None

    prompt = (
        "You are a short daily recommendation assistant for BIST equities.\n"
        "Rules:\n"
        "- Return only JSON (no other text).\n"
        "- action must be one of: AL, SAT, TUT\n"
        "- reason in Turkish, at most 20 words.\n"
        "- If unsure, choose TUT.\n"
        "- BASE field (heuristic) is reference only; do not copy blindly.\n\n"
        "Produce a JSON array in this format:\n"
        '[{"ticker":"XXX.IS","action":"AL|SAT|TUT","reason":"..."}, ...]\n\n'
        "Data lines:\n" + "\n".join(lines)
    )

    try:
        client = genai.Client(api_key=KEY_GEMINI)
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = getattr(resp, "text", "")
        block = extract_json_block(text)
        if not block:
            print("Gemini JSON not found.")
            return None
        return json.loads(block)
    except Exception as e:
        print(f"Gemini error: {e}")
        return None


def openai_review_json(
    snapshots: List[StockSnapshot],
    gemini_json: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    if not KEY_OPENAI:
        print("OPENAI_API_KEY is missing.")
        return None

    compact_data = [
        {
            "ticker": s.ticker,
            "price": s.price,
            "rsi": s.rsi14,
            "sma20": s.sma20,
            "sma50": s.sma50,
            "r5": s.ret_5d_pct,
            "r21": s.ret_21d_pct,
            "baseline": s.baseline,
            "purchase": s.purchase_price,
            "low": s.target_low,
            "high": s.target_high,
        }
        for s in snapshots
    ]

    msg = (
        "Review the data and Gemini output below.\n"
        "Task: For each ticker, fix AL/SAT/TUT if there is a logic error and keep reason under 20 words.\n"
        "Return only JSON, no other text.\n\n"
        f"DATA={json.dumps(compact_data, ensure_ascii=False)}\n\n"
        f"GEMINI={json.dumps(gemini_json, ensure_ascii=False)}"
    )

    try:
        client = OpenAI(api_key=KEY_OPENAI)
        res = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Output only JSON. action=AL|SAT|TUT; reason <= 20 words, in Turkish.",
                },
                {"role": "user", "content": msg},
            ],
            temperature=0.2,
        )
        text = res.choices[0].message.content or ""
        block = extract_json_block(text)
        if not block:
            print("OpenAI JSON not found.")
            return None
        return json.loads(block)
    except Exception as e:
        print(f"OpenAI error: {e}")
        return None


def send_email(subject: str, body: str) -> None:
    if not (MAIL_USER and MAIL_PASS and MAIL_RCVR):
        print("Mail env variables missing (EMAIL_USER/EMAIL_PASS/RECEIVER_EMAIL).")
        return

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
    except Exception as e:
        print(f"Mail error: {e}")


def format_daily_email(recs: List[Dict[str, Any]]) -> Tuple[str, str]:
    today = datetime.now().strftime("%d-%m-%Y")

    counts = {"AL": 0, "SAT": 0, "TUT": 0}
    for r in recs:
        a = str(r.get("action", "TUT")).upper()
        if a in counts:
            counts[a] += 1

    lines = [
        f"BIST Daily BUY/SELL/HOLD - {today}",
        f"Summary: AL={counts['AL']} | SAT={counts['SAT']} | TUT={counts['TUT']}",
        "",
        "Format: TICKER: ACTION — reason (<=20 words)",
        "",
    ]

    for r in recs:
        t = r.get("ticker", "")
        a = str(r.get("action", "TUT")).upper()
        reason = str(r.get("reason", "")).strip()
        reason_words = reason.split()
        if len(reason_words) > 20:
            reason = " ".join(reason_words[:20])
        lines.append(f"{t}: {a} — {reason}")

    lines.append("")
    lines.append(
        "Note: This output is not investment advice; for informational purposes only."
    )

    subject = f"BIST Daily Recommendation (AL/SAT/TUT) - {today}"
    return subject, "\n".join(lines)


def run_track_only(config: Dict[str, Any]) -> None:
    target_prices = config.get("target_prices", {})
    if not target_prices:
        print("target_prices is empty.")
        return

    tickers = list(target_prices.keys())
    snapshots = build_snapshots(
        tickers=tickers, target_prices=target_prices, period="1mo"
    )

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

    write_json("stock_states.json", states)
    print("stock_states.json updated.")


def run_daily_analysis(config: Dict[str, Any], universe: str) -> None:
    target_prices = config.get("target_prices", {})
    tickers = load_universe(config, universe=universe)

    if not tickers:
        print("No ticker list found (config or BIST source).")
        return

    snapshots = build_snapshots(
        tickers=tickers, target_prices=target_prices, period="3mo"
    )
    if not snapshots:
        print("Market data could not be fetched.")
        return

    lines = [format_snapshot_line(s) for s in snapshots]

    gemini = gemini_json_recommendations(lines)
    if not gemini:
        print("No Gemini output; continuing with baseline.")
        gemini = [
            {
                "ticker": s.ticker,
                "action": s.baseline,
                "reason": "Technical view / target band",
            }
            for s in snapshots
        ]

    final = openai_review_json(snapshots, gemini) or gemini

    write_json(
        "daily_recommendations.json",
        {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "gemini_model": GEMINI_MODEL,
            "openai_model": OPENAI_MODEL,
            "recommendations": final,
        },
    )

    subject, body = format_daily_email(final)
    send_email(subject, body)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BIST tracker + daily AL/SAT/TUT recommender"
    )
    p.add_argument(
        "mode",
        nargs="?",
        default="track",
        choices=["track", "daily_analysis"],
        help="track: update target_prices state file only (no email). daily_analysis: single email, all tickers.",
    )
    p.add_argument(
        "--universe",
        default="config",
        choices=["config", "bist", "auto"],
        help="config: config.json tickers; bist: Borsa Istanbul CSV; auto: use BIST if config empty.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json("config.json")

    if args.mode == "track":
        run_track_only(config)
    else:
        run_daily_analysis(config, universe=args.universe)


if __name__ == "__main__":
    main()
