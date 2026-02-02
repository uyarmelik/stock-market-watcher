import os
import sys
import json
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

try:
    import yfinance as yf
    import google.generativeai as genai
    from openai import OpenAI
except ImportError as e:
    print(f"Library missing: {e}")
    sys.exit(1)

load_dotenv()
ENV = os.environ
KEY_GEMINI = ENV.get("GEMINI_API_KEY")
KEY_OPENAI = ENV.get("OPENAI_API_KEY")
MAIL_USER = ENV.get("EMAIL_USER")
MAIL_PASS = ENV.get("EMAIL_PASS")
MAIL_RCVR = ENV.get("RECEIVER_EMAIL")


def get_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_market_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        if hist.empty:
            return None
        return {
            "symbol": ticker,
            "price": round(hist["Close"].iloc[-1], 2),
            "prev": round(hist["Close"].iloc[-2], 2) if len(hist) > 1 else 0,
        }
    except:
        return None


def ask_gemini(prompt):
    if not (genai and KEY_GEMINI):
        return "Gemini API Key missing."
    try:
        genai.configure(api_key=KEY_GEMINI)

        model_name = "gemini-1.5-flash-latest"
        try:
            model = genai.GenerativeModel(model_name)
        except:
            model = genai.GenerativeModel("gemini-pro")

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        response = model.generate_content(prompt, safety_settings=safety_settings)

        if response.parts:
            return response.text
        elif response.prompt_feedback:
            return f"Gemini Did Not Respond (Blocked): {response.prompt_feedback}"
        else:
            return "Gemini returned empty response."

    except Exception as e:
        return f"Gemini Error: {str(e)}"


def ask_openai(context):
    if not (OpenAI and KEY_OPENAI):
        return "OpenAI API Key missing."
    try:
        client = OpenAI(api_key=KEY_OPENAI)
        return (
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"Critique the analysis below, specify risks and write a short conclusion:\n{context}",
                    }
                ],
            )
            .choices[0]
            .message.content
        )
    except Exception as e:
        return f"OpenAI Error: {str(e)}"


def send_report(body):
    if not (MAIL_USER and MAIL_PASS and MAIL_RCVR):
        return
    msg = MIMEMultipart()
    msg["Subject"] = f"Stock Market Report - {datetime.now().strftime('%d-%m-%Y')}"
    msg["From"] = MAIL_USER
    msg["To"] = MAIL_RCVR
    msg.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(MAIL_USER, MAIL_PASS)
            s.send_message(msg)
    except Exception as e:
        print(f"Mail Error: {e}")


def main():
    config = get_json("config.json")
    report = f"DAILY FINANCE REPORT\n{'='*30}\n\n"

    targets = config.get("target_prices", {})
    if targets:
        report += "SCENARIO 1: PORTFOLIO ANALYSIS\n" + "-" * 30 + "\n"
        for t, info in targets.items():
            data = get_market_data(t)
            if not data:
                continue
            cost = info.get("buy_price", "Unknown")
            prompt = f"Stock: {t}. Cost: {cost}. Price: {data['price']}. Buy/Sell/Hold?"
            gemini_res = ask_gemini(prompt)
            chatgpt_res = ask_openai(gemini_res)
            report += f"[{t}] Price: {data['price']}\nANALYST (Gemini): {gemini_res[:400]}...\nRISK (ChatGPT): {chatgpt_res}\n\n"

    tickers = config.get("tickers", [])
    if tickers:
        report += "SCENARIO 2: MARKET SCAN (TOP 5)\n" + "-" * 30 + "\n"
        for t in tickers[:5]:
            data = get_market_data(t)
            if not data:
                continue
            gemini_res = ask_gemini(
                f"Stock: {t}. Price: {data['price']}. Is it a buying opportunity?"
            )
            report += f"[{t}] {gemini_res[:300]}...\n\n"

    send_report(report)


if __name__ == "__main__":
    main()
