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
    print(f"Kutuphane eksik: {e}")
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
        return {"symbol": ticker, "price": round(hist["Close"].iloc[-1], 2)}
    except:
        return None


def ask_gemini(prompt):
    if not (genai and KEY_GEMINI):
        return "Gemini API Key eksik."
    try:
        genai.configure(api_key=KEY_GEMINI)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Hatasi: {str(e)}"


def ask_openai(context):
    if not (OpenAI and KEY_OPENAI):
        return "OpenAI API Key eksik."
    try:
        client = OpenAI(api_key=KEY_OPENAI)
        return (
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"Analizi elestir ve riskleri belirt:\n{context}",
                    }
                ],
            )
            .choices[0]
            .message.content
        )
    except Exception as e:
        return f"OpenAI Hatasi: {str(e)}"


def send_report(body):
    if not (MAIL_USER and MAIL_PASS and MAIL_RCVR):
        return
    msg = MIMEMultipart()
    msg["Subject"] = f"Borsa Analizi - {datetime.now().strftime('%d-%m-%Y')}"
    msg["From"] = MAIL_USER
    msg["To"] = MAIL_RCVR
    msg.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(MAIL_USER, MAIL_PASS)
            s.send_message(msg)
    except Exception as e:
        print(f"Mail Hatasi: {e}")


def main():
    config = get_json("config.json")
    report = "GUNLUK FINANS ANALIZI\n" + "=" * 25 + "\n\n"

    targets = config.get("target_prices", {})
    for t, info in targets.items():
        data = get_market_data(t)
        if not data:
            continue

        prompt = f"Hisse: {t}. Maliyet: {info.get('buy_price')}. Fiyat: {data['price']}. Yorumla."
        gemini_res = ask_gemini(prompt)
        chatgpt_res = ask_openai(gemini_res)

        report += f"[{t}] Fiyat: {data['price']}\nAI: {gemini_res[:300]}...\nRISK: {chatgpt_res}\n\n"

    if targets:
        send_report(report)


if __name__ == "__main__":
    main()
