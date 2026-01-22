import yfinance as yf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import json

EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")
STATE_FILE = "stock_states.json"

tickers = [
    "AEFES.IS",
    "AGHOL.IS",
    "AKBNK.IS",
    "AKSA.IS",
    "AKSEN.IS",
    "ALARK.IS",
    "ALTNY.IS",
    "ANSGR.IS",
    "ARCLK.IS",
    "ASELS.IS",
    "ASTOR.IS",
    "BALSU.IS",
    "BIMAS.IS",
    "BRSAN.IS",
    "BRYAT.IS",
    "BSOKE.IS",
    "BTCIM.IS",
    "CANTE.IS",
    "CCOLA.IS",
    "CIMSA.IS",
    "CWENE.IS",
    "DAPGM.IS",
    "DOAS.IS",
    "DOHOL.IS",
    "DSTKF.IS",
    "ECILC.IS",
    "EFOR.IS",
    "EGEEN.IS",
    "EKGYO.IS",
    "ENERY.IS",
    "ENJSA.IS",
    "ENKAI.IS",
    "EREGL.IS",
    "EUPWR.IS",
    "FROTO.IS",
    "GARAN.IS",
    "GENIL.IS",
    "GESAN.IS",
    "GLRMK.IS",
    "GRSEL.IS",
    "GRTHO.IS",
    "GUBRF.IS",
    "HALKB.IS",
    "HEKTS.IS",
    "ISCTR.IS",
    "ISMEN.IS",
    "IZENR.IS",
    "KCAER.IS",
    "KCHOL.IS",
    "KLRHO.IS",
    "KONTR.IS",
    "KRDMD.IS",
    "KTLEV.IS",
    "KUYAS.IS",
    "MAGEN.IS",
    "MAVI.IS",
    "MGROS.IS",
    "MIATK.IS",
    "MPARK.IS",
    "OBAMS.IS",
    "ODAS.IS",
    "OTKAR.IS",
    "OYAKC.IS",
    "PASEU.IS",
    "PATEK.IS",
    "PETKM.IS",
    "PGSUS.IS",
    "QUAGR.IS",
    "RALYH.IS",
    "REEDR.IS",
    "SAHOL.IS",
    "SASA.IS",
    "SISE.IS",
    "SKBNK.IS",
    "SOKM.IS",
    "TABGD.IS",
    "TAVHL.IS",
    "TCELL.IS",
    "THYAO.IS",
    "TKFEN.IS",
    "TOASO.IS",
    "TRALT.IS",
    "TRENJ.IS",
    "TRMET.IS",
    "TSKB.IS",
    "TSPOR.IS",
    "TTKOM.IS",
    "TTRAK.IS",
    "TUKAS.IS",
    "TUPRS.IS",
    "TUREX.IS",
    "TURSG.IS",
    "ULKER.IS",
    "VAKBN.IS",
    "YEOTK.IS",
    "YKBNK.IS",
    "ZOREN.IS",
]


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_state(states):
    with open(STATE_FILE, "w") as f:
        json.dump(states, f)


def send_email(subject, alert_message):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = subject

    msg.attach(MIMEText(alert_message, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")


def check_stocks():
    avg_alerts = []
    max_alerts = []
    current_states = load_state()
    new_states = {}

    print("Checking stocks...")

    for symbol in tickers:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")

            if hist.empty:
                continue

            current_price = hist["Close"].iloc[-1]
            yearly_avg = hist["Close"].mean()
            yearly_max = hist["Close"].max()
            avg_threshold = yearly_avg * 0.80
            max_threshold = yearly_max * 0.70

            prev_state = current_states.get(symbol, {})
            if isinstance(prev_state, bool):
                prev_state = {"avg_below": prev_state, "max_below": False}

            was_below_avg = prev_state.get("avg_below", False)
            was_below_max = prev_state.get("max_below", False)

            is_below_avg = current_price < avg_threshold
            is_below_max = current_price < max_threshold

            new_states[symbol] = {"avg_below": is_below_avg, "max_below": is_below_max}

            if is_below_avg and not was_below_avg:
                diff = ((yearly_avg - current_price) / yearly_avg) * 100
                info = (
                    f"🔻 DROP ALERT: {symbol}\n"
                    f"   Price: {current_price:.2f} TL\n"
                    f"   Yearly Avg: {yearly_avg:.2f} TL\n"
                    f"   Status: {diff:.2f}% below average!\n"
                    f"--------------------------------"
                )
                avg_alerts.append(info)

            elif not is_below_avg and was_below_avg:
                info = (
                    f"✅ RECOVERY ALERT: {symbol}\n"
                    f"   Price: {current_price:.2f} TL\n"
                    f"   Yearly Avg: {yearly_avg:.2f} TL\n"
                    f"   Status: Climbed back above the 20% threshold.\n"
                    f"--------------------------------"
                )
                avg_alerts.append(info)

            if is_below_max and not was_below_max:
                diff = ((yearly_max - current_price) / yearly_max) * 100
                info = (
                    f"📉 DROP ALERT: {symbol}\n"
                    f"   Price: {current_price:.2f} TL\n"
                    f"   Yearly Max: {yearly_max:.2f} TL\n"
                    f"   Status: {diff:.2f}% below yearly maximum!\n"
                    f"--------------------------------"
                )
                max_alerts.append(info)

            elif not is_below_max and was_below_max:
                info = (
                    f"🚀 RECOVERY ALERT: {symbol}\n"
                    f"   Price: {current_price:.2f} TL\n"
                    f"   Yearly Max: {yearly_max:.2f} TL\n"
                    f"   Status: Climbed back above the 30% max threshold.\n"
                    f"--------------------------------"
                )
                max_alerts.append(info)

        except Exception as e:
            print(f"Error checking {symbol}: {e}")
            continue

    if avg_alerts:
        full_text = "Stock status changes (Average-based):\n\n" + "\n".join(avg_alerts)
        send_email("🚨 BIST Average Alert", full_text)
    else:
        print("No average-based status changes found.")

    if max_alerts:
        full_text = "Stock status changes (Max-based):\n\n" + "\n".join(max_alerts)
        send_email("📉 BIST Max Drop Alert", full_text)
    else:
        print("No max-based status changes found.")

    save_state(new_states)


if __name__ == "__main__":
    check_stocks()
