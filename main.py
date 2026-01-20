import yfinance as yf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")

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
    "ZOREN.IS"
]


def send_email(alert_message):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = "🚨 BIST Stock Alert: Price Drop Detected"

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
    alerts = []
    print("Checking stocks...")

    for symbol in tickers:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")

            if hist.empty:
                continue

            current_price = hist["Close"].iloc[-1]
            yearly_avg = hist["Close"].mean()
            threshold = yearly_avg * 0.80

            if current_price < threshold:
                diff = ((yearly_avg - current_price) / yearly_avg) * 100
                info = (
                    f"⚠️ STOCK: {symbol}\n"
                    f"   Price: {current_price:.2f} TL\n"
                    f"   Yearly Avg: {yearly_avg:.2f} TL\n"
                    f"   Drop: %{diff:.2f} below average!\n"
                    f"--------------------------------"
                )
                alerts.append(info)

        except Exception:
            continue

    if alerts:
        full_text = (
            "The following stocks have dropped 20% below their yearly average:\n\n"
            + "\n".join(alerts)
        )
        send_email(full_text)
    else:
        print("No stocks found matching criteria.")


if __name__ == "__main__":
    check_stocks()
