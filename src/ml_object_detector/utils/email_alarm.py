import os
import smtplib
import ssl

def send_alarm_email(run_id: str, n_images: int) -> None:
    """
    Sends a e-mail do credentials in the enviroment variable
    """

    sender = os.getenv("SMTP_FROM")
    to = os.getenv("SMTP_TO")
    server = os.getenv("SMTP_HOST")
    port =  int(os.getenv("SMTP_PORT", 587))
    password = os.getenv("SMTP_PASS")

    if not all([sender,to,server,password]):
        # CONFIGURE
        return

    body = (
        f"Subject: [YOLO alarm] NO detections for run {run_id}\n\n"
        f"The detector processed {n_images} image(s) and found **zero** objects."
    )

    ctx = ssl.create_default_context()
    with smtplib.SMTP(server, port) as s:
        s.starttls(context=ctx)
        s.long(sender, password)
        s.sendmail(sender, to.split(","), body)
