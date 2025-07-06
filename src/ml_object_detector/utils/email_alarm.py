import os
import smtplib
import ssl
from ml_object_detector.utils.logging import setup_logs

log = setup_logs()

def send_alarm_email(run_id: str, n_images: int) -> bool:
    """
    Send an alert email when a YOLO run detects zero objects.
    SMTP credentials must be provided via environment variables.
    Returns True if the email was sent, False otherwise.
    """


    sender = os.getenv("SMTP_FROM")
    to = os.getenv("SMTP_TO")
    server = os.getenv("SMTP_HOST")
    port =  int(os.getenv("SMTP_PORT", 587))
    password = os.getenv("SMTP_PASSWORD")

    if not all([sender,to,server,password]):
        log.warning("Missing or incorrect SMTP configuration in enviroment variables")
        return

    body = (
        f"Subject: [YOLO alarm] NO detections for run {run_id}\n"
        f"From: {sender}\n"
        f"To: {to}\n\n"
        f"The detector processed {n_images} image(s) and found **zero** objects."
    )

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(server, port) as s:
            s.ehlo()
            s.starttls(context=context)
            s.ehlo()
            s.login(sender, password)
            s.sendmail(sender, to.split(","), body)
            log.info("[EMAIL ALARM] Email sent to %s", to)
            return True
    except Exception as e:
        log.warning("[EMAIL ALARM] Failed to send email: %s", e)
        return False
