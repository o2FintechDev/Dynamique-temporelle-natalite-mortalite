from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

def send_pdf_mail(*, to: str, subject: str, body: str, pdf_path: Path) -> None:
    host = os.environ["EMAIL_HOST"]
    port = int(os.environ.get("EMAIL_PORT", "587"))
    user = os.environ["EMAIL_USER"]
    pwd = os.environ["EMAIL_PASS"]

    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    msg.add_attachment(
        pdf_path.read_bytes(),
        maintype="application",
        subtype="pdf",
        filename=pdf_path.name,
    )

    with smtplib.SMTP(host, port) as s:
        s.starttls()
        s.login(user, pwd)
        s.send_message(msg)
