from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


def _env_required(key: str) -> str:
    v = os.getenv(key)
    if not v:
        raise RuntimeError(f"Missing env var: {key}")
    return v


def send_pdf_mail(*, to: str, subject: str, body: str, pdf_path: Path) -> None:
    host = _env_required("EMAIL_HOST")
    port = int(os.getenv("EMAIL_PORT", "587"))
    user = _env_required("EMAIL_USER")
    pwd = _env_required("EMAIL_PASSWORD")  # <-- standard

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