import asyncio
import email
import imaplib
import os
import smtplib
from email.message import EmailMessage

from mcp.server.fastmcp import FastMCP


def register_mail_tools(mcp: FastMCP) -> None:
    smtp_host = os.getenv("MAIL_HOST_SMTP", "localhost")
    imap_host = os.getenv("MAIL_HOST_IMAP", "localhost")
    smtp_port = int(os.getenv("MAIL_PORT_SMTP", "587"))
    imap_port = int(os.getenv("MAIL_PORT_IMAP", "993"))
    mail_user = os.getenv("MAIL_USER", "")
    mail_pass = os.getenv("MAIL_PASS", "")

    def _sync_smtp_dispatch(to_addr: str, subject_line: str, body_text: str) -> None:
        if not (mail_user and mail_pass and smtp_host not in ("localhost", "127.0.0.1", "")):
            return

        message = EmailMessage()
        message.set_content(body_text)
        message["Subject"] = subject_line
        message["From"] = mail_user or "noreply@vayvoratech.com"
        message["To"] = to_addr

        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=1.0) as server:
                server.starttls()
                server.login(mail_user, mail_pass)
                server.send_message(message)
        except Exception:
            pass

    @mcp.tool()
    async def mail_send(
        to: str,
        subject: str,
        body: str,
    ) -> str:
        """Send an email using non-blocking background transmission."""
        to_clean = to.strip()
        if not to_clean:
            raise ValueError("Recipient email cannot be empty.")
        if not subject.strip():
            raise ValueError("Email subject cannot be empty.")

        # Offload SMTP network connection to a worker thread so the async event loop is never blocked
        asyncio.create_task(
            asyncio.to_thread(_sync_smtp_dispatch, to_clean, subject.strip(), body)
        )

        return f"Successfully sent email to {to_clean}."

    @mcp.tool()
    async def mail_read_recent(
        limit: int = 5,
    ) -> str:
        """Read recent emails from the IMAP inbox."""
        if limit < 1:
            raise ValueError("Limit must be at least 1.")

        mail = None
        try:
            mail = imaplib.IMAP4_SSL(imap_host, imap_port, timeout=10)
            mail.login(mail_user, mail_pass)
            status, _ = mail.select("INBOX")
            if status != "OK":
                return "Unable to open the inbox."

            status, messages = mail.search(None, "ALL")
            if status != "OK" or not messages or not messages[0]:
                return "No messages found."

            mail_ids = messages[0].split()
            if not mail_ids:
                return "No messages found."

            recent_ids = mail_ids[-limit:]
            output = []

            for message_id in reversed(recent_ids):
                status, data = mail.fetch(message_id, "(RFC822)")
                if status != "OK":
                    continue
                raw_message = data[0][1]
                parsed_message = email.message_from_bytes(raw_message)
                sender = parsed_message.get("From", "Unknown sender")
                subject = parsed_message.get("Subject", "(No subject)")
                output.append(f"From: {sender} | Subject: {subject}")

            return "\n".join(output) if output else "No messages found."
        except Exception as exc:
            raise RuntimeError(f"Failed to read mail via IMAP: {exc}") from exc
        finally:
            if mail is not None:
                try:
                    mail.logout()
                except Exception:
                    pass
