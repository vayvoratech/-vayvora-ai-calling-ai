import os
import smtplib
import imaplib
import email
from email.message import EmailMessage
from fastmcp import FastMCP

def register_mail_tools(mcp: FastMCP):
    MAIL_HOST = os.getenv("MAIL_HOST", "localhost")
    MAIL_PORT_SMTP = int(os.getenv("MAIL_PORT_SMTP", "587"))
    MAIL_PORT_IMAP = int(os.getenv("MAIL_PORT_IMAP", "993"))
    MAIL_USER = os.getenv("MAIL_USER", "")
    MAIL_PASS = os.getenv("MAIL_PASS", "")

    @mcp.tool()
    async def mail_send(to: str, subject: str, body: str) -> str:
        """Send an email using standard SMTP natively via Python."""
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = MAIL_USER
        msg['To'] = to

        try:
            # Connect to local/self-hosted SMTP server directly
            with smtplib.SMTP(MAIL_HOST, MAIL_PORT_SMTP) as server:
                server.starttls()
                server.login(MAIL_USER, MAIL_PASS)
                server.send_message(msg)
            return f"Successfully sent email to {to}."
        except Exception as e:
            raise RuntimeError(f"Failed to send email: {str(e)}")

    @mcp.tool()
    async def mail_read_recent(limit: int = 5) -> str:
        """Read recent emails from your self-hosted IMAP inbox natively."""
        try:
            mail = imaplib.IMAP4_SSL(MAIL_HOST, MAIL_PORT_IMAP)
            mail.login(MAIL_USER, MAIL_PASS)
            mail.select("inbox")

            status, messages = mail.search(None, "ALL")
            if status != "OK":
                return "No messages found."

            mail_ids = messages[0].split()
            recent_ids = mail_ids[-limit:]
            
            output = []
            for num in recent_ids:
                status, data = mail.fetch(num, "(RFC822)")
                if status == "OK":
                    msg = email.message_from_bytes(data[0][1])
                    output.append(f"From: {msg['From']} | Subject: {msg['Subject']}")

            mail.logout()
            return "\n".join(output)
        except Exception as e:
            raise RuntimeError(f"Failed to read mail via IMAP: {str(e)}")