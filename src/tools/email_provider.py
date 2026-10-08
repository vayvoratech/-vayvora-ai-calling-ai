"""SMTP Email Provider implementations and abstractions for external MCP tool execution.

Provides SMTPEmailProvider for authenticated real-world SMTP dispatch and
MockEmailProvider for isolated unit and integration testing without network calls.
"""

from abc import ABC, abstractmethod
import asyncio
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import imaplib
import smtplib
import socket
import time
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field

from src.config import Settings, get_settings
from src.core.decision import is_valid_email
from src.logging import get_logger

logger = get_logger("tools.email_provider")


class EmailSendResult(BaseModel):
    """Structured result returned by email provider implementations."""

    model_config = ConfigDict(extra="forbid")

    success: bool = Field(..., description="True if email was accepted by SMTP server")
    message_id: Optional[str] = Field(
        default=None,
        description="External tracking ID or generated RFC 2822 Message-ID",
    )
    error: Optional[str] = Field(default=None, description="Detailed error description if delivery failed")
    status: str = Field(
        default="pending",
        description="Result classification: sent, auth_failed, connection_failed, timeout, failed, mock_sent",
    )


class EmailReadResult(BaseModel):
    """Structured result returned when querying recent emails via IMAP."""

    model_config = ConfigDict(extra="forbid")

    success: bool = Field(..., description="True if mailbox was successfully read")
    messages: List[Dict[str, str]] = Field(default_factory=list, description="Parsed email message summaries")
    summary: str = Field(default="", description="Voice or text summary of retrieved emails")
    error: Optional[str] = Field(default=None, description="Detailed error description if query failed")
    status: str = Field(
        default="ok",
        description="Result classification: ok, empty, auth_failed, connection_failed, timeout, failed",
    )


class EmailProvider(ABC):
    """Abstract interface for transactional email dispatch and inbox access."""

    @abstractmethod
    async def send_email(
        self,
        recipient: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        sender: Optional[str] = None,
    ) -> EmailSendResult:
        """Send an email message asynchronously."""
        pass

    @abstractmethod
    async def read_recent_emails(
        self,
        limit: int = 5,
    ) -> EmailReadResult:
        """Read recent emails from inbox asynchronously."""
        pass


class SMTPEmailProvider(EmailProvider):
    """Production SMTP provider executing dispatch via smtplib in a background thread."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_email: Optional[str] = None,
        use_tls: Optional[bool] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.host = host or self.settings.smtp_host
        self.port = port if port is not None else self.settings.smtp_port
        self.username = username if username is not None else self.settings.smtp_username
        self._explicit_password = password
        self.from_email = from_email or self.settings.smtp_from_email
        self.use_tls = use_tls if use_tls is not None else self.settings.smtp_use_tls
        self.timeout = timeout if timeout is not None else self.settings.smtp_timeout_seconds

    def _get_password(self) -> Optional[str]:
        """Fetch password securely without logging."""
        if self._explicit_password:
            return self._explicit_password
        if self.settings.smtp_password:
            return self.settings.smtp_password.get_secret_value()
        return None

    def _send_sync(
        self,
        recipient: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        sender: Optional[str] = None,
    ) -> EmailSendResult:
        """Synchronous SMTP worker invoked within asyncio.to_thread."""
        if not recipient or not is_valid_email(recipient):
            logger.error("Attempted to dispatch email to invalid or prohibited recipient: %s", recipient)
            return EmailSendResult(
                success=False,
                error="Invalid, missing, or prohibited recipient email address",
                status="failed",
            )

        from_addr = sender or self.from_email
        msg_id_domain = from_addr.split("@")[-1] if "@" in from_addr else "vayvora.com"
        message_id = f"<{uuid.uuid4().hex[:16]}.{int(time.time())}@{msg_id_domain}>"

        # Build MIME payload
        msg: MIMEMultipart
        if html_body:
            msg = MIMEMultipart("alternative")
            msg.attach(MIMEText(body, "plain", "utf-8"))
            msg.attach(MIMEText(html_body, "html", "utf-8"))
        else:
            msg = MIMEMultipart()
            msg.attach(MIMEText(body, "plain", "utf-8"))

        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = recipient
        msg["Message-ID"] = message_id
        msg["Date"] = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())

        password = self._get_password()

        try:
            logger.info(
                "Connecting to SMTP host %s:%d (TLS=%s, from=%s)",
                self.host,
                self.port,
                self.use_tls,
                from_addr,
            )
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as server:
                server.ehlo()
                if self.use_tls:
                    server.starttls()
                    server.ehlo()

                if self.username and password:
                    server.login(self.username, password)

                refused = server.send_message(msg)

            # smtplib returns a dictionary of recipients rejected by the SMTP server.
            # Treat any refusal as a failed dispatch instead of falsely reporting success.
            if isinstance(refused, dict) and bool(refused):
                refused_detail = "; ".join(
                    f"{addr}: {detail!r}" for addr, detail in refused.items()
                )
                logger.error(
                    "SMTP server refused recipient(s) for Message-ID %s: %s",
                    message_id,
                    refused_detail,
                )
                return EmailSendResult(
                    success=False,
                    message_id=message_id,
                    error=f"SMTP recipient refused: {refused_detail}",
                    status="failed",
                )

            logger.info(
                "Email accepted by SMTP server for %s (Message-ID: %s)",
                recipient,
                message_id,
            )
            return EmailSendResult(
                success=True,
                message_id=message_id,
                status="sent",
            )
        except smtplib.SMTPAuthenticationError as exc:
            logger.error("SMTP authentication failed for user %s: %s", self.username, exc)
            return EmailSendResult(
                success=False,
                error=f"SMTP authentication failure: {exc}",
                status="auth_failed",
            )
        except (socket.timeout, TimeoutError) as exc:
            logger.error("SMTP operation timed out after %s seconds: %s", self.timeout, exc)
            return EmailSendResult(
                success=False,
                error=f"SMTP operation timed out after {self.timeout}s: {exc}",
                status="timeout",
            )
        except (ConnectionRefusedError, smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected, OSError) as exc:
            logger.error("SMTP connection to %s:%d failed: %s", self.host, self.port, exc)
            return EmailSendResult(
                success=False,
                error=f"SMTP connection failure to {self.host}:{self.port}: {exc}",
                status="connection_failed",
            )
        except smtplib.SMTPException as exc:
            logger.error("SMTP delivery error: %s", exc)
            return EmailSendResult(
                success=False,
                error=f"SMTP delivery error: {exc}",
                status="failed",
            )
        except Exception as exc:
            logger.error("Unexpected error dispatching SMTP email: %s", exc)
            return EmailSendResult(
                success=False,
                error=f"Unexpected error sending email: {exc}",
                status="failed",
            )

    async def send_email(
        self,
        recipient: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        sender: Optional[str] = None,
    ) -> EmailSendResult:
        """Asynchronously dispatch email via thread pool."""
        return await asyncio.to_thread(
            self._send_sync,
            recipient=recipient,
            subject=subject,
            body=body,
            html_body=html_body,
            sender=sender,
        )

    def _read_recent_sync(self, limit: int = 5) -> EmailReadResult:
        """Synchronously query recent emails from IMAP inbox."""
        if limit < 1:
            return EmailReadResult(
                success=False,
                error="Limit must be at least 1.",
                status="failed",
            )
        imap_host = getattr(self.settings, "imap_host", "imap.gmail.com")
        imap_port = getattr(self.settings, "imap_port", 993)
        imap_user = getattr(self.settings, "imap_username", None) or self.username
        imap_pass = None
        if getattr(self.settings, "imap_password", None):
            imap_pass = self.settings.imap_password.get_secret_value()
        else:
            imap_pass = self._get_password()

        if not (imap_user and imap_pass and imap_host not in ("localhost", "127.0.0.1", "")):
            logger.warning("IMAP credentials or host not configured. Returning empty inbox.")
            return EmailReadResult(
                success=True,
                messages=[],
                summary="IMAP inbox not configured or offline.",
                status="empty",
            )

        mail = None
        try:
            timeout = getattr(self.settings, "imap_timeout_seconds", 10.0)
            mail = imaplib.IMAP4_SSL(imap_host, imap_port, timeout=timeout)
            mail.login(imap_user, imap_pass)
            status, _ = mail.select("INBOX")
            if status != "OK":
                return EmailReadResult(
                    success=False,
                    error="Unable to open the inbox.",
                    status="failed",
                )

            status, messages = mail.search(None, "ALL")
            if status != "OK" or not messages or not messages[0]:
                return EmailReadResult(
                    success=True,
                    messages=[],
                    summary="No messages found in inbox.",
                    status="empty",
                )

            mail_ids = messages[0].split()
            if not mail_ids:
                return EmailReadResult(
                    success=True,
                    messages=[],
                    summary="No messages found in inbox.",
                    status="empty",
                )

            recent_ids = mail_ids[-limit:]
            records: List[Dict[str, str]] = []
            lines: List[str] = []

            for mid in reversed(recent_ids):
                status, data = mail.fetch(mid, "(RFC822)")
                if status != "OK" or not data or not data[0]:
                    continue
                raw_message = data[0][1]
                parsed_message = email.message_from_bytes(raw_message)
                sender = parsed_message.get("From", "Unknown sender")
                subject = parsed_message.get("Subject", "(No subject)")
                date_str = parsed_message.get("Date", "")
                records.append({
                    "id": mid.decode("utf-8") if isinstance(mid, bytes) else str(mid),
                    "from": sender,
                    "subject": subject,
                    "date": date_str,
                })
                lines.append(f"From: {sender} | Subject: {subject}")

            summary_text = "\n".join(lines) if lines else "No messages found."
            return EmailReadResult(
                success=True,
                messages=records,
                summary=summary_text,
                status="ok",
            )
        except imaplib.IMAP4.error as exc:
            logger.error("IMAP authentication or protocol error: %s", exc)
            return EmailReadResult(
                success=False,
                error=f"IMAP protocol failure: {exc}",
                status="auth_failed",
            )
        except (socket.timeout, TimeoutError) as exc:
            logger.error("IMAP read timed out: %s", exc)
            return EmailReadResult(
                success=False,
                error=f"IMAP operation timed out: {exc}",
                status="timeout",
            )
        except Exception as exc:
            logger.error("Failed to read mail via IMAP: %s", exc)
            return EmailReadResult(
                success=False,
                error=f"Failed to read mail via IMAP: {exc}",
                status="failed",
            )
        finally:
            if mail is not None:
                try:
                    mail.logout()
                except Exception:
                    pass

    async def read_recent_emails(self, limit: int = 5) -> EmailReadResult:
        """Asynchronously query recent emails from IMAP inbox."""
        return await asyncio.to_thread(self._read_recent_sync, limit=limit)


class MockEmailProvider(EmailProvider):
    """Deterministic in-memory mock email provider for unit testing without network dependencies."""

    def __init__(
        self,
        force_success: bool = True,
        force_auth_failure: bool = False,
        force_connection_failure: bool = False,
        force_timeout: bool = False,
        mock_inbox: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        self.force_success = force_success
        self.force_auth_failure = force_auth_failure
        self.force_connection_failure = force_connection_failure
        self.force_timeout = force_timeout
        self.sent_messages: List[Dict[str, Any]] = []
        self.mock_inbox: List[Dict[str, str]] = mock_inbox if mock_inbox is not None else [
            {"from": "counseling@edusaas.com", "subject": "Enrollment Inquiry Followup", "date": "2026-09-28"},
            {"from": "partners@vayvora.com", "subject": "Enterprise AI Voice Consultation", "date": "2026-09-29"},
        ]

    async def send_email(
        self,
        recipient: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        sender: Optional[str] = None,
    ) -> EmailSendResult:
        """Simulate email sending."""
        if not recipient or not is_valid_email(recipient):
            return EmailSendResult(
                success=False,
                error="Invalid, missing, or prohibited recipient email address",
                status="failed",
            )

        if self.force_auth_failure:
            return EmailSendResult(
                success=False,
                error="SMTP authentication failure: 535 Authentication credentials invalid",
                status="auth_failed",
            )
        if self.force_connection_failure:
            return EmailSendResult(
                success=False,
                error="SMTP connection failure: [Errno 111] Connection refused",
                status="connection_failed",
            )
        if self.force_timeout:
            return EmailSendResult(
                success=False,
                error="SMTP operation timed out after 10.0s",
                status="timeout",
            )

        if self.force_success:
            msg_id = f"<mock_{uuid.uuid4().hex[:12]}@vayvora.com>"
            record = {
                "recipient": recipient,
                "subject": subject,
                "body": body,
                "html_body": html_body,
                "sender": sender,
                "message_id": msg_id,
                "timestamp": time.time(),
            }
            self.sent_messages.append(record)
            logger.info("MockEmailProvider recorded sent email to %s (id: %s)", recipient, msg_id)
            return EmailSendResult(
                success=True,
                message_id=msg_id,
                status="mock_sent",
            )

        return EmailSendResult(
            success=False,
            error="Mock email dispatch configured to fail",
            status="failed",
        )

    async def read_recent_emails(self, limit: int = 5) -> EmailReadResult:
        """Simulate IMAP email reading."""
        if limit < 1:
            return EmailReadResult(
                success=False,
                error="Limit must be at least 1.",
                status="failed",
            )
        if self.force_auth_failure:
            return EmailReadResult(
                success=False,
                error="IMAP authentication failure",
                status="auth_failed",
            )
        if self.force_connection_failure:
            return EmailReadResult(
                success=False,
                error="IMAP connection refused",
                status="connection_failed",
            )
        if self.force_timeout:
            return EmailReadResult(
                success=False,
                error="IMAP operation timed out",
                status="timeout",
            )

        recent = self.mock_inbox[-limit:]
        lines = [f"From: {m.get('from', '')} | Subject: {m.get('subject', '')}" for m in reversed(recent)]
        return EmailReadResult(
            success=True,
            messages=recent,
            summary="\n".join(lines) if lines else "No messages found.",
            status="ok",
        )
