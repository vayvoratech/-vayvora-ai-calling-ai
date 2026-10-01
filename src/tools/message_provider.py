"""Messaging and WhatsApp provider implementations for external MCP tool execution.

Provides WhatsAppMessageProvider communicating with a self-hosted OpenWA instance,
and MockMessageProvider for isolated unit and integration testing. Reuses and adapts
phone normalization and REST dispatch from legacy MCP WhatsApp tools.
"""

from abc import ABC, abstractmethod
import asyncio
import re
import time
from typing import Any, Dict, List, Optional
import uuid
import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.config import Settings, get_settings
from src.logging import get_logger

logger = get_logger("tools.message_provider")


def normalize_phone_number(raw_phone: str) -> str:
    """Clean phone numbers into standard digits format (e.g. 919949350699).

    Prepends '91' country code if a 10-digit number is provided.
    Handles numbers with leading 0 (11 digits).
    """
    cleaned = re.sub(r"[^\d]", "", raw_phone.strip())
    if len(cleaned) == 10:
        cleaned = f"91{cleaned}"
    elif cleaned.startswith("0") and len(cleaned) == 11:
        cleaned = f"91{cleaned[1:]}"
    return cleaned


class MessageSendResult(BaseModel):
    """Structured result returned by messaging provider implementations."""

    model_config = ConfigDict(extra="forbid")

    success: bool = Field(..., description="True if message was accepted by gateway")
    message_id: Optional[str] = Field(
        default=None,
        description="External tracking ID or generated message identifier",
    )
    recipient: str = Field(default="", description="Target destination number or chat ID")
    error: Optional[str] = Field(default=None, description="Detailed error description if delivery failed")
    status: str = Field(
        default="pending",
        description="Result classification: delivered, queued, auth_failed, connection_failed, timeout, failed, mock_sent",
    )


class MessageProvider(ABC):
    """Abstract interface for instant messaging and SMS dispatch."""

    @abstractmethod
    async def send_message(
        self,
        recipient: str,
        message: str,
    ) -> MessageSendResult:
        """Send message asynchronously."""
        pass


class WhatsAppMessageProvider(MessageProvider):
    """Production WhatsApp provider communicating with OpenWA instance over REST."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        session_id: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_url = (base_url or getattr(self.settings, "openwa_base_url", "http://localhost:2785")).rstrip("/")
        self._explicit_api_key = api_key
        self.session_id = session_id or getattr(self.settings, "whatsapp_session_id", "default")
        self.timeout = timeout

    def _get_api_key(self) -> Optional[str]:
        if self._explicit_api_key:
            return self._explicit_api_key
        key = getattr(self.settings, "openwa_api_key", None)
        if key:
            return key.get_secret_value()
        return None

    async def send_message(
        self,
        recipient: str,
        message: str,
    ) -> MessageSendResult:
        """Dispatch WhatsApp text message via OpenWA session endpoint."""
        clean_num = normalize_phone_number(recipient)
        if not clean_num:
            return MessageSendResult(
                success=False,
                recipient=recipient,
                error="Recipient phone number cannot be empty or invalid.",
                status="failed",
            )
        if not message or not message.strip():
            return MessageSendResult(
                success=False,
                recipient=clean_num,
                error="Message content cannot be empty.",
                status="failed",
            )

        chat_id = f"{clean_num}@c.us"
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        api_key = self._get_api_key()
        if api_key:
            headers["X-API-Key"] = api_key

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # 1. Discover ready session ID if possible
                effective_session = self.session_id
                try:
                    sess_res = await client.get(
                        f"{self.base_url}/api/sessions",
                        headers=headers,
                        timeout=min(self.timeout, 5.0),
                    )
                    if sess_res.status_code == 200:
                        sessions = sess_res.json()
                        if isinstance(sessions, list) and len(sessions) > 0:
                            ready_sess = next((s for s in sessions if s.get("status") == "ready"), sessions[0])
                            effective_session = ready_sess.get("id") or effective_session
                except Exception as sess_err:
                    logger.debug("Could not query OpenWA sessions list, using default session: %s", sess_err)

                # 2. Dispatch text message
                url = f"{self.base_url}/api/sessions/{effective_session}/messages/send-text"
                payload = {
                    "chatId": chat_id,
                    "text": message.strip(),
                }

                logger.info("Dispatching WhatsApp via %s to %s", url, chat_id)
                response = await client.post(url, json=payload, headers=headers)

                if response.status_code in (200, 201):
                    resp_json = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                    ext_id = resp_json.get("id") or resp_json.get("messageId") or f"msg_wa_{uuid.uuid4().hex[:12]}"
                    logger.info("WhatsApp message successfully dispatched to %s (id: %s)", clean_num, ext_id)
                    return MessageSendResult(
                        success=True,
                        message_id=ext_id,
                        recipient=clean_num,
                        status="delivered",
                    )
                elif response.status_code in (401, 403):
                    logger.error("OpenWA authentication failure HTTP %d", response.status_code)
                    return MessageSendResult(
                        success=False,
                        recipient=clean_num,
                        error=f"WhatsApp gateway authentication failed HTTP {response.status_code}",
                        status="auth_failed",
                    )
                else:
                    logger.error("OpenWA gateway returned error HTTP %d: %s", response.status_code, response.text)
                    return MessageSendResult(
                        success=False,
                        recipient=clean_num,
                        error=f"WhatsApp delivery failed HTTP {response.status_code}",
                        status="failed",
                    )

        except httpx.TimeoutException:
            logger.error("WhatsApp delivery to %s timed out after %.1fs", clean_num, self.timeout)
            return MessageSendResult(
                success=False,
                recipient=clean_num,
                error=f"WhatsApp dispatch timed out after {self.timeout}s",
                status="timeout",
            )
        except (httpx.ConnectError, OSError) as exc:
            logger.error("Failed to connect to OpenWA gateway at %s: %s", self.base_url, exc)
            return MessageSendResult(
                success=False,
                recipient=clean_num,
                error=f"Cannot connect to WhatsApp gateway: {exc}",
                status="connection_failed",
            )
        except Exception as exc:
            logger.error("Unexpected exception dispatching WhatsApp to %s: %s", clean_num, exc)
            return MessageSendResult(
                success=False,
                recipient=clean_num,
                error=f"WhatsApp dispatch exception: {exc}",
                status="failed",
            )


class MockMessageProvider(MessageProvider):
    """Deterministic in-memory mock message provider for unit testing."""

    def __init__(
        self,
        force_success: bool = True,
        force_auth_failure: bool = False,
        force_connection_failure: bool = False,
        force_timeout: bool = False,
    ) -> None:
        self.force_success = force_success
        self.force_auth_failure = force_auth_failure
        self.force_connection_failure = force_connection_failure
        self.force_timeout = force_timeout
        self.sent_messages: List[Dict[str, Any]] = []

    async def send_message(
        self,
        recipient: str,
        message: str,
    ) -> MessageSendResult:
        clean_num = normalize_phone_number(recipient)
        if not clean_num:
            return MessageSendResult(
                success=False,
                recipient=recipient,
                error="Recipient phone number cannot be empty or invalid.",
                status="failed",
            )

        if self.force_auth_failure:
            return MessageSendResult(
                success=False,
                recipient=clean_num,
                error="WhatsApp authentication failure: Invalid API token",
                status="auth_failed",
            )
        if self.force_connection_failure:
            return MessageSendResult(
                success=False,
                recipient=clean_num,
                error="Cannot connect to WhatsApp gateway: Connection refused",
                status="connection_failed",
            )
        if self.force_timeout:
            return MessageSendResult(
                success=False,
                recipient=clean_num,
                error="WhatsApp dispatch timed out after 10.0s",
                status="timeout",
            )

        if self.force_success:
            msg_id = f"msg_wa_mock_{uuid.uuid4().hex[:12]}"
            record = {
                "recipient": clean_num,
                "message": message,
                "message_id": msg_id,
                "timestamp": time.time(),
            }
            self.sent_messages.append(record)
            logger.info("MockMessageProvider recorded sent message to %s (id: %s)", clean_num, msg_id)
            return MessageSendResult(
                success=True,
                message_id=msg_id,
                recipient=clean_num,
                status="mock_sent",
            )

        return MessageSendResult(
            success=False,
            recipient=clean_num,
            error="Mock message dispatch configured to fail",
            status="failed",
        )
