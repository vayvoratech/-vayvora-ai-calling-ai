"""Messaging and WhatsApp provider implementations for external MCP tool execution.

Provides WhatsAppMessageProvider communicating with a self-hosted OpenWA instance,
and MockMessageProvider for isolated unit and integration testing. Reuses and adapts
phone normalization and REST dispatch from legacy MCP WhatsApp tools.
"""

from abc import ABC, abstractmethod
import hashlib
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid
import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.config import Settings, get_settings
from src.logging import get_logger

logger = get_logger("tools.message_provider")


_idempotency_cache: Dict[str, Tuple[float, Any]] = {}
IDEMPOTENCY_TTL_SECONDS: float = 60.0


def _make_idempotency_key(scope: str, recipient: str, message: str) -> str:
    """Build a stable key without storing message text in the cache key."""
    digest = hashlib.sha256(message.strip().encode("utf-8")).hexdigest()
    return f"{scope}:{recipient}:{digest}"


def _get_cached_result(key: str, now: float) -> Optional[Any]:
    """Return an unexpired result and opportunistically prune expired entries."""
    for cached_key, (created_at, _) in list(_idempotency_cache.items()):
        if now - created_at >= IDEMPOTENCY_TTL_SECONDS:
            _idempotency_cache.pop(cached_key, None)
    entry = _idempotency_cache.get(key)
    if entry and now - entry[0] < IDEMPOTENCY_TTL_SECONDS:
        return entry[1]
    return None


def validate_phone_number(raw_phone: str) -> Tuple[bool, str, Optional[str]]:
    """Validate and normalize phone number into standard digits format.

    Rejects dummy, placeholder, repeating, or sequential numbers.
    Prepends '91' country code if a 10-digit number is provided.
    Returns (is_valid, normalized_phone, error_message).
    """
    if not raw_phone or not isinstance(raw_phone, str):
        return False, "", "Phone number cannot be empty."

    cleaned = re.sub(r"[^\d]", "", raw_phone.strip())
    if not cleaned:
        return False, "", "Phone number must contain digits."

    # Handle leading zero for domestic 11-digit numbers
    if cleaned.startswith("0") and len(cleaned) == 11:
        cleaned = f"91{cleaned[1:]}"
    elif len(cleaned) == 10:
        cleaned = f"91{cleaned}"

    # Verify standard international length (10 to 15 digits)
    if len(cleaned) < 10 or len(cleaned) > 15:
        return False, cleaned, f"Invalid phone number length ({len(cleaned)} digits). Expected 10-15 digits."

    # Reject placeholder dummy numbers (all identical digits, e.g. 0000000000, 9999999999)
    # Check core number (last 10 digits)
    core = cleaned[-10:]
    if len(set(core)) <= 1:
        return False, cleaned, "Phone number is a dummy/placeholder with repeating identical digits."

    # Reject sequential numbers (e.g. 1234567890, 9876543210)
    sequences = ("1234567890", "0123456789", "9876543210", "0987654321")
    if core in sequences:
        return False, cleaned, "Phone number is a dummy sequential placeholder."

    return True, cleaned, None


def normalize_phone_number(raw_phone: str) -> str:
    """Clean phone numbers into standard digits format (e.g. 919949350699)."""
    valid, normalized, _ = validate_phone_number(raw_phone)
    if valid:
        return normalized
    # Fallback to digit stripping
    cleaned = re.sub(r"[^\d]", "", raw_phone.strip())
    if len(cleaned) == 10:
        return f"91{cleaned}"
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
        description="Result classification: queued, accepted, sent, delivered, read, auth_failed, connection_failed, timeout, failed, mock_sent",
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
        configured_url = base_url or getattr(self.settings, "openwa_base_url", "http://localhost:2785")
        self.base_url = (configured_url or "").rstrip("/")
        self._explicit_api_key = api_key
        self.session_id = session_id or getattr(self.settings, "whatsapp_session_id", "default")
        self.timeout = timeout

    def _get_api_key(self) -> Optional[str]:
        if self._explicit_api_key:
            return self._explicit_api_key
        key = getattr(self.settings, "openwa_api_key", None)
        if not key:
            return None
        getter = getattr(key, "get_secret_value", None)
        return getter() if callable(getter) else str(key)

    async def send_message(
        self,
        recipient: str,
        message: str,
    ) -> MessageSendResult:
        """Dispatch WhatsApp text message via OpenWA session endpoint."""
        # 1. Phone number validation
        is_valid, clean_num, err_msg = validate_phone_number(recipient)
        if not is_valid:
            logger.warning("WhatsApp recipient validation failed for '%s': %s", recipient, err_msg)
            return MessageSendResult(
                success=False,
                recipient=recipient,
                error=err_msg or "Recipient phone number is invalid.",
                status="failed",
            )

        if not isinstance(message, str) or not message.strip():
            return MessageSendResult(
                success=False,
                recipient=clean_num,
                error="Message content must be a non-empty string.",
                status="failed",
            )

        # 2. Gateway configuration verification
        if not self.base_url or self.base_url.lower() in ("none", "", "disabled"):
            logger.error("WhatsApp gateway is unconfigured: base_url is empty.")
            return MessageSendResult(
                success=False,
                recipient=clean_num,
                error="WhatsApp gateway is not configured. Please set OPENWA_BASE_URL.",
                status="failed",
            )

        # 3. Idempotency check: prevent duplicate dispatch within window
        idempotency_key = _make_idempotency_key(
            f"openwa:{self.base_url}:{self.session_id}", clean_num, message
        )
        now = time.time()
        cached_result = _get_cached_result(idempotency_key, now)
        if cached_result is not None:
            logger.info("Idempotency guard: duplicate WhatsApp message to %s suppressed.", clean_num)
            return cached_result

        chat_id = f"{clean_num}@c.us"
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        api_key = self._get_api_key()
        if api_key:
            headers["X-API-Key"] = api_key

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # 4. Discover ready session ID if possible
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

                # 5. Dispatch text message
                url = f"{self.base_url}/api/sessions/{effective_session}/messages/send-text"
                payload = {
                    "chatId": chat_id,
                    "text": message.strip(),
                }

                logger.info("Dispatching WhatsApp via %s to %s", url, chat_id)
                response = await client.post(url, json=payload, headers=headers)

                if response.status_code in (200, 201):
                    resp_json = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                    if not isinstance(resp_json, dict):
                        resp_json = {}
                    ext_id = resp_json.get("id") or resp_json.get("messageId") or f"msg_wa_{uuid.uuid4().hex[:12]}"
                    logger.info("WhatsApp message successfully accepted by gateway for %s (id: %s)", clean_num, ext_id)
                    res = MessageSendResult(
                        success=True,
                        message_id=ext_id,
                        recipient=clean_num,
                        status="accepted",  # HTTP 200/201 proves gateway acceptance, not confirmed recipient delivery
                    )
                    _idempotency_cache[idempotency_key] = (now, res)
                    return res
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
        is_valid, clean_num, err_msg = validate_phone_number(recipient)
        if not is_valid:
            return MessageSendResult(
                success=False,
                recipient=recipient,
                error=err_msg or "Recipient phone number cannot be empty or invalid.",
                status="failed",
            )

        if not isinstance(message, str) or not message.strip():
            return MessageSendResult(
                success=False,
                recipient=clean_num,
                error="Message content must be a non-empty string.",
                status="failed",
            )

        # Idempotency check for mock provider as well.
        idempotency_key = _make_idempotency_key("mock", clean_num, message)
        now = time.time()
        cached_result = _get_cached_result(idempotency_key, now)
        if cached_result is not None:
            return cached_result

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
            res = MessageSendResult(
                success=True,
                message_id=msg_id,
                recipient=clean_num,
                status="mock_sent",
            )
            _idempotency_cache[idempotency_key] = (now, res)
            return res

        return MessageSendResult(
            success=False,
            recipient=clean_num,
            error="Mock message dispatch configured to fail",
            status="failed",
        )
