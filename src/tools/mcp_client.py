"""MCP tool client implementations and mock provider for testing.

Provides MockToolProvider for deterministic unit tests and HttpMCPToolProvider
for external Model Context Protocol server transport with idempotency caching.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional

import httpx
from src.core.types import DomainType, ToolCallRequest, ToolExecutionResult
from src.config import Settings, get_settings
from src.core.interfaces import ToolProvider
from src.core.types import ToolCallRequest, ToolExecutionResult
from src.logging import get_logger
from src.tools.email_provider import EmailProvider
from src.tools.schemas import ToolResult, ValidatedToolRequest, VerificationStatus
from src.tools.validation import ActionValidator
from src.tools.verifier import ActionVerifier

logger = get_logger("tools.mcp_client")


SUPPORTED_TOOL_NAMES = [
    "send_email",
    "find_available_slots",
    "create_calendar_event",
    "cancel_calendar_event",
    "calendar_cancel_event",
    "update_lead",
    "create_hr_followup",
    "send_message",
    "update_business_status",
    "mail_read_recent",
]


class BaseMCPClient(ToolProvider):
    """Base MCP client enforcing validation, verification, and idempotency."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        email_provider: Optional[EmailProvider] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.email_provider = email_provider
        self._idempotency_cache: Dict[str, ToolResult] = {}

    def _trusted_domain(self) -> DomainType:
        """Resolve the active business domain from trusted configuration."""
        configured_domain = getattr(self.settings, "active_domain", None)

        if configured_domain is None:
            configured_domain = getattr(self.settings, "domain", None)

        if configured_domain is None:
            return DomainType.INFRA_BUILDER

        value = (
            configured_domain.value
            if isinstance(configured_domain, DomainType)
            else str(configured_domain).strip().lower()
        )

        try:
            return DomainType(value)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported configured domain: {value!r}"
            ) from exc
    def _compute_idempotency_key(self, request: ToolCallRequest) -> str:
        """Derive a deterministic key from call identity and normalized arguments."""
        raw_args = json.dumps(
            request.arguments,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        digest = hashlib.sha256(raw_args.encode("utf-8")).hexdigest()[:16]

        return f"{request.call_id}:{request.tool_name}:{digest}"


    async def execute_tool(self, request: ToolCallRequest) -> ToolExecutionResult:
        """Validate, execute, verify, and cache verified tool outcomes."""
        idempotency_key = self._compute_idempotency_key(request)

        cached = self._idempotency_cache.get(idempotency_key)
        if cached is not None:
            logger.info(
                "Idempotent replay detected for tool '%s'.",
                request.tool_name,
            )
            return cached.to_tool_execution_result()

        arguments = dict(request.arguments or {})

        # Resolve the trusted domain from application configuration.
        try:
            trusted_domain = self._trusted_domain()
        except Exception as exc:
            logger.error("Could not resolve the trusted tool domain: %s", exc)
            rejected = ToolResult(
                action_name=request.tool_name,
                requested=True,
                started=False,
                succeeded=False,
                failed=True,
                verification_status=VerificationStatus.REJECTED,
                error="The configured business domain is invalid.",
            )
            return rejected.to_tool_execution_result()

        # Reject attempts to override the trusted domain.
        supplied_domain = arguments.get("domain")
        if supplied_domain is not None:
            supplied_value = (
                supplied_domain.value
                if isinstance(supplied_domain, DomainType)
                else str(supplied_domain).strip().lower()
            )

            if supplied_value != trusted_domain.value:
                logger.warning(
                    "Rejected domain override for tool '%s'.",
                    request.tool_name,
                )
                rejected = ToolResult(
                    action_name=request.tool_name,
                    requested=True,
                    started=False,
                    succeeded=False,
                    failed=True,
                    verification_status=VerificationStatus.REJECTED,
                    error=(
                        "Tool domain does not match the trusted active domain."
                    ),
                )
                return rejected.to_tool_execution_result()

        # Domain belongs to the validated request context.
        # Do not send it as a model-controlled MCP action argument.
        arguments.pop("domain", None)

        validated_req = ValidatedToolRequest(
            action_name=request.tool_name,
            arguments=arguments,
            session_id=request.call_id,
            correlation_id=request.call_id,
            idempotency_key=idempotency_key,
            domain=trusted_domain,
            caller_name=arguments.get("caller_name"),
            caller_email=(
                arguments.get("caller_email")
                or arguments.get("email")
            ),
            caller_phone=(
                arguments.get("caller_phone")
                or arguments.get("phone")
                or arguments.get("phone_number")
            ),
            intent=arguments.get("intent"),
        )

        try:
            ActionValidator.validate_action(validated_req)
        except Exception as exc:
            logger.warning(
                "Action validation rejected '%s': %s",
                request.tool_name,
                exc,
            )
            rejected = ToolResult(
                action_name=request.tool_name,
                requested=True,
                started=False,
                succeeded=False,
                failed=True,
                verification_status=VerificationStatus.REJECTED,
                error=str(exc),
            )
            return rejected.to_tool_execution_result()

        try:
            raw_result = await self._dispatch(validated_req)
        except Exception as exc:
            logger.exception(
                "Tool dispatch raised an exception for '%s'.",
                request.tool_name,
            )
            failed = ToolResult(
                action_name=request.tool_name,
                requested=True,
                started=True,
                succeeded=False,
                failed=True,
                verification_status=VerificationStatus.FAILED,
                error=f"Tool dispatch failed: {type(exc).__name__}",
            )
            return failed.to_tool_execution_result()

        verified_result = ActionVerifier.verify(raw_result)

        # Cache only verified successes.
        if verified_result.is_verified_success:
            if len(self._idempotency_cache) >= 2_000:
                self._idempotency_cache.clear()

            self._idempotency_cache[idempotency_key] = verified_result

        return verified_result.to_tool_execution_result()

    async def _dispatch(self, request: ValidatedToolRequest) -> ToolResult:
        """Dispatch to the configured transport."""
        raise NotImplementedError

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return the common supported action names."""
        return [{"name": name} for name in SUPPORTED_TOOL_NAMES]


class MockToolProvider(BaseMCPClient):
    """Deterministic mock provider for tests.

    Every simulated outcome is explicitly marked as simulation data so it
    cannot be reported as a verified real-world action.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        email_provider: Optional[EmailProvider] = None,
        force_timeout: bool = False,
        force_auth_failure: bool = False,
        force_unavailable: bool = False,
        force_malformed: bool = False,
        force_verification_failure: bool = False,
    ) -> None:
        super().__init__(settings=settings, email_provider=email_provider)

        self.force_timeout = force_timeout
        self.force_auth_failure = force_auth_failure
        self.force_unavailable = force_unavailable
        self.force_malformed = force_malformed
        self.force_verification_failure = force_verification_failure

        self.dispatched_requests: List[ValidatedToolRequest] = []

    async def _dispatch(self, request: ValidatedToolRequest) -> ToolResult:
        self.dispatched_requests.append(request)

        action = request.action_name
        args = request.arguments

        if self.force_unavailable:
            return self._failure(action, "MCP server unavailable (HTTP 503).")

        if self.force_auth_failure:
            return self._failure(action, "MCP authentication failed (HTTP 401).")

        if self.force_timeout:
            return self._failure(action, "MCP request timed out.")

        if self.force_malformed:
            return self._failure(action, "MCP server returned malformed data.")

        if self.force_verification_failure:
            return ToolResult(
                action_name=action,
                requested=True,
                started=True,
                succeeded=True,
                data={"status": "accepted", "simulation": True},
            )

        if action == "send_email":
            recipient = (
                args.get("recipient")
                or args.get("email")
                or request.caller_email
            )
            subject = args.get("subject") or (
                f"Information from {request.domain.value.title()}"
            )
            body = args.get("body") or (
                f"Hello {request.caller_name or 'there'},\n\n"
                "Here are the details you requested."
            )

            if self.email_provider:
                try:
                    send_result = await self.email_provider.send_email(
                        recipient=recipient,
                        subject=subject,
                        body=body,
                        html_body=args.get("html_body"),
                    )
                except Exception as exc:
                    return self._failure(action, f"Email dispatch failed: {exc}")

                if not send_result.success or not send_result.message_id:
                    return self._failure(
                        action,
                        send_result.error or "Email dispatch failed.",
                    )

                return ToolResult(
                    action_name=action,
                    requested=True,
                    started=True,
                    succeeded=True,
                    external_reference=send_result.message_id,
                    data={
                        "status": send_result.status,
                        "message_id": send_result.message_id,
                        "recipient": recipient,
                        "subject": subject,
                    },
                )

            return self._mock_result(
                action,
                "Mock email only; no real email was sent.",
                message_id="msg_mock_test",
                recipient=recipient,
                subject=subject,
            )

        if action == "find_available_slots":
            return self._mock_result(
                action,
                "Mock slots only; real calendar availability was not checked.",
                slots=[
                    "Tomorrow 10:00 AM",
                    "Tomorrow 2:00 PM",
                    "Next Monday 11:00 AM",
                ],
            )

        if action == "create_calendar_event":
            return self._mock_result(
                action,
                "Mock event only; no real calendar booking was created.",
                event_id="evt_mock_test",
                status="mock_confirmed",
                slot=args.get("slot"),
            )

        if action in ("cancel_calendar_event", "calendar_cancel_event"):
            return self._mock_result(
                action,
                "Mock cancellation only; no real appointment was cancelled.",
                appointment_id=args.get("appointment_id"),
                event_id=args.get("external_event_id"),
                cancelled=True,
                status="cancelled",
            )

        if action == "update_lead":
            return self._mock_result(
                action,
                "Mock CRM update only; no production record was changed.",
                lead_id="lead_mock_test",
                updated=True,
            )

        if action == "create_hr_followup":
            return self._mock_result(
                action,
                "Mock follow-up only; no real ticket was created.",
                ticket_id="ticket_mock_test",
                status="mock_queued",
            )

        if action == "send_message":
            return self._mock_result(
                action,
                "Mock message only; no real WhatsApp message was sent.",
                message_id="msg_mock_test",
                status="mock_sent",
            )

        if action == "update_business_status":
            return self._mock_result(
                action,
                "Mock status update only; no production system was changed.",
                status=args.get("status"),
                updated=True,
            )

        if action == "mail_read_recent":
            if self.email_provider:
                try:
                    read_result = await self.email_provider.read_recent_emails(
                        limit=args.get("limit", 5)
                    )
                except Exception as exc:
                    return self._failure(action, f"Inbox read failed: {exc}")

                if not read_result.success:
                    return self._failure(
                        action,
                        read_result.error or "Inbox read failed.",
                    )

                return ToolResult(
                    action_name=action,
                    requested=True,
                    started=True,
                    succeeded=True,
                    external_reference=f"emails_read_{len(read_result.messages)}",
                    data={
                        "messages": read_result.messages,
                        "summary": read_result.summary,
                        "status": read_result.status,
                    },
                )

            return self._mock_result(
                action,
                "Mock inbox only; no live inbox was read.",
                messages=[],
            )

        return self._failure(
            action,
            f"No mock handler exists for action '{action}'.",
        )

    @staticmethod
    def _failure(action: str, message: str) -> ToolResult:
        return ToolResult(
            action_name=action,
            requested=True,
            started=False,
            succeeded=False,
            failed=True,
            verification_status=VerificationStatus.FAILED,
            error=message,
        )

    @staticmethod
    def _mock_result(
        action: str,
        notice: str,
        **data: Any,
    ) -> ToolResult:
        return ToolResult(
            action_name=action,
            requested=True,
            started=True,
            succeeded=True,
            data={
                **data,
                "simulation": True,
                "notice": notice,
            },
        )


class HttpMCPToolProvider(BaseMCPClient):
    """HTTP JSON-RPC MCP client with centralized outcome verification."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        email_provider: Optional[EmailProvider] = None,
    ) -> None:
        super().__init__(settings=settings, email_provider=email_provider)

        self.endpoint = self.settings.mcp_server_url
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self, timeout: float) -> httpx.AsyncClient:
        """Create or reuse an HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
                keepalive_expiry=30.0,
            )
            self._client = httpx.AsyncClient(
                timeout=timeout,
                limits=limits,
            )

        return self._client

    async def aclose(self) -> None:
        """Close the persistent HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _dispatch(self, request: ValidatedToolRequest) -> ToolResult:
        action = request.action_name

        if action == "send_email" and self.email_provider:
            args = request.arguments
            recipient = (
                args.get("recipient")
                or args.get("email")
                or request.caller_email
            )
            subject = args.get("subject") or (
                f"Information from {request.domain.value.title()}"
            )
            body = args.get("body") or (
                f"Hello {request.caller_name or 'there'},\n\n"
                "Here are the details you requested."
            )

            try:
                send_result = await self.email_provider.send_email(
                    recipient=recipient,
                    subject=subject,
                    body=body,
                    html_body=args.get("html_body"),
                )
            except Exception as exc:
                return self._failure(action, f"Email dispatch failed: {exc}")

            if not send_result.success or not send_result.message_id:
                return self._failure(
                    action,
                    send_result.error or "Email dispatch failed.",
                )

            return ToolResult(
                action_name=action,
                requested=True,
                started=True,
                succeeded=True,
                external_reference=send_result.message_id,
                data={
                    "status": send_result.status,
                    "message_id": send_result.message_id,
                    "recipient": recipient,
                    "subject": subject,
                },
            )

        if not self.endpoint:
            return self._failure(
                action,
                "MCP endpoint URL is not configured.",
            )

        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": action,
                "arguments": request.arguments,
            },
            "id": request.correlation_id,
        }

        timeout = getattr(self.settings, "mcp_timeout_seconds", 15.0)

        try:
            client = await self._get_client(timeout)
            response = await client.post(self.endpoint, json=payload)

            if response.status_code in (401, 403):
                return self._failure(
                    action,
                    f"MCP authentication error HTTP {response.status_code}.",
                )

            if response.status_code != 200:
                return self._failure(
                    action,
                    f"MCP server error HTTP {response.status_code}.",
                )

            try:
                body = response.json()
            except ValueError:
                return self._failure(
                    action,
                    "MCP server returned invalid JSON.",
                )

            if not isinstance(body, dict):
                return self._failure(
                    action,
                    "MCP server returned an invalid JSON-RPC response.",
                )

            if body.get("error"):
                return self._failure(
                    action,
                    f"MCP JSON-RPC error: {body['error']}",
                )

            rpc_result = body.get("result")
            if not isinstance(rpc_result, dict):
                return self._failure(
                    action,
                    "MCP response is missing a valid result object.",
                )

            if rpc_result.get("isError"):
                return self._failure(
                    action,
                    str(
                        rpc_result.get("error")
                        or "MCP tool execution reported an error."
                    ),
                )

            structured = rpc_result.get("structuredContent")
            result_data = (
                structured
                if isinstance(structured, dict)
                else rpc_result
            )

            # MCP transport acceptance is not business-action verification.
            # ActionVerifier decides whether this evidence proves success.
            external_reference = (
                result_data.get("external_reference")
                or result_data.get("message_id")
                or result_data.get("event_id")
                or result_data.get("booking_ref")
                or result_data.get("lead_id")
                or result_data.get("ticket_id")
                or result_data.get("id")
            )

            return ToolResult(
                action_name=action,
                requested=True,
                started=True,
                succeeded=True,
                data=result_data,
                external_reference=(
                    str(external_reference)
                    if external_reference is not None
                    else None
                ),
            )

        except httpx.TimeoutException:
            return self._failure(
                action,
                f"MCP tool request timed out after {timeout} seconds.",
            )
        except httpx.HTTPError as exc:
            logger.warning("HTTP transport error for '%s': %s", action, exc)
            return self._failure(
                action,
                "Network error contacting the MCP server.",
            )
        except Exception as exc:
            logger.exception("Unexpected MCP error for '%s'.", action)
            return self._failure(
                action,
                f"Unexpected MCP client error: {exc}",
            )

    @staticmethod
    def _failure(action: str, message: str) -> ToolResult:
        return ToolResult(
            action_name=action,
            requested=True,
            started=False,
            succeeded=False,
            failed=True,
            verification_status=VerificationStatus.FAILED,
            error=message,
        )