"""MCP tool client implementations and mock provider for testing.

Provides MockToolProvider for deterministic unit tests and HttpMCPToolProvider
for external Model Context Protocol server transport with idempotency caching.
"""

import asyncio
import hashlib
import json
from typing import Any, Dict, List, Optional
import httpx

from src.config import Settings, get_settings
from src.core.interfaces import ToolProvider
from src.core.types import ToolCallRequest, ToolExecutionResult
from src.logging import get_logger
from src.tools.email_provider import EmailProvider
from src.tools.schemas import ToolResult, ValidatedToolRequest, VerificationStatus
from src.tools.validation import ActionValidator
from src.tools.verifier import ActionVerifier

logger = get_logger("tools.mcp_client")


class BaseMCPClient(ToolProvider):
    """Abstract base for MCP tool clients enforcing validation and verification."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        email_provider: Optional[EmailProvider] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.email_provider = email_provider
        self._idempotency_cache: Dict[str, ToolResult] = {}

    def _compute_idempotency_key(self, request: ToolCallRequest) -> str:
        """Derive deterministic idempotency key from request arguments and session."""
        raw_args = json.dumps(request.arguments, sort_keys=True, separators=(",", ":"), default=str)
        hash_digest = hashlib.sha256(raw_args.encode("utf-8")).hexdigest()[:16]
        return f"{request.call_id}:{request.tool_name}:{hash_digest}"

    async def execute_tool(self, request: ToolCallRequest) -> ToolExecutionResult:
        """Execute tool conforming to ToolProvider contract with verification."""
        idempotency_key = self._compute_idempotency_key(request)

        # Idempotency check: return cached result if already executed
        if idempotency_key in self._idempotency_cache:
            logger.info("Idempotent replay detected for key '%s'", idempotency_key)
            return self._idempotency_cache[idempotency_key].to_tool_execution_result()

        # Step 1: Build and validate the request. Invalid domains or malformed
        # arguments must fail closed instead of escaping the tool-execution boundary.
        try:
            validated_req = ValidatedToolRequest(
                action_name=request.tool_name,
                arguments=request.arguments,
                session_id=request.call_id,
                idempotency_key=idempotency_key,
                domain=request.arguments.get("domain", "infra_builder"),
                caller_name=request.arguments.get("caller_name"),
                caller_email=request.arguments.get("caller_email") or request.arguments.get("email"),
                caller_phone=request.arguments.get("caller_phone") or request.arguments.get("phone"),
                intent=request.arguments.get("intent"),
            )
            ActionValidator.validate_action(validated_req)
        except Exception as val_err:
            logger.warning("Action validation rejected for '%s': %s", request.tool_name, val_err)
            failed_res = ToolResult(
                action_name=request.tool_name,
                requested=True,
                started=False,
                succeeded=False,
                failed=True,
                verification_status=VerificationStatus.REJECTED,
                error=str(val_err),
            )
            return failed_res.to_tool_execution_result()

        # Step 2: Dispatch to transport
        raw_result = await self._dispatch(validated_req)

        # Step 3: Run Central Verification
        verified_result = ActionVerifier.verify(raw_result)

        # Cache only verified or completed mutations
        if verified_result.is_verified_success:
            # Keep the cache bounded for long-running voice sessions.
            if len(self._idempotency_cache) >= 2_000:
                self._idempotency_cache.clear()
            self._idempotency_cache[idempotency_key] = verified_result

        return verified_result.to_tool_execution_result()

    async def _dispatch(self, request: ValidatedToolRequest) -> ToolResult:
        """Dispatch request to specific transport (HTTP/Mock). Must be implemented by subclasses."""
        raise NotImplementedError


class MockToolProvider(BaseMCPClient):
    """Deterministic mock provider simulating external MCP responses, errors, and timeouts."""

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

        if self.force_unavailable:
            return ToolResult(
                action_name=request.action_name,
                requested=True,
                started=False,
                failed=True,
                error="MCP server unreachable (HTTP 503)",
            )

        if self.force_auth_failure:
            return ToolResult(
                action_name=request.action_name,
                requested=True,
                started=False,
                failed=True,
                error="MCP authentication failed (HTTP 401: Invalid token)",
            )

        if self.force_timeout:
            return ToolResult(
                action_name=request.action_name,
                requested=True,
                started=True,
                failed=True,
                error="MCP request timed out after 30.0s",
            )

        if self.force_malformed:
            return ToolResult(
                action_name=request.action_name,
                requested=True,
                started=True,
                succeeded=False,
                failed=True,
                error="Malformed response from MCP server: invalid JSON payload",
            )

        # Verification failure simulation: server claims success but omits external tracking reference
        if self.force_verification_failure:
            return ToolResult(
                action_name=request.action_name,
                requested=True,
                started=True,
                succeeded=True,  # Server said yes
                data={"status": "accepted"},  # Missing external reference!
            )

        # Default successful mock responses with verified external references
        action = request.action_name
        args = request.arguments

        if action == "send_email":
            recipient = args.get("recipient") or args.get("email") or request.caller_email
            subject = args.get("subject") or f"Information from {request.domain.value.title()}"
            body = args.get("body") or f"Hello {request.caller_name or 'there'},\n\nHere are the details you requested."
            html_body = args.get("html_body")
            if self.email_provider:
                send_res = await self.email_provider.send_email(
                    recipient=recipient,
                    subject=subject,
                    body=body,
                    html_body=html_body,
                )
                if send_res.success and send_res.message_id:
                    return ToolResult(
                        action_name=action,
                        requested=True,
                        started=True,
                        succeeded=True,
                        external_reference=send_res.message_id,
                        data={
                            "status": send_res.status,
                            "message_id": send_res.message_id,
                            "recipient": recipient,
                            "subject": subject,
                            "body": body,
                            "html_body": html_body,
                        },
                    )
                else:
                    return ToolResult(
                        action_name=action,
                        requested=True,
                        started=True,
                        succeeded=False,
                        failed=True,
                        verification_status=VerificationStatus.FAILED,
                        error=send_res.error or "Email dispatch failed",
                    )

            return ToolResult(
                action_name=action,
                requested=True,
                started=True,
                succeeded=True,
                external_reference=f"msg_mock_{hash(request.correlation_id) % 100000}",
                data={
                    "status": "dispatched",
                    "message_id": f"msg_mock_{hash(request.correlation_id) % 100000}",
                    "recipient": recipient,
                    "subject": subject,
                    "body": body,
                    "html_body": html_body,
                },
            )

        elif action == "find_available_slots":
            return ToolResult(
                action_name=action,
                requested=True,
                started=True,
                succeeded=True,
                data={
                    "slots": ["Tomorrow 10:00 AM", "Tomorrow 2:00 PM", "Next Monday 11:00 AM"],
                    "simulation": True,
                    "notice": "Mock slots for tests only; not real calendar availability.",
                },
            )

        elif action == "create_calendar_event":
            return ToolResult(
                action_name=action,
                requested=True,
                started=True,
                succeeded=True,
                external_reference="evt_cal_987654",
                data={
                    "event_id": "evt_cal_987654",
                    "status": "mock_confirmed",
                    "slot": args.get("slot"),
                    "simulation": True,
                    "notice": "Mock event only; no real calendar booking was created.",
                },
            )

        elif action == "update_lead":
            return ToolResult(
                action_name=action,
                requested=True,
                started=True,
                succeeded=True,
                external_reference="lead_crm_12345",
                data={
                    "lead_id": "lead_crm_12345",
                    "updated": True,
                    "simulation": True,
                    "notice": "Mock CRM update only; no production record was changed.",
                },
            )

        elif action == "create_hr_followup":
            return ToolResult(
                action_name=action,
                requested=True,
                started=True,
                succeeded=True,
                external_reference="ticket_hr_554433",
                data={
                    "ticket_id": "ticket_hr_554433",
                    "status": "mock_queued",
                    "simulation": True,
                    "notice": "Mock follow-up only; no real ticket was created.",
                },
            )

        elif action == "send_message":
            return ToolResult(
                action_name=action,
                requested=True,
                started=True,
                succeeded=True,
                external_reference="msg_sms_778899",
                data={
                    "message_id": "msg_sms_778899",
                    "status": "mock_sent",
                    "simulation": True,
                    "notice": "Mock message only; no WhatsApp/SMS was sent or delivered.",
                },
            )

        elif action == "update_business_status":
            return ToolResult(
                action_name=action,
                requested=True,
                started=True,
                succeeded=True,
                external_reference=f"status_{args.get('status')}",
                data={
                    "status": args.get("status"),
                    "simulation": True,
                    "notice": "Mock status update only; no production system was changed.",
                },
            )

        elif action == "mail_read_recent":
            if self.email_provider:
                read_res = await self.email_provider.read_recent_emails(limit=args.get("limit", 5))
                if read_res.success:
                    return ToolResult(
                        action_name=action,
                        requested=True,
                        started=True,
                        succeeded=True,
                        external_reference=f"emails_read_{len(read_res.messages)}",
                        data={"messages": read_res.messages, "summary": read_res.summary, "status": read_res.status},
                    )
                else:
                    return ToolResult(
                        action_name=action,
                        requested=True,
                        started=True,
                        succeeded=False,
                        failed=True,
                        verification_status=VerificationStatus.FAILED,
                        error=read_res.error or "IMAP read failed",
                    )
            return ToolResult(
                action_name=action,
                requested=True,
                started=True,
                succeeded=True,
                external_reference="emails_read_2",
                data={
                    "messages": [
                        {"from": "inquiries@infrabuilder.com", "subject": "Real Estate Brochure Inquiry"},
                        {"from": "sales@infrabuilder.com", "subject": "Site Visit Booking Followup"},
                    ],
                    "summary": "From: inquiries@infrabuilder.com | Subject: Real Estate Brochure Inquiry\nFrom: sales@infrabuilder.com | Subject: Site Visit Booking Followup",
                },
            )

        return ToolResult(
            action_name=action,
            requested=True,
            failed=True,
            error=f"No mock handler for action '{action}'",
        )

    def list_tools(self) -> List[Dict[str, Any]]:
        return [{"name": a} for a in [
            "send_email",
            "find_available_slots",
            "create_calendar_event",
            "update_lead",
            "create_hr_followup",
            "send_message",
            "update_business_status",
            "mail_read_recent",
        ]]


class HttpMCPToolProvider(BaseMCPClient):
    """External MCP server client executing actions via HTTP JSON-RPC."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        email_provider: Optional[EmailProvider] = None,
    ) -> None:
        super().__init__(settings=settings, email_provider=email_provider)
        self.endpoint = self.settings.mcp_server_url
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self, timeout: float) -> httpx.AsyncClient:
        """Get or initialize persistent AsyncClient with HTTP connection pooling."""
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=30.0)
            self._client = httpx.AsyncClient(timeout=timeout, limits=limits)
        return self._client

    async def aclose(self) -> None:
        """Close persistent HTTP client session."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _dispatch(self, request: ValidatedToolRequest) -> ToolResult:
        if request.action_name == "send_email" and self.email_provider:
            args = request.arguments
            recipient = args.get("recipient") or args.get("email") or request.caller_email
            subject = args.get("subject") or f"Information from {request.domain.value.title()}"
            body = args.get("body") or f"Hello {request.caller_name or 'there'},\n\nHere are the details you requested."
            html_body = args.get("html_body")
            send_res = await self.email_provider.send_email(
                recipient=recipient,
                subject=subject,
                body=body,
                html_body=html_body,
            )
            if send_res.success and send_res.message_id:
                return ToolResult(
                    action_name=request.action_name,
                    requested=True,
                    started=True,
                    succeeded=True,
                    external_reference=send_res.message_id,
                    data={
                        "status": send_res.status,
                        "message_id": send_res.message_id,
                        "recipient": recipient,
                        "subject": subject,
                        "body": body,
                        "html_body": html_body,
                    },
                )
            else:
                return ToolResult(
                    action_name=request.action_name,
                    requested=True,
                    started=True,
                    succeeded=False,
                    failed=True,
                    verification_status=VerificationStatus.FAILED,
                    error=send_res.error or "Email dispatch failed",
                )

        if not self.endpoint:
            return ToolResult(
                action_name=request.action_name,
                requested=True,
                failed=True,
                error="MCP endpoint URL is not configured.",
            )

        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": request.action_name,
                "arguments": request.arguments,
            },
            "id": request.correlation_id,
        }

        timeout = getattr(self.settings, "mcp_timeout_seconds", 15.0)
        client = await self._get_client(timeout)
        try:
            resp = await client.post(self.endpoint, json=payload)
            if resp.status_code == 401 or resp.status_code == 403:
                return ToolResult(
                    action_name=request.action_name,
                    requested=True,
                    failed=True,
                    error=f"MCP authentication error HTTP {resp.status_code}",
                )
            if resp.status_code != 200:
                return ToolResult(
                    action_name=request.action_name,
                    requested=True,
                    failed=True,
                    error=f"MCP server error HTTP {resp.status_code}: {resp.text}",
                )

            try:
                body = resp.json()
            except ValueError:
                return ToolResult(
                    action_name=request.action_name,
                    requested=True,
                    started=True,
                    failed=True,
                    error="MCP server returned invalid JSON.",
                )

            if not isinstance(body, dict):
                return ToolResult(
                    action_name=request.action_name,
                    requested=True,
                    started=True,
                    failed=True,
                    error="MCP server returned an invalid JSON-RPC response.",
                )

            if body.get("error"):
                return ToolResult(
                    action_name=request.action_name,
                    requested=True,
                    started=True,
                    failed=True,
                    error=str(body["error"]),
                )

            res_data = body.get("result")
            if not isinstance(res_data, dict):
                return ToolResult(
                    action_name=request.action_name,
                    requested=True,
                    started=True,
                    failed=True,
                    error="MCP response is missing a valid result object.",
                )

            if res_data.get("isError"):
                return ToolResult(
                    action_name=request.action_name,
                    requested=True,
                    started=True,
                    succeeded=False,
                    failed=True,
                    error=str(res_data.get("error") or "Tool execution reported error"),
                    data=res_data,
                )

            # Prefer structuredContent when supplied by an MCP server. Otherwise
            # preserve the result payload for the verifier rather than guessing success.
            structured = res_data.get("structuredContent")
            result_data = structured if isinstance(structured, dict) else res_data
            ext_ref = (
                result_data.get("external_reference")
                or result_data.get("id")
                or result_data.get("message_id")
                or result_data.get("event_id")
                or result_data.get("lead_id")
                or result_data.get("ticket_id")
            )
            return ToolResult(
                action_name=request.action_name,
                requested=True,
                started=True,
                succeeded=True,
                data=result_data,
                external_reference=str(ext_ref) if ext_ref is not None else None,
            )
        except httpx.TimeoutException:
            return ToolResult(
                action_name=request.action_name,
                requested=True,
                started=True,
                failed=True,
                error=f"MCP tool request timed out after {timeout}s.",
            )
        except Exception as exc:
            return ToolResult(
                action_name=request.action_name,
                requested=True,
                failed=True,
                error=f"Network error contacting MCP server: {exc}",
            )

    def list_tools(self) -> List[Dict[str, Any]]:
        return [{"name": a} for a in [
            "send_email",
            "find_available_slots",
            "create_calendar_event",
            "update_lead",
            "create_hr_followup",
            "send_message",
            "update_business_status",
            "mail_read_recent",
        ]]
