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
        raw_args = json.dumps(request.arguments, sort_keys=True)
        hash_digest = hashlib.sha256(raw_args.encode("utf-8")).hexdigest()[:16]
        return f"{request.call_id}:{request.tool_name}:{hash_digest}"

    async def execute_tool(self, request: ToolCallRequest) -> ToolExecutionResult:
        """Execute tool conforming to ToolProvider contract with verification."""
        idempotency_key = self._compute_idempotency_key(request)

        # Idempotency check: return cached result if already executed
        if idempotency_key in self._idempotency_cache:
            logger.info("Idempotent replay detected for key '%s'", idempotency_key)
            return self._idempotency_cache[idempotency_key].to_tool_execution_result()

        # Step 1: Wrap in ValidatedToolRequest and validate
        validated_req = ValidatedToolRequest(
            action_name=request.tool_name,
            arguments=request.arguments,
            session_id=request.call_id,
            idempotency_key=idempotency_key,
            domain=request.arguments.get("domain", "edusaas"),
            caller_name=request.arguments.get("caller_name"),
            caller_email=request.arguments.get("caller_email") or request.arguments.get("email"),
            caller_phone=request.arguments.get("caller_phone") or request.arguments.get("phone"),
        )

        try:
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
        if verified_result.succeeded:
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
            if self.email_provider:
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
                        action_name=action,
                        requested=True,
                        started=True,
                        succeeded=True,
                        external_reference=send_res.message_id,
                        data={"status": send_res.status, "message_id": send_res.message_id},
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
                data={"status": "dispatched", "message_id": f"msg_mock_{hash(request.correlation_id) % 100000}"},
            )

        elif action == "find_available_slots":
            return ToolResult(
                action_name=action,
                requested=True,
                started=True,
                succeeded=True,
                data={"slots": ["Tomorrow 10:00 AM", "Tomorrow 2:00 PM", "Next Monday 11:00 AM"]},
            )

        elif action == "create_calendar_event":
            return ToolResult(
                action_name=action,
                requested=True,
                started=True,
                succeeded=True,
                external_reference="evt_cal_987654",
                data={"event_id": "evt_cal_987654", "status": "confirmed", "slot": args.get("slot")},
            )

        elif action == "update_lead":
            return ToolResult(
                action_name=action,
                requested=True,
                started=True,
                succeeded=True,
                external_reference="lead_crm_12345",
                data={"lead_id": "lead_crm_12345", "updated": True},
            )

        elif action == "create_hr_followup":
            return ToolResult(
                action_name=action,
                requested=True,
                started=True,
                succeeded=True,
                external_reference="ticket_hr_554433",
                data={"ticket_id": "ticket_hr_554433", "status": "queued"},
            )

        elif action == "send_message":
            return ToolResult(
                action_name=action,
                requested=True,
                started=True,
                succeeded=True,
                external_reference="msg_sms_778899",
                data={"message_id": "msg_sms_778899", "status": "delivered"},
            )

        elif action == "update_business_status":
            return ToolResult(
                action_name=action,
                requested=True,
                started=True,
                succeeded=True,
                external_reference=f"status_{args.get('status')}",
                data={"status": args.get("status")},
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
                    data={"status": send_res.status, "message_id": send_res.message_id},
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

        async with httpx.AsyncClient(timeout=15.0) as client:
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

                body = resp.json()
                if "error" in body:
                    return ToolResult(
                        action_name=request.action_name,
                        requested=True,
                        failed=True,
                        error=str(body["error"]),
                    )

                res_data = body.get("result", {})
                return ToolResult(
                    action_name=request.action_name,
                    requested=True,
                    started=True,
                    succeeded=True,
                    data=res_data,
                    external_reference=res_data.get("external_reference") or res_data.get("id"),
                )
            except httpx.TimeoutException:
                return ToolResult(
                    action_name=request.action_name,
                    requested=True,
                    started=True,
                    failed=True,
                    error="MCP tool request timed out.",
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
        ]]
