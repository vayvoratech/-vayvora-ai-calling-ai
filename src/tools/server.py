"""Model Context Protocol (MCP) server implementation for the Unified AI Voice Agent.

Provides FastMCP server supporting JSON-RPC 2.0 tool invocation over stdio and HTTP.
Reuses and adapts tool implementations for Email, Calendar, and WhatsApp from
the legacy ai_call_mcp package while maintaining strict ActionVerifier compliance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional

from src.config import Settings, get_settings
from src.logging import get_logger
from src.tools.calendar_provider import CalendarProvider, FileCalendarProvider
from src.tools.email_provider import EmailProvider, SMTPEmailProvider
from src.tools.message_provider import MessageProvider, WhatsAppMessageProvider, normalize_phone_number

logger = get_logger("tools.server")


class Tool:
    """Metadata and execution wrapper for a registered MCP tool."""

    def __init__(self, name: str, description: str = "", fn: Optional[Callable] = None):
        self.name = name
        self.description = description
        self.fn = fn

    def __repr__(self) -> str:
        return f"<Tool name={self.name}>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
        }


class FastMCP:
    """FastMCP server conforming to Model Context Protocol specification.

    Provides tool registration, enumeration, JSON-RPC 2.0 handling, and stdio loop.
    """

    def __init__(self, name: str = "InfraBuilder-AI-MCP"):
        self.name = name
        self._tools: Dict[str, Tool] = {}

    def tool(self, name: Optional[str] = None, description: str = ""):
        """Decorator to register a function as an MCP tool."""
        def decorator(fn: Callable):
            tool_name = name or fn.__name__
            tool_desc = description or (fn.__doc__ or "").strip()
            self._tools[tool_name] = Tool(name=tool_name, description=tool_desc, fn=fn)
            return fn

        return decorator

    async def list_tools(self) -> List[Tool]:
        """Return list of registered tools."""
        return list(self._tools.values())

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Execute registered tool by name with arguments."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found.")
        args = arguments or {}
        fn = self._tools[name].fn
        if fn is None:
            raise RuntimeError(f"Tool '{name}' has no callable function.")
        if asyncio.iscoroutinefunction(fn):
            return await fn(**args)
        return fn(**args)

    async def _handle_jsonrpc_request(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process incoming JSON-RPC 2.0 request dictionary."""
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": self.name, "version": "2.0.0"},
                    "capabilities": {"tools": {}},
                },
            }

        elif method == "tools/list":
            tools = await self.list_tools()
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": [t.to_dict() for t in tools]},
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            try:
                raw_res = await self.call_tool(tool_name, tool_args)

                # Format response payload to satisfy both MCP standard content and ActionVerifier
                res_dict: Dict[str, Any] = {
                    "isError": False,
                }

                if isinstance(raw_res, dict):
                    res_dict.update(raw_res)
                    content_text = raw_res.get("message") or raw_res.get("summary") or json.dumps(raw_res)
                    res_dict["content"] = [{"type": "text", "text": str(content_text)}]
                else:
                    res_dict["content"] = [{"type": "text", "text": str(raw_res)}]
                    res_dict["message"] = str(raw_res)

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": res_dict,
                }
            except Exception as err:
                logger.error("Error executing tool '%s': %s", tool_name, err)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {err}"}],
                        "isError": True,
                        "error": str(err),
                    },
                }

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if req_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method '{method}' not found"},
            }
        return None

    async def handle_jsonrpc(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Public alias for handling JSON-RPC requests."""
        return await self._handle_jsonrpc_request(req)

    async def _run_stdio(self) -> None:
        """Run JSON-RPC stdio event loop."""
        loop = asyncio.get_running_loop()
        reader = None
        use_pipe = False
        if sys.platform != "win32":
            try:
                reader = asyncio.StreamReader()
                protocol = asyncio.StreamReaderProtocol(reader)
                await loop.connect_read_pipe(lambda: protocol, sys.stdin)
                use_pipe = True
            except (OSError, ValueError):
                use_pipe = False

        if not use_pipe and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
            sys.stderr.write(f"{self.name} server running on stdio (JSON-RPC 2.0). Ready for input...\n")
            sys.stderr.flush()

        while True:
            if use_pipe and reader is not None:
                line = await reader.readline()
                if not line:
                    break
                line_str = line.decode("utf-8").strip()
            else:
                line_raw = await loop.run_in_executor(None, sys.stdin.readline)
                if not line_raw:
                    break
                line_str = line_raw.strip()

            if not line_str:
                continue
            try:
                req = json.loads(line_str)
                resp = await self._handle_jsonrpc_request(req)
                if resp is not None:
                    out = json.dumps(resp) + "\n"
                    sys.stdout.write(out)
                    sys.stdout.flush()
            except Exception as err:
                err_resp = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": f"Parse error: {err}"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()

    def run(self, transport: str = "stdio") -> None:
        """Run the MCP server using the specified transport ('stdio')."""
        if transport == "stdio":
            if sys.platform == "win32":
                import msvcrt
                try:
                    msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
                    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
                except Exception:
                    pass
            try:
                asyncio.run(self._run_stdio())
            except (KeyboardInterrupt, SystemExit):
                pass
        else:
            raise ValueError(f"Unsupported transport: '{transport}'")


def create_mcp_server(
    settings: Optional[Settings] = None,
    email_provider: Optional[EmailProvider] = None,
    calendar_provider: Optional[CalendarProvider] = None,
    message_provider: Optional[MessageProvider] = None,
    postgres_repo: Optional[Any] = None,
) -> FastMCP:
    """Factory creating and configuring the FastMCP server with all registered tools."""
    cfg = settings or get_settings()
    mail = email_provider or SMTPEmailProvider(settings=cfg)
    cal = calendar_provider or FileCalendarProvider(settings=cfg)
    msg = message_provider or WhatsAppMessageProvider(settings=cfg)

    db_repo = postgres_repo
    if db_repo is None:
        try:
            from src.database.repository import PostgresRepository
            db_repo = PostgresRepository()
        except Exception as exc:
            logger.debug("PostgresRepository not initialized for MCP server: %s", exc)
            db_repo = None

    server = FastMCP(name="InfraBuilder-AI-MCP")

    # -------------------------------------------------------------------------
    # 1. Email Tools (send_email & mail_send, mail_read_recent)
    # -------------------------------------------------------------------------
    @server.tool(name="send_email", description="Dispatch an email containing property information or real estate materials.")
    async def send_email(
        recipient: str,
        subject: str = "Information from Infra Builder",
        body: str = "",
        html_body: Optional[str] = None,
        email: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        target = (recipient or email or "").strip()
        if not target or "@" not in target:
            raise ValueError("Action 'send_email' requires a valid recipient email address.")

        res = await mail.send_email(
            recipient=target,
            subject=subject or "Information from Infra Builder",
            body=body or "Hello, here are the requested details.",
            html_body=html_body,
        )
        if not res.success or not res.message_id:
            raise RuntimeError(res.error or "Failed to deliver email.")

        return {
            "message_id": res.message_id,
            "id": res.message_id,
            "external_reference": res.message_id,
            "status": "sent",
            "recipient": target,
            "message": f"Successfully queued email to {target}.",
        }

    @server.tool(name="mail_send", description="Alias for sending email via SMTP.")
    async def mail_send(to: str, subject: str, body: str, **kwargs: Any) -> Dict[str, Any]:
        return await send_email(recipient=to, subject=subject, body=body, **kwargs)

    @server.tool(name="mail_read_recent", description="Read recent emails from IMAP inbox.")
    async def mail_read_recent(limit: int = 5, **kwargs: Any) -> Dict[str, Any]:
        if not kwargs.get("caller_verified_sensitive", False):
            raise PermissionError("Access denied: Reading private email records requires verified caller identity.")
        if limit < 1:
            raise ValueError("Limit must be at least 1.")
        res = await mail.read_recent_emails(limit=limit)
        if not res.success:
            raise RuntimeError(res.error or "Failed to read recent emails.")
        return {
            "messages": res.messages,
            "summary": res.summary,
            "count": len(res.messages),
            "status": res.status,
            "message": res.summary,
        }

    # -------------------------------------------------------------------------
    # 2. Calendar Tools (find_available_slots, create_calendar_event, etc.)
    # -------------------------------------------------------------------------
    @server.tool(name="find_available_slots", description="Lookup calendar availability for consultation or demo.")
    async def find_available_slots(
        preferred_date: Optional[str] = None,
        meeting_type: Optional[str] = None,
        date: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        target_date = preferred_date or date
        res = cal.get_available_slots(preferred_date=target_date, meeting_type=meeting_type)
        return {
            "slots": res.get("slots", []),
            "count": res.get("count", len(res.get("slots", []))),
            "external_reference": f"slots_found_{res.get('count', len(res.get('slots', [])))}",
            "message": f"Available slots: {', '.join(res.get('slots', []))}",
        }

    @server.tool(name="calendar_get_events", description="Retrieve scheduled appointments.")
    async def calendar_get_events(**kwargs: Any) -> Dict[str, Any]:
        if not kwargs.get("caller_verified_sensitive", False):
            raise PermissionError("Access denied: Accessing appointment calendar records requires verified caller identity.")
        events = cal.get_events()
        if not events:
            summary = "You have no upcoming appointments on your calendar."
        else:
            earliest = events[0]
            summary = f"You currently have {len(events)} upcoming appointments. Your next one is on {earliest.get('date')} at {earliest.get('time')}."
        return {
            "events": events,
            "count": len(events),
            "summary": summary,
            "message": summary,
        }

    @server.tool(name="calendar_get_month_view", description="Generate text calendar for specific month.")
    async def calendar_get_month_view(year: int, month: int, **kwargs: Any) -> str:
        return cal.get_month_view(year=year, month=month)

    @server.tool(name="create_calendar_event", description="Book a confirmed property site visit slot on the calendar.")
    async def create_calendar_event(
        slot: Optional[str] = None,
        start_time: Optional[str] = None,
        time: Optional[str] = None,
        title: Optional[str] = "Property Site Visit",
        attendee_name: Optional[str] = None,
        attendee_email: Optional[str] = None,
        phone: Optional[str] = None,
        phone_number: Optional[str] = None,
        property_name: Optional[str] = None,
        confirmed: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        target_slot = slot or start_time or time
        if not target_slot:
            raise ValueError("Calendar Safety Violation: Cannot create calendar event without a specified time slot.")
        if confirmed is not True:
            raise ValueError("Calendar Safety Violation: Cannot create calendar event before caller explicitly confirms the slot.")

        raw_phone = phone or phone_number or kwargs.get("caller_phone")
        from src.database.repository import normalize_lead_phone
        clean_phone = normalize_lead_phone(raw_phone)

        if not clean_phone:
            raise ValueError("A valid caller phone number is required to persist the appointment.")
        if db_repo is None:
            raise RuntimeError(
                "Appointment booking is unavailable because the database repository is not initialized."
            )

        # 1. External store write: Add event to calendar provider
        res = cal.add_event(
            title=title or "Property Site Visit",
            slot=target_slot,
            attendee_name=attendee_name,
            attendee_email=attendee_email,
            phone_number=clean_phone,
            lead_id=kwargs.get("lead_id"),
        )
        ext_event_id = res.get("event_id")

        # 2. Database store write: Persist to PostgreSQL if db_repo is available and valid phone present
        appointment_id = None
        if db_repo is not None and clean_phone:
            try:
                event_date = res.get("date") or (target_slot.split()[0] if " " in target_slot else "Upcoming")
                event_time = res.get("time") or (" ".join(target_slot.split()[1:]) if " " in target_slot else target_slot)
                proj = property_name or kwargs.get("project_name") or title

                appt_record = await db_repo.create_appointment(
                    phone_number=clean_phone,
                    title=title or "Property Site Visit",
                    event_date=event_date,
                    event_time=event_time,
                    slot=target_slot,
                    project_name=proj,
                    external_event_id=ext_event_id,
                    lead_id=kwargs.get("lead_id"),
                    attendee_name=attendee_name,
                    attendee_email=attendee_email,
                )
                if appt_record:
                    appointment_id = appt_record.get("id")
                else:
                    # Bounded compensation: Roll back calendar event to prevent untracked partial creation
                    if ext_event_id:
                        try:
                            cal.delete_event(ext_event_id)
                        except Exception as comp_err:
                            logger.error("Failed to compensate calendar event %s: %s", ext_event_id, comp_err)
                    raise RuntimeError("Database failure persisting appointment: rolled back calendar event.")
            except Exception as exc:
                if ext_event_id:
                    try:
                        cal.delete_event(ext_event_id)
                    except Exception as comp_err:
                        logger.error("Failed to rollback calendar event %s on database exception: %s", ext_event_id, comp_err)
                logger.error("Appointment creation aborted due to database error: %s", exc)
                raise RuntimeError(f"Failed to persist appointment to database: {exc}") from exc

        return {
            "event_id": ext_event_id,
            "id": ext_event_id,
            "appointment_id": appointment_id,
            "external_reference": ext_event_id,
            "status": "confirmed",
            "slot": target_slot,
            "message": res.get("message", f"Successfully scheduled site visit for {target_slot}."),
        }

    @server.tool(name="cancel_calendar_event", description="Cancel and delete an existing property site visit appointment.")
    async def cancel_calendar_event(
        appointment_id: Optional[int] = None,
        event_id: Optional[str] = None,
        external_event_id: Optional[str] = None,
        phone: Optional[str] = None,
        phone_number: Optional[str] = None,
        confirmed: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # 1. Authorization check: sensitive action verification gate
        # Do not treat phone-number recognition or returning_caller=True as authorization!
        if kwargs.get("caller_verified_sensitive") is not True:
            raise PermissionError(
                "Access denied: Cancelling an appointment requires verified caller identity."
            )

        # 2. Confirmation check: explicit caller confirmation
        if confirmed is not True:
            raise ValueError("Cancellation Safety Violation: Explicit caller confirmation is required before deleting an appointment.")

        # 3. Caller phone verification
        raw_phone = phone or phone_number or kwargs.get("caller_phone")
        from src.database.repository import normalize_lead_phone

        clean_phone = normalize_lead_phone(raw_phone)
        if not clean_phone:
            raise ValueError("A verified caller phone number is required to cancel an appointment.")

        target_ext_id = external_event_id or event_id

        # 4. Resolve the exact appointment owned by this verified caller.
        if db_repo is None:
            raise RuntimeError(
                "Appointment cancellation is unavailable because the database repository is not initialized."
            )

        target_appt = None
        if appointment_id is not None:
            target_appt = await db_repo.get_appointment_by_id(
                appointment_id, phone_number=clean_phone
            )
        elif target_ext_id:
            caller_appts = await db_repo.get_appointments_by_phone(clean_phone)
            for appointment in caller_appts:
                if appointment.get("external_event_id") == target_ext_id:
                    target_appt = appointment
                    appointment_id = appointment.get("id")
                    break

        if not target_appt:
            return {
                "cancelled": False,
                "status": "not_found",
                "error": "No matching active appointment found for this caller.",
                "appointment_id": appointment_id,
            }

        # Use the stored event ID, not an untrusted caller-supplied ID.
        stored_event_id = target_appt.get("external_event_id")
        if not stored_event_id:
            raise RuntimeError(
                "The appointment has no linked calendar event ID; cancellation was not performed."
            )
        target_ext_id = stored_event_id

        # Step 4b: Delete from calendar provider first
        if target_ext_id:
            try:
                cal.delete_event(target_ext_id)
            except Exception as cal_err:
                logger.warning("Calendar provider deletion failed for %s: %s", target_ext_id, cal_err)
                # Calendar deletion failed or ambiguous -> retain PostgreSQL row and report failure!
                return {
                    "cancelled": False,
                    "status": "failed",
                    "error": "Calendar provider deletion could not be confirmed. Appointment retained in database.",
                    "appointment_id": appointment_id,
                }

        # Step 4c: Delete PostgreSQL row with ownership-constrained query
        deleted_row = None
        if db_repo is not None and appointment_id:
            deleted_row = await db_repo.delete_appointment(appointment_id, clean_phone)
            if not deleted_row:
                # Distinguish an already-completed/concurrent cancellation from a genuine database deletion failure
                existing = await db_repo.get_appointment_by_id(appointment_id, clean_phone)
                if not existing:
                    # The record is already absent/deleted; desired deletion outcome is satisfied.
                    logger.info(
                        "Appointment %s for caller phone ending %s was already deleted or absent.",
                        appointment_id,
                        clean_phone[-4:],
                    )
                    return {
                        "cancelled": True,
                        "status": "cancelled",
                        "appointment_id": appointment_id,
                        "event_id": target_ext_id,
                        "external_reference": str(appointment_id or target_ext_id or "cancelled"),
                        "message": "Appointment has already been successfully cancelled.",
                    }

                # PostgreSQL deletion failed while appointment still exists! Record recoverable inconsistency!
                await db_repo.record_reconciliation_inconsistency(
                    appointment_id=appointment_id,
                    external_event_id=target_ext_id,
                    phone_number=clean_phone,
                    action="delete_sync_failed",
                    details="Calendar event deleted from provider, but PostgreSQL row deletion failed while row is still present.",
                )
                return {
                    "cancelled": False,
                    "status": "inconsistent",
                    "error": "Failed to complete database appointment deletion after calendar cancellation. Inconsistency recorded.",
                    "appointment_id": appointment_id,
                }

        deleted_date = deleted_row.get("event_date") if deleted_row else (target_appt.get("event_date") if target_appt else "the scheduled date")
        deleted_time = deleted_row.get("event_time") if deleted_row else (target_appt.get("event_time") if target_appt else "the scheduled time")

        return {
            "cancelled": True,
            "status": "cancelled",
            "appointment_id": appointment_id,
            "event_id": target_ext_id,
            "external_reference": str(appointment_id or target_ext_id or "cancelled"),
            "message": f"Successfully cancelled appointment on {deleted_date} at {deleted_time}.",
        }

    @server.tool(name="calendar_cancel_event", description="Alias for cancel_calendar_event.")
    async def calendar_cancel_event(**kwargs: Any) -> Dict[str, Any]:
        return await cancel_calendar_event(**kwargs)

    @server.tool(name="calendar_add_event", description="Alias for scheduling calendar appointment with date and time strings.")
    async def calendar_add_event(
        title: str,
        date_str: str,
        time_str: str,
        description: str = "",
        confirmed: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        slot = f"{date_str} {time_str}".strip()
        return await create_calendar_event(
            slot=slot,
            title=title,
            description=description,
            confirmed=confirmed,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # 3. WhatsApp & Messaging Tools (send_message & whatsapp_send_message)
    # -------------------------------------------------------------------------
    @server.tool(name="send_message", description="Send an authorized WhatsApp message to the caller containing verified property details or confirmation.")
    async def send_message(
        recipient: Optional[str] = None,
        phone: Optional[str] = None,
        message: Optional[str] = None,
        whatsapp_consent: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        target_phone = recipient or phone
        if not target_phone:
            raise ValueError("Action 'send_message' requires a recipient phone number.")
        if not message:
            raise ValueError("Message content cannot be empty.")
        if whatsapp_consent is not True:
            raise PermissionError(
                "WhatsApp Safety Violation: Explicit caller consent is required "
                "before sending a WhatsApp message."
            )
        res = await msg.send_message(recipient=target_phone, message=message)
        if not res.success or not res.message_id:
            raise RuntimeError(res.error or "Failed to deliver message.")

        return {
            "message_id": res.message_id,
            "id": res.message_id,
            "external_reference": res.message_id,
            "status": res.status,
            "recipient": res.recipient,
            "message": f"Successfully dispatched WhatsApp message to {res.recipient}.",
        }

    @server.tool(name="whatsapp_send_message", description="Alias for sending WhatsApp message.")
    async def whatsapp_send_message(phone: str, message: str, **kwargs: Any) -> Dict[str, Any]:
        return await send_message(recipient=phone, message=message, **kwargs)

    # -------------------------------------------------------------------------
    # 4. Lead & CRM Tools (update_lead, create_hr_followup, update_business_status)
    # -------------------------------------------------------------------------
    @server.tool(name="update_lead", description="Update CRM lead details and qualification notes.")
    async def update_lead(
        status: Optional[str] = None,
        notes: Optional[str] = None,
        name: Optional[str] = None,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if kwargs.get("returning_caller") is True and not kwargs.get("caller_verified_sensitive", False):
            if kwargs.get("sensitive_action") is True or kwargs.get("overwrite_registered_profile") is True:
                raise PermissionError("Access denied: Modifying registered profile details requires verified caller identity.")

        lead_id = None

        if db_repo is None:
            raise RuntimeError(
                "CRM update failed: database repository is unavailable."
            )

        try:
            db_id = await db_repo.upsert_lead(
                name=name or kwargs.get("caller_name") or "Lead Prospect",
                phone_number=(
                    phone
                    or kwargs.get("caller_phone")
                    or kwargs.get("phone_number")
                ),
                email=email or kwargs.get("caller_email"),
                status=status or "updated",
                interested_in=(
                    kwargs.get("interested_in")
                    or kwargs.get("course")
                    or kwargs.get("service")
                ),
                metadata={"notes": notes, **kwargs} if (notes or kwargs) else None,
            )

            if db_id is None:
                raise RuntimeError("CRM repository did not confirm the lead update.")

            lead_id = f"lead_crm_{db_id}"

        except Exception as exc:
            logger.error("CRM lead update failed: %s", exc)
            raise RuntimeError(
                "CRM lead update failed; no successful update was confirmed."
            ) from exc

        return {
            "lead_id": lead_id,
            "id": lead_id,
            "external_reference": lead_id,
            "status": "updated",
            "message": "Lead record was updated successfully.",
        }

    @server.tool(name="create_hr_followup", description="Queue HR recruiter callback for applicant or student.")
    async def create_hr_followup(
        candidate_name: Optional[str] = None,
        name: Optional[str] = None,
        phone: Optional[str] = None,
        notes: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        c_name = candidate_name or name
        if not c_name:
            raise ValueError("Action 'create_hr_followup' requires candidate name or identification.")

        ticket_id = None

        if db_repo is None:
            raise RuntimeError(
                "Follow-up creation failed: database repository is unavailable."
            )

        try:
            session_id = (
                kwargs.get("session_id")
                or kwargs.get("call_id")
                or f"session_{os.getpid()}"
            )
            phone_num = phone or kwargs.get("phone_number") or kwargs.get("caller_phone")
            email_addr = kwargs.get("email") or kwargs.get("caller_email")
            reason = notes or kwargs.get("reason") or "HR callback requested"

            db_ticket = await db_repo.create_hr_followup(
                candidate_name=c_name,
                session_id=session_id,
                phone_number=phone_num,
                email=email_addr,
                reason=reason,
                status="queued",
            )

            if db_ticket is None:
                raise RuntimeError("Repository did not confirm follow-up creation.")

            ticket_id = f"ticket_hr_{db_ticket}"

        except Exception as exc:
            logger.error("HR follow-up creation failed: %s", exc)
            raise RuntimeError(
                "Follow-up creation failed; no successful ticket creation was confirmed."
            ) from exc

        return {
            "ticket_id": ticket_id,
            "id": ticket_id,
            "external_reference": ticket_id,
            "status": "queued",
            "message": "HR follow-up was queued successfully.",
        }

    @server.tool(
        name="update_business_status",
        description="Update domain business disposition status.",
    )
    async def update_business_status(status: str, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError(
            "Business status update is not connected to a persistent repository. "
            "No status update was performed."
        )

    return server


# Default singleton instance
mcp = create_mcp_server()


if __name__ == "__main__":
    mcp.run(transport="stdio")
