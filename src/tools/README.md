# MCP Tool Layer & Verified Action Protocol

This module implements the external action and mutation layer for the AI Voice Agent across both **EduSaaS** and **Vayvora** business domains using the **Model Context Protocol (MCP)** specification.

---

## 1. Architectural Boundaries

```text
ConversationEngine
        ↓
ProposedAction
        ↓
ActionValidator (Pre-execution guards & schemas)
        ↓
ToolProvider / MCP Client (HttpMCPToolProvider / MockToolProvider)
        ↓
External MCP Server
        ↓
Raw Tool Result
        ↓
ActionVerifier (Central post-execution verification protocol)
        ↓
ConversationState (Records verified external references / outcomes)
        ↓
Conversational Spoken Confirmation / Non-terminating Error Recovery
```

### Core Invariants & Security Principles
1. **No Direct Database/Service Access**: The LLM engine is strictly prohibited from direct database queries, CRM writes, or email dispatches. All side-effects must route through registered tool schemas.
2. **Untrusted Tool Outputs**: Data returned from external MCP servers is treated as untrusted input. Outputs are validated, sanitized, and strictly checked before incorporation into state or speech synthesis.
3. **No Fabricated Success**: An HTTP 200 response or server acknowledgment is **not** treated as business success unless verified by an authentic external tracking identifier (e.g. `message_id`, `event_id`, `lead_id`, `ticket_id`).
4. **Idempotency Guarantees**: A deterministic SHA256 idempotency cache keyed on `(session_id, action_name, arguments_hash)` prevents duplicate external mutations (e.g. double bookings or duplicate email dispatches).
5. **Non-Terminating Errors**: External tool failures, timeouts, or network outages must never crash the engine or hang up on the caller. The engine gracefully informs the caller, records the failure, and keeps the call active.

---

## 2. Available Actions & Schemas

| Action Name | Description | Required Arguments | Verified Reference Format |
|---|---|---|---|
| `send_email` | Dispatches follow-up emails, curriculum PDFs, or company brochures | `recipient` (valid email) | `message_id` (e.g. `msg_...`) |
| `find_available_slots` | Queries calendar for available consultation/demo slots | Optional `date`, `domain` | `slots_found_<count>` |
| `create_calendar_event` | Books confirmed calendar appointments | `slot` / `start_time`, `confirmed=True` | `event_id` (e.g. `evt_...`) |
| `update_lead` | Updates caller qualification status and notes in CRM | At least 1 field to update | `lead_id` (e.g. `lead_...`) |
| `create_hr_followup` | Queues HR recruiter follow-ups for candidates | `candidate_name` | `ticket_id` (e.g. `ticket_...`) |
| `send_message` | Sends transactional SMS confirmation | `recipient` / `phone` | `message_id` (e.g. `msg_...`) |
| `update_business_status` | Updates conversational state business disposition | `status` (e.g. `qualified_lead`) | `status_<status>` |

---

## 3. Verification Protocol

The `ActionVerifier` independently validates all tool execution responses against domain verification rules:

```python
from src.tools.verifier import ActionVerifier
from src.tools.schemas import ToolResult, VerificationStatus

# A response from an MCP server
raw_result = await client.post(...)

# Central verification check
verified_result = ActionVerifier.verify(raw_result)

if verified_result.is_verified_success:
    # State updated with external reference (e.g., event_id or message_id)
    state.complete_action(tool_name, verified_result)
else:
    # Marked as failed/unverified; verbal confirmation NEVER fabricates success
    logger.warning("Verification failed: %s", verified_result.error)
```

### Verification Rules
* **Email**: Fails verification if `message_id` is missing or empty.
* **Calendar Booking**: Fails verification if `event_id` is missing or empty. Booking without `confirmed=True` is blocked by pre-execution validation.
* **Available Slots**: Fails verification if `slots` is not a list.
* **Lead Update**: Fails verification if CRM update is not acknowledged or has errors.
* **HR Ticket**: Fails verification if `ticket_id` is not returned.

---

## 4. Calendar Safety Invariants

Booking an appointment requires explicit verification of two preconditions:
1. **Confirmed Slot**: A specific time slot must be selected and agreed upon by the caller. If the LLM proposes booking without a slot, the `ConversationEngine` intercepts the request and asks the caller:
   > *"I can certainly schedule that consultation. Which date or time slot would work best for you?"*
2. **Explicit Confirmation**: If an action is submitted with `confirmed=False`, `ActionValidator` raises `InvalidToolArgumentsError` before calling any external API.

---

## 5. Conversational Email Guard

If the caller requests email materials but their email address is not stored in `CallerProfile` and was not provided in the prompt:
1. `ConversationEngine` intercepts `send_email`.
2. Execution is deferred; no tool request is sent to the network.
3. The engine prompts the caller:
   > *"I would be happy to email those details to you. Could you please share your email address?"*
4. `pending_question` is set on `ConversationState`.

---

## 6. Failure Modes Handled

- **MCP Unavailable (HTTP 503)**: Trapped cleanly; conversation remains active; caller receives a polite message:
  > *"I attempted to complete that request, but encountered a system issue. I have noted this down so our team can follow up with you directly."*
- **MCP Timeout**: Requests exceeding the timeout deadline fail gracefully without blocking the voice loop.
- **Authentication Failure (HTTP 401/403)**: Logged as a security error; caller receives standard non-technical recovery response.
- **Malformed Response**: Invalid or non-JSON payloads are safely converted to `ToolResult(failed=True, error=...)`.
- **Pre-execution Validation Failure**: Unsupported action names and malformed parameters are caught immediately with `VerificationStatus.REJECTED`.

---

## 7. How to Run Tests

### Run all Phase 5 Tool tests:
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_tools.py -v
```

### Run entire test suite (Phases 1 through 5):
```bash
.\.venv\Scripts\python.exe -m pytest -v
```
