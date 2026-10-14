# Testing Workbench

An interactive testing and evaluation workbench for pair-testing the **Infra Builder Real Estate AI Calling Agent** across inbound and outbound calls before connecting real audio, VAD, STT/TTS, or telephony pipelines.

---

## 1. Quick Start: How to Run the Workbench

Ensure your virtual environment is active, then launch the Streamlit app:

```bash
.\.venv\Scripts\python.exe -m streamlit run src/ui/app.py
```

By default, the application opens in your browser at `http://localhost:8501`.

### Runtime Modes
The application displays an explicit status badge in the sidebar:
* **MOCK Mode (🟡 `MOCK`)**: Uses `InteractiveMockLLMProvider`, in-memory vector store seeded with documents from `knowledge_base/infra_builder/`, and verified mock MCP tools. Does not require API keys or external server dependencies.
* **LIVE Mode (🟢 `LIVE`)**: Activated when valid credentials and services are reachable:
  * Gemini LLM (`GEMINI_API_KEY`)
  * Redis Stack Vector Search (`REDIS_HOST`, `REDIS_PORT`)
  * External MCP Server (`MCP_SERVER_URL`)
* **HYBRID Mode (🔵 `HYBRID`)**: Combines live and mock subsystems (e.g. live Gemini with local in-memory RAG).

---

## 2. Interactive Features

1. **Company & Direction**: Infra Builder is the sole active business domain. Switch between Inbound (unknown/new callers) and Outbound (campaign outreach context).
2. **Text vs Voice Mode**: Full Text Mode active. Voice test panel available for voice pipeline evaluation.
3. **Turn-by-Turn Conversational Interface**: Preserves transcript history, manages turns sequentially, and keeps the call active across tool invocations.
4. **Real-Time Debug & Diagnostic Inspection**:
   * **Session & Lifecycle**: Tracks session ID, domain (`infra_builder`), call direction, active state, conversation stage, and business status.
   * **Intent & Dialogue**: Displays active intent, sub-intent, complete intent history stack, and pending agent questions.
   * **Caller & Entities**: Displays extracted real estate slots (project name, configuration, budget, location, visit date/time) and caller profile without exposing secrets.
   * **Grounded RAG**: Details query, knowledge availability, retrieved chunk IDs, and grounded real estate citations (Skyline Towers, Green Valley Villas, Urban Heights).
   * **MCP Tools**: Clearly distinguishes **Proposed**, **Executed**, and **Verified** statuses with external tracking references. Never displays success merely because an action was proposed.
   * **WhatsApp Messaging**: Displays WhatsApp action status (`accepted`, `delivered`, `failed`) and phone normalization.
   * **Errors & Diagnostics**: Non-fatal error reporting and subsystem health.
5. **Session Reset**: "New Session" and "Reset Call" controls create clean, unpolluted conversation sessions.

---

## 3. Manual Testing Scenarios

### Scenario A — Inbound Property Inquiry
1. In the sidebar, verify **Company: Infra Builder**, **Call Direction: Inbound**.
2. Click **🚀 New Session**.
3. **Caller**:
   > *"Hi, can you tell me about Skyline Towers in Whitefield?"*
   * **Agent Response**: Grounded overview of Skyline Towers 2 & 3 BHK luxury apartments starting from ₹1.2 Crores with clubhouse, swimming pool, and December 2026 possession. Under **Grounded RAG**, observe `knowledge_required: True` and retrieved project chunks.

---

### Scenario B — Site Visit Scheduling Flow
1. In an active Infra Builder call:
2. **Caller**:
   > *"I would like to schedule a site visit for Skyline Towers tomorrow at 10 AM."*
   * **Agent Response**: Agent runs `find_available_slots`, confirms slot availability, and asks for caller's explicit confirmation.
3. **Caller**:
   > *"Yes, please confirm that slot."*
   * **Agent Response**: Agent invokes `create_calendar_event` with `confirmed=True`, verifies booking (`evt_site_visit_...`), and confirms to caller.
   * **Debug Panel Verification**: Open **⚡ MCP Tool Layer**:
     * **Action**: `create_calendar_event`
     * **Confirmed**: `True`
     * **Status**: `succeeded`
     * **Verification Status**: `verified`

---

### Scenario C — WhatsApp Brochure Delivery
1. In an active Infra Builder call:
2. **Caller**:
   > *"Can you send me the brochure on WhatsApp? My number is +91 98765 43210."*
   * **Agent Response**: Agent validates phone number, invokes `send_message`, and confirms WhatsApp brochure dispatch.
   * **Verification**: Tool dispatches `send_message` with recipient `+919876543210` and real estate brochure payload.

---

### Scenario D — Action Failure & Safety Verification
1. Propose WhatsApp delivery without a verified phone number:
2. **Caller**:
   > *"Please WhatsApp me the floor plans."* (with no phone number provided).
   * **Agent Response**:
     > *"I would be happy to send the floor plans via WhatsApp. Could you please share your WhatsApp phone number?"*
   * **Verification**: The tool is NOT dispatched; agent does NOT fabricate delivery success; conversation remains open (`conversation_active: True`).

---

### Scenario E — Explicit Termination vs Non-Terminating Negatives
1. In an active call:
   **Caller**:
   > *"No thanks, that will be all for now."*
   * **Agent Response**: Polite closing or assistance check.
2. **Caller**:
   > *"Goodbye."*
   * **Agent Response**: Warm farewell.
   * **State**: `conversation_active` changes to **False**; `stage` updates to `COMPLETED`; call concludes cleanly.
