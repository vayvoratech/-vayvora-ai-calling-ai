# Streamlit Testing Workbench (Phase 6)

A developer-facing interactive testing and evaluation workbench for pair-testing the unified AI voice agent across **EduSaaS** and **Vayvora** business domains before connecting real audio, VAD, STT/TTS, or telephony pipelines.

---

## 1. Quick Start: How to Run the Workbench

Ensure your virtual environment is active, then launch the Streamlit app:

```bash
.\.venv\Scripts\python.exe -m streamlit run src/ui/app.py
```

By default, the application opens in your browser at `http://localhost:8501`.

### Runtime Modes
The application displays an explicit status badge in the sidebar:
* **MOCK Mode (🟡 `MOCK`)**: Uses `InteractiveMockLLMProvider`, in-memory vector store seeded with documents from `knowledge_base/`, and verified mock MCP tools. Does not require API keys or external server dependencies.
* **LIVE Mode (🟢 `LIVE`)**: Activated when valid credentials and services are reachable:
  * Gemini LLM (`GEMINI_API_KEY`)
  * Redis Stack Vector Search (`REDIS_HOST`, `REDIS_PORT`)
  * External MCP Server (`MCP_SERVER_URL`)
* **HYBRID Mode (🔵 `HYBRID`)**: Combines live and mock subsystems (e.g. live Gemini with local in-memory RAG).

---

## 2. Interactive Features

1. **Company & Direction Switcher**: Seamlessly switch between EduSaaS and Vayvora, and between Inbound (unknown/new callers) and Outbound (enriched campaign context).
2. **Text vs Voice Mode**: Full Text Mode active. Voice Mode is visibly marked as *Coming in Phase 7/8* (do not implement audio or telephony until subsequent phases).
3. **Turn-by-Turn Conversational Interface**: Preserves transcript history, manages turns sequentially, and keeps the call active across tool invocations.
4. **Real-Time Debug & Diagnostic Inspection**:
   * **Session & Lifecycle**: Tracks session ID, domain, call direction, active state, conversation stage, and business status.
   * **Intent & Dialogue**: Displays active intent, sub-intent, complete intent history stack, and pending agent questions.
   * **Caller & Entities**: Displays extracted slots and caller profile without exposing secrets.
   * **Grounded RAG**: Details query, knowledge availability, retrieved chunk IDs, and grounded citations.
   * **MCP Tools**: Clearly distinguishes **Proposed**, **Executed**, and **Verified** statuses with external tracking references. Never displays success merely because an action was proposed.
   * **Errors & Diagnostics**: Non-fatal error reporting and subsystem health.
5. **Session Reset**: "New Session" and "Reset Call" controls create clean, unpolluted conversation sessions.

---

## 3. Manual Testing Scenarios

### Scenario A — EduSaaS Inbound (Course Inquiries)
1. In the sidebar, select **Company: EduSaaS**, **Call Direction: Inbound**.
2. Click **🚀 New Session**.
3. **Caller**:
   > *"Hi"*
   * **Agent Response**: Courteous greeting introducing EduSaaS AI & Data Science courses.
4. **Caller**:
   > *"I want to know about your courses."*
   * **Agent Response**: Explains available programs (AI Engineering, Data Science). In the debug panel under **Grounded RAG**, observe `knowledge_required: True` and retrieved course chunks.
5. **Caller**:
   > *"What course would be suitable for me?"*
   * **Agent Response**: Offers recommendations based on background or asks about beginner vs advanced experience.

---

### Scenario B — Vayvora Corporate (Enterprise Consultation)
1. Select **Company: Vayvora**, **Call Direction: Inbound**.
2. Click **🚀 New Session**.
3. **Caller**:
   > *"We are looking for an AI solution for our company."*
   * **Agent Response**: Introduces Vayvora enterprise AI workflows and software engineering solutions.
4. **Caller**:
   > *"Can we schedule a meeting?"*
   * **Agent Response**: Confirms scheduling for Tomorrow at 10:00 AM.
   * **Debug Panel Verification**: Open **⚡ MCP Tool Layer**:
     * **Proposed Action**: `create_calendar_event`
     * **Status**: `succeeded`
     * **Verification Status**: `verified`
     * **External Reference**: `evt_cal_987654`

---

### Scenario C — Domain & Intent Preemption (Mid-Call Switching)
1. Start an Inbound **EduSaaS** session.
2. **Caller**:
   > *"Tell me about courses."*
   * Agent replies regarding EduSaaS courses (Domain: `edusaas`, Intent: `course_information`).
3. **Caller**:
   > *"Actually, I want to know about Vayvora careers."*
   * **Agent Response**: Acknowledges engineering opportunities at Vayvora and queues recruiter follow-up.
   * **Debug Panel Verification**:
     * **Domain**: Dynamically switched to `vayvora`.
     * **Intent**: Preempted to `career_inquiry`.
     * **Intent History**: `["course_information", "career_inquiry"]`.

---

### Scenario D — Action Failure & Safety Verification
1. Propose an email without a recipient or simulate an unreachable MCP tool server.
2. **Caller**:
   > *"Please email me the brochure."* (with no caller email configured).
   * **Agent Response**:
     > *"I would be happy to email those details to you. Could you please share your email address?"*
   * **Verification**: The tool is NOT dispatched; agent does NOT fabricate delivery success; conversation remains open (`conversation_active: True`).
3. In case of server error:
   * **Agent Response**: Explains system issue politely.
   * **Verification Status**: Displays `unverified` / `failed`; does not claim success; call remains active.

---

### Scenario E — Explicit Termination vs Non-Terminating Negatives
1. In an active call:
   **Caller**:
   > *"No thanks."*
   * **Agent Response**: Confirms understanding and asks if any other assistance is needed.
   * **State**: `conversation_active` remains **True**; `termination_requested` is **False**.
2. **Caller**:
   > *"That's all, thank you. Goodbye."*
   * **Agent Response**: Warm farewell.
   * **State**: `conversation_active` changes to **False**; `stage` updates to `COMPLETED`; call concludes cleanly.
