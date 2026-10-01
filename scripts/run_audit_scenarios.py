"""Script to execute all 6 audit scenarios with live Gemini + RAG and record full metrics."""

import asyncio
import io
import json
import sys
import time
from typing import Any, Dict, List

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.config import Settings
from src.core.types import DomainType, ConversationStage
from src.ui.bootstrap import bootstrap_workbench
from src.ui.service import WorkbenchService


async def run_scenario_flow(
    service: WorkbenchService,
    title: str,
    direction: str,
    domain: DomainType,
    caller_name: str,
    caller_phone: str,
    caller_email: str,
    caller_company: str,
    campaign_objective: str,
    turns: List[str],
) -> List[Dict[str, Any]]:
    session_id = f"audit-{int(time.time()*1000)%100000}"
    results = []

    print(f"\n========================================================")
    print(f"RUNNING: {title}")
    print(f"Direction: {direction}, Domain: {domain.value}")
    print(f"========================================================")

    if direction == "outbound":
        state = service.create_outbound_session(
            call_id=session_id,
            caller_phone=caller_phone,
            domain=domain,
            caller_name=caller_name,
            campaign_id="CAMP-AUDIT-01",
            campaign_objective=campaign_objective,
            caller_email=caller_email,
            company=caller_company,
        )
        opening_text = service.last_turn_result.response_text if service.last_turn_result else "Hello"
        print(f"[Turn 0 (Agent Opening)]: {opening_text}")
        results.append({
            "turn": 0,
            "speaker": "Agent (Opening)",
            "input": "—",
            "response": opening_text,
            "intent": "outbound_opening",
            "stage": state.stage.value,
            "rag": False,
            "rag_query": None,
            "citations": [],
            "tool": None,
            "latency_ms": service.last_turn_result.latency_ms if service.last_turn_result else 0.0,
            "breakdown": service.last_turn_result.latency_breakdown if service.last_turn_result else {},
            "evaluation": "PASS",
            "reason": "Agent opened call proactively with context",
        })
    else:
        state = service.create_inbound_session(
            call_id=session_id,
            caller_phone=caller_phone,
            domain=domain,
            caller_name=caller_name,
            caller_email=caller_email,
            caller_company=caller_company,
        )

    for idx, user_input in enumerate(turns, start=1):
        await asyncio.sleep(1.2)
        t0 = time.perf_counter()
        state, turn_res = await service.process_turn_async(session_id, user_input)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        diag = service.get_debug_payload(state, turn_res)

        rag_used = diag["rag"]["knowledge_required"]
        rag_query = diag["rag"]["knowledge_query"]
        citations = diag["rag"]["grounded_citations"]
        tool_info = diag["tools"]["proposed_action"]["tool_name"] if diag["tools"].get("proposed_action") else None
        intent = diag["intent"]["current_intent"]
        stage = diag["session"]["conversation_stage"]

        # Check for bug: "how can i help you today"
        resp_lower = turn_res.response_text.lower()
        has_greeting_bug = "how can i help you today" in resp_lower and user_input.lower() != "hello"

        eval_status = "FAIL" if has_greeting_bug else "PASS"
        reason = "Greeting bug occurred!" if has_greeting_bug else "Natural context-aware response"

        print(f"[Turn {idx} (Caller)]: {user_input}")
        print(f"[Turn {idx} (Agent)]: {turn_res.response_text}")
        print(f"       -> Intent: {intent}, Stage: {stage}, RAG: {rag_used}, Tool: {tool_info}, Latency: {turn_res.latency_ms:.1f}ms")

        results.append({
            "turn": idx,
            "speaker": "Caller / Agent",
            "input": user_input,
            "response": turn_res.response_text,
            "intent": intent,
            "stage": stage,
            "rag": rag_used,
            "rag_query": rag_query,
            "citations": citations,
            "tool": tool_info,
            "latency_ms": turn_res.latency_ms or elapsed_ms,
            "breakdown": turn_res.latency_breakdown,
            "evaluation": eval_status,
            "reason": reason,
        })

    return results


async def main():
    print("Bootstrapping workbench service with Gemini...")
    settings = Settings()
    service = bootstrap_workbench(settings=settings, force_mock=False)

    all_scenarios = {}

    # Flow 1: Inbound Vayvora Corporate
    f1 = await run_scenario_flow(
        service=service,
        title="Flow 1: Inbound Vayvora Corporate",
        direction="inbound",
        domain=DomainType.VAYVORA,
        caller_name="Alice Smith",
        caller_phone="+15551234567",
        caller_email="alice@example.com",
        caller_company="Acme Corp",
        campaign_objective="",
        turns=[
            "Hello",
            "What services does Vayvora provide?",
            "Can you email me more details?",
            "alice@example.com",
        ],
    )
    all_scenarios["flow_1_inbound_vayvora_corporate"] = f1

    # Flow 2: Inbound Vayvora Career
    f2 = await run_scenario_flow(
        service=service,
        title="Flow 2: Inbound Vayvora Career",
        direction="inbound",
        domain=DomainType.VAYVORA,
        caller_name="Bob Jones",
        caller_phone="+15559876543",
        caller_email="bob@example.com",
        caller_company="Independent",
        campaign_objective="",
        turns=[
            "Hi, are you hiring software engineers?",
            "What is the interview process like?",
            "Thanks, that's all",
        ],
    )
    all_scenarios["flow_2_inbound_vayvora_career"] = f2

    # Flow 3: Inbound EduSaaS
    f3 = await run_scenario_flow(
        service=service,
        title="Flow 3: Inbound EduSaaS Course Inquiry",
        direction="inbound",
        domain=DomainType.EDUSAAS,
        caller_name="Charlie Brown",
        caller_phone="+15553334444",
        caller_email="charlie@example.com",
        caller_company="Student",
        campaign_objective="",
        turns=[
            "Hi, I want to know about your data science course",
            "Do you offer placement assistance?",
            "Can you schedule a demo class for me?",
        ],
    )
    all_scenarios["flow_3_inbound_edusaas"] = f3

    # Flow 4: Outbound Vayvora AI Outreach
    f4 = await run_scenario_flow(
        service=service,
        title="Flow 4: Outbound Vayvora AI Outreach",
        direction="outbound",
        domain=DomainType.VAYVORA,
        caller_name="Alice Smith",
        caller_phone="+15555551234",
        caller_email="alice@example.com",
        caller_company="Acme Corp",
        campaign_objective="Explore enterprise AI calling system",
        turns=[
            "yes",
            "We need an AI agent for customer support",
            "Sure, let's schedule a call for tomorrow",
        ],
    )
    all_scenarios["flow_4_outbound_vayvora"] = f4

    # Flow 5: Outbound EduSaaS Follow-up
    f5 = await run_scenario_flow(
        service=service,
        title="Flow 5: Outbound EduSaaS Follow-up",
        direction="outbound",
        domain=DomainType.EDUSAAS,
        caller_name="David Miller",
        caller_phone="+15557778888",
        caller_email="david@example.com",
        caller_company="Candidate",
        campaign_objective="Course enrollment consultation",
        turns=[
            "I'm busy right now",
            "Send me an email instead",
        ],
    )
    all_scenarios["flow_5_outbound_edusaas"] = f5

    # Flow 6: Intent Switching & Edge Cases
    f6 = await run_scenario_flow(
        service=service,
        title="Flow 6: Intent Switching & Edge Cases",
        direction="inbound",
        domain=DomainType.EDUSAAS,
        caller_name="Elena Vance",
        caller_phone="+15559990000",
        caller_email="elena@example.com",
        caller_company="Tech Corp",
        campaign_objective="",
        turns=[
            "How much does the AI Engineering course cost?",
            "Where are your offices located?",
            "Wait, before that, what about your refund policy?",
            "sure",
            "no",
        ],
    )
    all_scenarios["flow_6_intent_switching"] = f6

    with open("audit_scenario_results.json", "w", encoding="utf-8") as f:
        json.dump(all_scenarios, f, indent=2)

    print("\nSaved detailed scenario audit results to audit_scenario_results.json")


if __name__ == "__main__":
    asyncio.run(main())
