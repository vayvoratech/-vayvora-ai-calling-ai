"""Automated Scenario Test Suite for Conversation Engine Audit.

Tests real-life Inbound, Outbound, Intent Switching, RAG Isolation,
and Action Verification flows with live Gemini provider and seeded RAG.
"""

import pytest
import asyncio
import json
from typing import Dict, Any, List

from src.config import Settings
from src.core.types import CallDirection, DomainType, ConversationStage
from src.ui.bootstrap import bootstrap_workbench
from src.ui.service import WorkbenchService


@pytest.fixture(scope="module")
def live_workbench() -> WorkbenchService:
    """Bootstrap a live workbench instance with Gemini and in-memory seeded RAG."""
    settings = Settings()
    return bootstrap_workbench(settings=settings, force_mock=False)


@pytest.mark.asyncio
async def test_flow_1_inbound_vayvora_corporate(live_workbench: WorkbenchService):
    """Flow 1: Inbound Vayvora Corporate conversation."""
    session_id = "test-flow-1-vayvora"
    state = live_workbench.create_inbound_session(
        call_id=session_id,
        caller_phone="+15551234567",
        domain=DomainType.VAYVORA,
        caller_name="Alice Smith",
    )

    # Turn 1: Hello
    state, res1 = await live_workbench.process_turn_async(session_id, "Hello")
    assert "help" in res1.response_text.lower() or "vayvora" in res1.response_text.lower()
    assert state.stage in [ConversationStage.GREETING, ConversationStage.PURPOSE_DISCOVERY, ConversationStage.DISCOVERY]

    # Turn 2: What services does Vayvora provide?
    state, res2 = await live_workbench.process_turn_async(session_id, "What services does Vayvora provide?")
    resp_lower = res2.response_text.lower()
    # Must NOT return a generic greeting
    assert "how can i help you today" not in resp_lower
    # Should mention AI or software or engineering services
    assert any(term in resp_lower for term in ["ai", "software", "agent", "engineering", "service", "development"])

    # Turn 3: Can you email me more details?
    state, res3 = await live_workbench.process_turn_async(session_id, "Can you email me more details?")
    assert "how can i help you today" not in res3.response_text.lower()
    # Agent should ask for email or acknowledge preparing info
    assert any(term in res3.response_text.lower() for term in ["email", "address", "details", "send", "sure", "certainly"])

    # Turn 4: alice@example.com
    state, res4 = await live_workbench.process_turn_async(session_id, "alice@example.com")
    assert "how can i help you today" not in res4.response_text.lower()
    assert any(term in res4.response_text.lower() for term in ["send", "sent", "email", "forward", "alice@example.com", "shared"])


@pytest.mark.asyncio
async def test_flow_2_inbound_vayvora_career(live_workbench: WorkbenchService):
    """Flow 2: Inbound Vayvora Career inquiries."""
    session_id = "test-flow-2-career"
    state = live_workbench.create_inbound_session(
        call_id=session_id,
        caller_phone="+15559876543",
        domain=DomainType.VAYVORA,
        caller_name="Bob Jones",
    )

    # Turn 1: Are you hiring software engineers?
    state, res1 = await live_workbench.process_turn_async(session_id, "Hi, are you hiring software engineers?")
    r1 = res1.response_text.lower()
    assert "how can i help you today" not in r1
    assert any(term in r1 for term in ["hiring", "engineer", "apply", "career", "role", "position", "team"])

    # Turn 2: What is the interview process like?
    state, res2 = await live_workbench.process_turn_async(session_id, "What is the interview process like?")
    r2 = res2.response_text.lower()
    assert "how can i help you today" not in r2
    assert any(term in r2 for term in ["interview", "process", "round", "technical", "screening", "team", "stage"])

    # Turn 3: Thanks, that's all
    state, res3 = await live_workbench.process_turn_async(session_id, "Thanks, that's all")
    r3 = res3.response_text.lower()
    assert any(term in r3 for term in ["welcome", "day", "bye", "goodbye", "help", "pleasure"])


@pytest.mark.asyncio
async def test_flow_3_inbound_edusaas(live_workbench: WorkbenchService):
    """Flow 3: Inbound EduSaaS course inquiry."""
    session_id = "test-flow-3-edusaas"
    state = live_workbench.create_inbound_session(
        call_id=session_id,
        caller_phone="+15553334444",
        domain=DomainType.EDUSAAS,
        caller_name="Charlie Brown",
    )

    # Turn 1: I want to know about your data science course
    state, res1 = await live_workbench.process_turn_async(session_id, "Hi, I want to know about your data science course")
    r1 = res1.response_text.lower()
    assert "how can i help you today" not in r1
    assert any(term in r1 for term in ["data science", "course", "program", "month", "python", "machine learning", "curriculum"])

    # Turn 2: Do you offer placement assistance?
    state, res2 = await live_workbench.process_turn_async(session_id, "Do you offer placement assistance?")
    r2 = res2.response_text.lower()
    assert "how can i help you today" not in r2
    assert any(term in r2 for term in ["placement", "assistance", "career", "support", "interview", "job", "guarantee", "partner"])

    # Turn 3: Can you schedule a demo class for me?
    state, res3 = await live_workbench.process_turn_async(session_id, "Can you schedule a demo class for me?")
    r3 = res3.response_text.lower()
    assert "how can i help you today" not in r3
    assert any(term in r3 for term in ["demo", "schedule", "class", "session", "date", "time", "book", "slot"])


@pytest.mark.asyncio
async def test_flow_4_outbound_vayvora(live_workbench: WorkbenchService):
    """Flow 4: Outbound Vayvora AI Outreach with 'yes' response bug verification."""
    session_id = "test-flow-4-outbound"
    state = live_workbench.create_outbound_session(
        call_id=session_id,
        caller_phone="+15555551234",
        domain=DomainType.VAYVORA,
        caller_name="Alice Smith",
        campaign_id="CAMP-VAYVORA-01",
        campaign_objective="Product follow-up consultation",
        company="Acme Corp",
        known_purpose="Enterprise voice AI agent consultation",
    )

    # Verify agent opened the call
    assert live_workbench.last_turn_result is not None
    opening_text = live_workbench.last_turn_result.response_text
    assert "Alice" in opening_text or "Vayvora" in opening_text

    # Turn 1: Lead responds "yes" (CRITICAL TEST FOR BUG)
    await asyncio.sleep(1.0)
    state, res1 = await live_workbench.process_turn_async(session_id, "yes")
    r1 = res1.response_text.lower()
    # MUST NOT return "Hi, how can I help you today?"
    assert "how can i help you today" not in r1
    assert "hi alice smith, this is" not in r1  # Must not repeat opening
    # Agent should ask discovery or qualification question
    assert state.stage in [ConversationStage.DISCOVERY, ConversationStage.PURPOSE_DISCOVERY, ConversationStage.INFORMATION, ConversationStage.RECOMMENDATION]

    # Turn 2: We need an AI agent for customer support
    await asyncio.sleep(1.0)
    state, res2 = await live_workbench.process_turn_async(session_id, "We need an AI agent for customer support")
    r2 = res2.response_text.lower()
    assert "how can i help you today" not in r2
    assert any(term in r2 for term in ["agent", "support", "voice", "customer", "ai", "solution", "call"])

    # Turn 3: Sure, let's schedule a call for tomorrow
    await asyncio.sleep(1.0)
    state, res3 = await live_workbench.process_turn_async(session_id, "Sure, let's schedule a call for tomorrow")
    r3 = res3.response_text.lower()
    assert "how can i help you today" not in r3
    assert any(term in r3 for term in ["schedule", "call", "tomorrow", "time", "calendar", "confirm", "meeting"])


@pytest.mark.asyncio
async def test_flow_5_outbound_edusaas(live_workbench: WorkbenchService):
    """Flow 5: Outbound EduSaaS Follow-up handling objections."""
    session_id = "test-flow-5-edusaas"
    state = live_workbench.create_outbound_session(
        call_id=session_id,
        caller_phone="+15557778888",
        domain=DomainType.EDUSAAS,
        caller_name="David Miller",
        campaign_id="CAMP-EDU-02",
        campaign_objective="Course enrollment consultation",
        known_purpose="AI Engineering course details",
    )

    assert live_workbench.last_turn_result is not None

    # Turn 1: I'm busy right now
    await asyncio.sleep(1.0)
    state, res1 = await live_workbench.process_turn_async(session_id, "I'm busy right now")
    r1 = res1.response_text.lower()
    assert "how can i help you today" not in r1
    # Call should NOT be abruptly ended; agent should acknowledge and offer callback or email
    assert any(term in r1 for term in ["understand", "busy", "callback", "later", "email", "time", "reach out"])

    # Turn 2: Send me an email instead
    await asyncio.sleep(1.0)
    state, res2 = await live_workbench.process_turn_async(session_id, "Send me an email instead")
    r2 = res2.response_text.lower()
    assert "how can i help you today" not in r2
    assert any(term in r2 for term in ["email", "send", "details", "syllabus", "information", "sure"])


@pytest.mark.asyncio
async def test_flow_6_intent_switching_and_short_replies(live_workbench: WorkbenchService):
    """Flow 6: Edge cases - Intent switching, pending context preemption, and short answers."""
    session_id = "test-flow-6-edge"
    state = live_workbench.create_inbound_session(
        call_id=session_id,
        caller_phone="+15559990000",
        domain=DomainType.EDUSAAS,
        caller_name="Elena Vance",
    )

    # 1. Ask about fees
    await asyncio.sleep(1.0)
    state, res1 = await live_workbench.process_turn_async(session_id, "How much does the AI Engineering course cost?")
    assert "how can i help you today" not in res1.response_text.lower()

    # 2. Sudden intent switch: where are your offices located?
    await asyncio.sleep(1.0)
    state, res2 = await live_workbench.process_turn_async(session_id, "Where are your offices located?")
    r2 = res2.response_text.lower()
    assert "how can i help you today" not in r2
    # Intent must have switched to location/office/general
    assert state.current_intent != "fee_inquiry"

    # 3. Interruption during scheduling: "Wait, before that, what about your refund policy?"
    await asyncio.sleep(1.0)
    state, res3 = await live_workbench.process_turn_async(session_id, "Wait, before that, what about your refund policy?")
    r3 = res3.response_text.lower()
    assert "how can i help you today" not in r3
    assert any(term in r3 for term in ["refund", "policy", "days", "money", "fee", "cancel"])

    # 4. Short answer: "sure"
    await asyncio.sleep(1.0)
    state, res4 = await live_workbench.process_turn_async(session_id, "sure")
    assert "how can i help you today" not in res4.response_text.lower()

    # 5. Short answer: "no"
    await asyncio.sleep(1.0)
    state, res5 = await live_workbench.process_turn_async(session_id, "no")
    assert "how can i help you today" not in res5.response_text.lower()
    # Call should not be terminated by a bare "no"
    assert not state.termination_requested
