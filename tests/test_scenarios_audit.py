"""Automated Scenario Test Suite for Conversation Engine Audit.

Tests real-life Inbound, Outbound, Intent Switching, RAG Isolation,
and Action Verification flows for Infra Builder Real Estate.
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
async def test_flow_1_inbound_infra_builder_overview(live_workbench: WorkbenchService):
    """Flow 1: Inbound Infra Builder residential project inquiries & WhatsApp brochure."""
    session_id = "test-flow-1-infra"
    state = live_workbench.create_inbound_session(
        call_id=session_id,
        caller_phone="+919845123890",
        domain=DomainType.INFRA_BUILDER,
        caller_name="Alice Smith",
    )

    # Turn 1: Hello
    state, res1 = await live_workbench.process_turn_async(session_id, "Hello")
    assert any(term in res1.response_text.lower() for term in ["help", "assist", "welcome", "infra builder", "property"])
    assert state.stage in [ConversationStage.GREETING, ConversationStage.PURPOSE_DISCOVERY, ConversationStage.DISCOVERY, ConversationStage.INFORMATION]

    # Turn 2: What residential projects do you offer in Bangalore?
    state, res2 = await live_workbench.process_turn_async(session_id, "What residential projects do you offer in Bangalore?")
    resp_lower = res2.response_text.lower()
    # Must NOT return a generic greeting
    assert "how can i help you today" not in resp_lower
    # Should mention verified developments
    assert any(term in resp_lower for term in ["skyline", "towers", "green valley", "villas", "urban heights", "residential", "project", "whitefield", "sarjapur"])

    # Turn 3: Can you share the brochure on WhatsApp?
    state, res3 = await live_workbench.process_turn_async(session_id, "Can you share the brochure on WhatsApp?")
    r3_lower = res3.response_text.lower()
    assert "how can i help you today" not in r3_lower
    # Agent should confirm or acknowledge WhatsApp sharing
    assert any(term in r3_lower for term in ["whatsapp", "share", "send", "brochure", "details", "sure", "certainly", "number"])


@pytest.mark.asyncio
async def test_flow_2_inbound_project_specifics(live_workbench: WorkbenchService):
    """Flow 2: Inbound project specifics and amenities for Skyline Towers."""
    session_id = "test-flow-2-skyline"
    state = live_workbench.create_inbound_session(
        call_id=session_id,
        caller_phone="+919845123890",
        domain=DomainType.INFRA_BUILDER,
        caller_name="Bob Jones",
    )

    # Turn 1: Configurations and pricing for Skyline Towers
    state, res1 = await live_workbench.process_turn_async(session_id, "Hi, what configurations and prices do you have at Skyline Towers?")
    r1 = res1.response_text.lower()
    assert "how can i help you today" not in r1
    assert any(term in r1 for term in ["skyline", "bhk", "crore", "1.45", "whitefield", "bedroom", "starting"])

    # Turn 2: What amenities are included?
    state, res2 = await live_workbench.process_turn_async(session_id, "What amenities are included in the project?")
    r2 = res2.response_text.lower()
    assert "how can i help you today" not in r2
    assert any(term in r2 for term in ["amenities", "clubhouse", "pool", "gym", "court", "park", "security", "garden"])

    # Turn 3: Thanks, that's all
    state, res3 = await live_workbench.process_turn_async(session_id, "Thanks, that's all")
    r3 = res3.response_text.lower()
    assert any(term in r3 for term in ["welcome", "day", "bye", "goodbye", "help", "pleasure", "assist"])


@pytest.mark.asyncio
async def test_flow_3_inbound_site_visit_scheduling(live_workbench: WorkbenchService):
    """Flow 3: Inbound site visit scheduling inquiry and availability check."""
    session_id = "test-flow-3-visit"
    state = live_workbench.create_inbound_session(
        call_id=session_id,
        caller_phone="+919845123890",
        domain=DomainType.INFRA_BUILDER,
        caller_name="Charlie Brown",
    )

    # Turn 1: I want to schedule a site visit
    state, res1 = await live_workbench.process_turn_async(session_id, "Hi, I want to schedule a site visit to Skyline Towers")
    r1 = res1.response_text.lower()
    assert "how can i help you today" not in r1
    assert any(term in r1 for term in ["site visit", "visit", "schedule", "date", "time", "slot", "when", "day"])

    # Turn 2: What slots are available tomorrow?
    state, res2 = await live_workbench.process_turn_async(session_id, "What slots do you have available tomorrow?")
    r2 = res2.response_text.lower()
    assert "how can i help you today" not in r2
    assert any(term in r2 for term in ["tomorrow", "slot", "available", "time", "morning", "afternoon", "10", "2", "4"])


@pytest.mark.asyncio
async def test_flow_4_outbound_infra_builder(live_workbench: WorkbenchService):
    """Flow 4: Outbound Infra Builder outreach with identity confirmation and consultative discovery."""
    session_id = "test-flow-4-outbound"
    state = live_workbench.create_outbound_session(
        call_id=session_id,
        caller_phone="+919845123890",
        domain=DomainType.INFRA_BUILDER,
        caller_name="Alice Smith",
        campaign_id="CAMP-RE-01",
        campaign_objective="Follow up on 3 BHK apartment inquiries in Bangalore",
        company="Acme Corp",
        known_purpose="3 BHK luxury apartment inquiry",
    )

    # Verify agent opened the call with identity confirmation
    assert live_workbench.last_turn_result is not None
    opening_text = live_workbench.last_turn_result.response_text
    assert "Alice" in opening_text or "Infra Builder" in opening_text

    # Turn 1: Lead confirms identity
    await asyncio.sleep(0.5)
    state, res1 = await live_workbench.process_turn_async(session_id, "yes, speaking")
    r1 = res1.response_text.lower()
    # Must NOT repeat full opening
    assert "hi alice smith, this is" not in r1
    assert state.stage in [ConversationStage.GREETING, ConversationStage.DISCOVERY, ConversationStage.PURPOSE_DISCOVERY, ConversationStage.INFORMATION]

    # Turn 2: Inquire about 3 BHK luxury residences
    await asyncio.sleep(0.5)
    state, res2 = await live_workbench.process_turn_async(session_id, "We are looking for a 3 BHK luxury apartment in Bangalore")
    r2 = res2.response_text.lower()
    assert "how can i help you today" not in r2
    assert any(term in r2 for term in ["skyline", "towers", "whitefield", "3 bhk", "luxury", "apartment", "residence", "unit"])

    # Turn 3: Can you share the details on WhatsApp?
    await asyncio.sleep(0.5)
    state, res3 = await live_workbench.process_turn_async(session_id, "Can you share the details on WhatsApp?")
    r3 = res3.response_text.lower()
    assert "how can i help you today" not in r3
    assert any(term in r3 for term in ["whatsapp", "send", "details", "brochure", "share", "number", "sent"])


@pytest.mark.asyncio
async def test_flow_5_outbound_objection_handling(live_workbench: WorkbenchService):
    """Flow 5: Outbound Follow-up handling busy objection gracefully."""
    session_id = "test-flow-5-objection"
    state = live_workbench.create_outbound_session(
        call_id=session_id,
        caller_phone="+919845123890",
        domain=DomainType.INFRA_BUILDER,
        caller_name="David Miller",
        campaign_id="CAMP-RE-02",
        campaign_objective="Follow up on Green Valley Villas inquiry",
        known_purpose="Sarjapur Road villa inquiries",
    )

    assert live_workbench.last_turn_result is not None

    # Turn 1: Caller confirms identity
    await asyncio.sleep(0.5)
    state, res_id = await live_workbench.process_turn_async(session_id, "yes speaking")

    # Turn 2: Caller says busy right now
    await asyncio.sleep(0.5)
    state, res1 = await live_workbench.process_turn_async(session_id, "I'm busy right now")
    r1 = res1.response_text.lower()
    assert "how can i help you today" not in r1
    # Call should NOT be abruptly ended; agent should acknowledge and offer callback or WhatsApp
    assert any(term in r1 for term in ["understand", "busy", "callback", "later", "whatsapp", "time", "reach out", "another time", "no problem"])

    # Turn 3: Send details on WhatsApp instead
    await asyncio.sleep(0.5)
    state, res2 = await live_workbench.process_turn_async(session_id, "Send me the details on WhatsApp instead")
    r2 = res2.response_text.lower()
    assert "how can i help you today" not in r2
    assert any(term in r2 for term in ["whatsapp", "send", "details", "brochure", "information", "sure", "certainly"])


@pytest.mark.asyncio
async def test_flow_6_intent_switching_and_short_replies(live_workbench: WorkbenchService):
    """Flow 6: Edge cases - Intent switching, policy inquiries, and short answers."""
    session_id = "test-flow-6-edge"
    state = live_workbench.create_inbound_session(
        call_id=session_id,
        caller_phone="+919845123890",
        domain=DomainType.INFRA_BUILDER,
        caller_name="Elena Vance",
    )

    # 1. Ask about pricing
    await asyncio.sleep(0.5)
    state, res1 = await live_workbench.process_turn_async(session_id, "What is the price for Green Valley Villas?")
    assert "how can i help you today" not in res1.response_text.lower()

    # 2. Sudden intent switch: where is your head office located?
    await asyncio.sleep(1.0)
    state, res2 = await live_workbench.process_turn_async(session_id, "Where is your head office located?")
    r2 = res2.response_text.lower()
    assert "how can i help you today" not in r2
    assert any(term in r2 for term in ["indiranagar", "office", "head", "bangalore", "located", "road", "100 feet", "trouble", "retry", "apologize"])

    # 3. Policy inquiry
    await asyncio.sleep(0.5)
    state, res3 = await live_workbench.process_turn_async(session_id, "What is your site visit cancellation policy?")
    r3 = res3.response_text.lower()
    assert "how can i help you today" not in r3
    assert any(term in r3 for term in ["visit", "policy", "reschedule", "cancel", "fee", "charge", "notice", "time", "advance"])

    # 4. Short answer: "sure"
    await asyncio.sleep(0.5)
    state, res4 = await live_workbench.process_turn_async(session_id, "sure")
    assert "how can i help you today" not in res4.response_text.lower()

    # 5. Short answer: "no"
    await asyncio.sleep(0.5)
    state, res5 = await live_workbench.process_turn_async(session_id, "no")
    assert "how can i help you today" not in res5.response_text.lower()
    # Call should not be terminated by a bare "no"
    assert not state.termination_requested
