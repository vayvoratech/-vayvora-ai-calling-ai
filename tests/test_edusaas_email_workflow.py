"""Comprehensive test suite for EduSaaS Professional Email Generation & Sending Workflow.

Verifies all 16 required test scenarios:
TEST 1: Inbound EduSaaS + profile email + course information request (sends professional email).
TEST 2: Inbound EduSaaS + missing email (asks for recipient email, blocks tool).
TEST 3: Inbound EduSaaS + caller provides email (validates, stores, generates, sends, confirms).
TEST 4: Inbound EduSaaS + caller corrects email (corrected email is used, old email discarded).
TEST 5: Outbound EduSaaS + profile email exists (sends immediately, no redundant asking).
TEST 6: Outbound EduSaaS + profile email missing (asks for recipient email, blocks tool).
TEST 7: Caller requests course information (verified EduSaaS RAG/catalog info used).
TEST 8: Caller requests pricing (only verified pricing ₹5,000 INR used, no fabricated numbers).
TEST 9: Requested information is unavailable (no fabricated info, honest guidance).
TEST 10: Caller changes request (latest request wins: Data Science -> AI course).
TEST 11: Contact name and agent name are different (greeting addresses contact, signature is agent).
TEST 12: No agent name configured (EduSaaS organization signature, no invented name).
TEST 13: send_email fails (honest failure response, no false success claim).
TEST 14: No email requested (verbal answer only, no automatic email).
TEST 15: Caller requests vague email (asks 'What information would you like me to send you?').
TEST 16: Inbound conversation changes intent (intent changes while domain remains locked to EduSaaS).
"""

import re
from typing import Any, Dict, List, Optional
import pytest

from src.config import Settings
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine
from src.core.llm import MockLLMProvider
from src.core.types import (
    CallDirection,
    ConversationStage,
    DomainType,
    ToolCallRequest,
    ToolExecutionResult,
)
from src.domains.edusaas.email import (
    EDUSAAS_OFFICIAL_PORTAL,
    EDUSAAS_ORGANIZATION_SIGNATURE,
    EDUSAAS_STANDARD_TUITION,
    VERIFIED_COURSES,
    generate_edusaas_email,
    is_explicit_email_request,
    is_vague_email_request,
)
from src.domains.registry import get_domain_registry
from src.state.manager import ConversationStateManager
from src.tools.mcp_client import MockToolProvider


class FailingMockToolProvider(MockToolProvider):
    """Tool provider where send_email explicitly fails."""

    async def execute_tool(self, request: ToolCallRequest) -> ToolExecutionResult:
        if request.tool_name == "send_email":
            return ToolExecutionResult(
                tool_name=request.tool_name,
                success=False,
                error_message="SMTP connection timed out",
            )
        return await super().execute_tool(request)


@pytest.fixture
def base_settings():
    return Settings(
        agent_name="Sarah",
        smtp_from_email="noreply@vayvora.com",
    )


@pytest.fixture
def mock_tool_provider():
    return MockToolProvider()


@pytest.fixture
def state_mgr():
    return ConversationStateManager()


def make_engine(settings: Settings, tool_provider: Optional[Any] = None) -> ConversationEngine:
    return ConversationEngine(
        llm_provider=MockLLMProvider(),
        tool_provider=tool_provider or MockToolProvider(),
        settings=settings,
        registry=get_domain_registry(),
    )


# =============================================================================
# TEST 1: Inbound EduSaaS + Profile Email + Course Info Request
# =============================================================================

@pytest.mark.asyncio
async def test_scenario_1_inbound_profile_email_course_info(base_settings, mock_tool_provider, state_mgr):
    """TEST 1: Inbound EduSaaS + profile email + course info request -> sends professional email."""
    engine = make_engine(base_settings, mock_tool_provider)
    state = state_mgr.create_inbound_state(
        call_id="call-in-1",
        caller_phone="+15551112233",
        caller_name="Priya",
        caller_email="priya@example.com",
    )
    # Lock domain to EduSaaS
    state.lock_domain(DomainType.EDUSAAS)

    res = await engine.process_user_turn(state, "Can you send me the Data Science course details?")

    # Verify tool execution
    assert len(mock_tool_provider.dispatched_requests) == 1
    req = mock_tool_provider.dispatched_requests[0]
    assert req.action_name == "send_email"
    assert req.arguments.get("recipient") == "priya@example.com"
    assert "Data Science" in req.arguments.get("subject", "")

    # Verify generated email content
    body = req.arguments.get("body", "")
    assert "Hi Priya," in body
    assert "Thank you for contacting EduSaaS." in body
    assert "Exploratory data analysis" in body
    assert "Python, NumPy, Pandas" in body
    assert "₹5,000 INR" in body
    assert "Sarah" in body
    assert EDUSAAS_ORGANIZATION_SIGNATURE in body

    # Verify honest confirmation mentioning recipient
    assert "priya@example.com" in res.response_text
    assert "Data Science" in res.response_text or "course details" in res.response_text


# =============================================================================
# TEST 2: Inbound EduSaaS + Missing Email
# =============================================================================

@pytest.mark.asyncio
async def test_scenario_2_inbound_missing_email(base_settings, mock_tool_provider, state_mgr):
    """TEST 2: Inbound EduSaaS + missing email -> asks for email, blocks send_email."""
    engine = make_engine(base_settings, mock_tool_provider)
    state = state_mgr.create_inbound_state(
        call_id="call-in-2",
        caller_phone="+15552223344",
        caller_name="Rahul",
        caller_email=None,  # missing
    )
    state.lock_domain(DomainType.EDUSAAS)

    res = await engine.process_user_turn(state, "Can you send me the course details?")

    # Tool must NOT be executed
    assert len(mock_tool_provider.dispatched_requests) == 0
    # Agent asks for email address
    assert "what email address should i send" in res.response_text.lower() or "share your email" in res.response_text.lower()
    assert state.pending_action == "send_email"


# =============================================================================
# TEST 3: Inbound EduSaaS + Caller Provides Email
# =============================================================================

@pytest.mark.asyncio
async def test_scenario_3_inbound_caller_provides_email(base_settings, mock_tool_provider, state_mgr):
    """TEST 3: Inbound EduSaaS + caller provides email -> validates, stores, sends, confirms."""
    engine = make_engine(base_settings, mock_tool_provider)
    state = state_mgr.create_inbound_state(
        call_id="call-in-3",
        caller_phone="+15553334455",
        caller_name="Rahul",
        caller_email=None,
    )
    state.lock_domain(DomainType.EDUSAAS)

    # Turn 1: Caller requests course details
    await engine.process_user_turn(state, "Can you send me the course details?")
    assert len(mock_tool_provider.dispatched_requests) == 0

    # Turn 2: Caller provides email
    res2 = await engine.process_user_turn(state, "rahul@example.com")

    # Tool executed with provided email
    assert len(mock_tool_provider.dispatched_requests) == 1
    req = mock_tool_provider.dispatched_requests[0]
    assert req.action_name == "send_email"
    assert req.arguments.get("recipient") == "rahul@example.com"
    assert state.caller.email == "rahul@example.com"

    # Verbal confirmation
    assert "rahul@example.com" in res2.response_text
    assert "sent" in res2.response_text.lower()


# =============================================================================
# TEST 4: Inbound EduSaaS + Caller Corrects Email
# =============================================================================

@pytest.mark.asyncio
async def test_scenario_4_inbound_caller_corrects_email(base_settings, mock_tool_provider, state_mgr):
    """TEST 4: Inbound EduSaaS + caller corrects email -> uses corrected email."""
    engine = make_engine(base_settings, mock_tool_provider)
    state = state_mgr.create_inbound_state(
        call_id="call-in-4",
        caller_phone="+15554445566",
        caller_name="Priya",
        caller_email="priya@example.com",
    )
    state.lock_domain(DomainType.EDUSAAS)

    # Caller explicitly corrects email
    res = await engine.process_user_turn(
        state,
        "Actually, don't send it to that email. Use priya.personal@example.com.",
    )

    # Must be sent to corrected address
    assert len(mock_tool_provider.dispatched_requests) == 1
    req = mock_tool_provider.dispatched_requests[0]
    assert req.arguments.get("recipient") == "priya.personal@example.com"
    assert state.caller.email == "priya.personal@example.com"
    assert "priya.personal@example.com" in res.response_text


# =============================================================================
# TEST 5: Outbound EduSaaS + Profile Email Exists
# =============================================================================

@pytest.mark.asyncio
async def test_scenario_5_outbound_profile_email_exists(base_settings, mock_tool_provider, state_mgr):
    """TEST 5: Outbound EduSaaS + profile email exists -> sends without re-asking."""
    engine = make_engine(base_settings, mock_tool_provider)
    state = state_mgr.create_outbound_state(
        call_id="call-out-5",
        contact_name="Alice",
        caller_phone="+15555556677",
        contact_email="alice@example.com",
        domain=DomainType.EDUSAAS,
        agent_name="Sarah",
        campaign_objective="Discuss AI course enrollment",
    )

    res = await engine.process_user_turn(state, "Can you send me the course details?")

    # Dispatched immediately without asking
    assert len(mock_tool_provider.dispatched_requests) == 1
    req = mock_tool_provider.dispatched_requests[0]
    assert req.arguments.get("recipient") == "alice@example.com"

    # Outbound email opening
    body = req.arguments.get("body", "")
    assert "Hi Alice," in body
    assert "Thank you for speaking with us today." in body
    assert "Sarah" in body
    assert "Alice" in res.response_text or "alice@example.com" in res.response_text


# =============================================================================
# TEST 6: Outbound EduSaaS + Profile Email Missing
# =============================================================================

@pytest.mark.asyncio
async def test_scenario_6_outbound_profile_email_missing(base_settings, mock_tool_provider, state_mgr):
    """TEST 6: Outbound EduSaaS + profile email missing -> asks for recipient email."""
    engine = make_engine(base_settings, mock_tool_provider)
    state = state_mgr.create_outbound_state(
        call_id="call-out-6",
        contact_name="Alice",
        caller_phone="+15556667788",
        contact_email=None,  # missing
        domain=DomainType.EDUSAAS,
        agent_name="Sarah",
    )

    res = await engine.process_user_turn(state, "Can you send me the course details?")

    # Must NOT send email to fallback or company address
    assert len(mock_tool_provider.dispatched_requests) == 0
    assert "what email address should i send" in res.response_text.lower()
    assert state.pending_action == "send_email"


# =============================================================================
# TEST 7: Caller Requests Course Information (Grounded in Verified Knowledge)
# =============================================================================

def test_scenario_7_verified_course_information():
    """TEST 7: Caller requests course info -> verified EduSaaS knowledge used."""
    email = generate_edusaas_email(
        direction=CallDirection.INBOUND,
        contact_name="Rahul",
        agent_name="Sarah",
        topic_requested="Data Science",
        caller_message="Send me the Data Science course details",
        recipient="rahul@example.com",
    )

    assert email.subject == "Data Science Course Information"
    assert "Data Science & Business Analytics" in email.body
    assert "Exploratory data analysis" in email.body
    assert "Python, NumPy, Pandas" in email.body
    assert "₹5,000 INR" in email.body
    assert EDUSAAS_OFFICIAL_PORTAL in email.body
    assert "Hi Rahul," in email.body
    assert "Thank you for contacting EduSaaS." in email.body
    assert "Sarah" in email.body
    # Ensure no internal leakage
    assert not re.search(r"\brag\b", email.body.lower())
    assert not re.search(r"\bllm\b", email.body.lower())
    assert not re.search(r"\bgemini\b", email.body.lower())


# =============================================================================
# TEST 8: Caller Requests Pricing (Strict Anti-Hallucination)
# =============================================================================

def test_scenario_8_verified_pricing_only():
    """TEST 8: Caller requests pricing -> only verified pricing used (₹5,000 INR)."""
    email = generate_edusaas_email(
        direction=CallDirection.INBOUND,
        contact_name="Aarav",
        agent_name="Sarah",
        topic_requested="pricing",
        caller_message="Send me the course fees and financial options",
        recipient="aarav@example.com",
    )

    assert email.subject == "EduSaaS Tuition & Program Pricing Details"
    assert "₹5,000 INR (Five Thousand Rupees) per program track" in email.body
    assert "Partial assistance and installment options may be reviewed" in email.body
    # Strict anti-hallucination: No fake amounts like ₹50,000
    assert "₹50,000" not in email.body
    assert "₹25,000" not in email.body


# =============================================================================
# TEST 9: Requested Information is Unavailable (No Fabricated Info)
# =============================================================================

def test_scenario_9_unavailable_information_honest_guidance():
    """TEST 9: Requested info unavailable -> no fabricated info, honest guidance."""
    email = generate_edusaas_email(
        direction=CallDirection.INBOUND,
        contact_name="Neha",
        agent_name="Sarah",
        topic_requested="offline classroom batch in Mumbai",
        caller_message="Send me details about the offline classroom batch in Mumbai",
        recipient="neha@example.com",
    )

    # Must NOT invent classroom location, dates, or offline centers
    assert "verified details for this specific option are not currently available" in email.body.lower()
    assert "academic advisory team can provide personalized guidance" in email.body.lower()
    assert "mumbai center" not in email.body.lower()


# =============================================================================
# TEST 10: Caller Changes Request (Latest Request Wins)
# =============================================================================

@pytest.mark.asyncio
async def test_scenario_10_latest_request_wins(base_settings, mock_tool_provider, state_mgr):
    """TEST 10: Caller changes request (Data Science -> AI) -> latest request wins."""
    engine = make_engine(base_settings, mock_tool_provider)
    state = state_mgr.create_inbound_state(
        call_id="call-in-10",
        caller_phone="+15557778899",
        caller_name="Vikram",
        caller_email="vikram@example.com",
    )
    state.lock_domain(DomainType.EDUSAAS)

    # Turn 1: Caller originally requests Data Science
    await engine.process_user_turn(state, "Send me the Data Science course details.")

    # Turn 2: Caller changes mind to AI
    res2 = await engine.process_user_turn(state, "Actually, send me the AI course information instead.")

    # Latest action must be for AI, not Data Science
    latest_req = mock_tool_provider.dispatched_requests[-1]
    assert latest_req.action_name == "send_email"
    subject = latest_req.arguments.get("subject", "")
    assert "Artificial Intelligence" in subject or "AI" in subject
    body = latest_req.arguments.get("body", "")
    assert "Artificial Intelligence & Machine Learning" in body
    assert "Generative AI, LLMs" in body
    assert "AI" in res2.response_text or "course details" in res2.response_text


# =============================================================================
# TEST 11: Contact Name != Agent Name (Identity Invariant)
# =============================================================================

def test_scenario_11_contact_name_distinct_from_agent_name():
    """TEST 11: Contact name != agent name -> greeting uses contact, signature uses agent."""
    email = generate_edusaas_email(
        direction=CallDirection.OUTBOUND,
        contact_name="Alice",
        agent_name="Sarah",
        topic_requested="ai_ml",
        recipient="alice@example.com",
    )

    # Greeting addresses contact Alice
    assert "Hi Alice," in email.body
    # Signature is Sarah
    assert "Sarah\nEduSaaS Academic Admissions & Guidance" in email.body
    # Sarah is NEVER greeted
    assert "Hi Sarah," not in email.body
    # Alice is NEVER signed as agent
    assert "Alice\nEduSaaS" not in email.body


# =============================================================================
# TEST 12: No Agent Name Configured (Safe Organization Fallback)
# =============================================================================

def test_scenario_12_no_agent_name_configured():
    """TEST 12: No agent name configured -> organization signature, no invented name."""
    email = generate_edusaas_email(
        direction=CallDirection.OUTBOUND,
        contact_name="Alice",
        agent_name=None,  # No agent name
        topic_requested="data_science",
        recipient="alice@example.com",
    )

    assert "Hi Alice," in email.body
    assert "Best regards,\nEduSaaS Academic Admissions & Guidance" in email.body
    # No invented representative names
    assert "Sarah" not in email.body
    assert "John" not in email.body


# =============================================================================
# TEST 13: send_email Fails (Honest Failure, No False Success)
# =============================================================================

@pytest.mark.asyncio
async def test_scenario_13_send_email_fails_no_false_success(base_settings, state_mgr):
    """TEST 13: send_email fails -> honest failure response, no false success claim."""
    failing_tool_provider = FailingMockToolProvider()
    engine = make_engine(base_settings, failing_tool_provider)
    state = state_mgr.create_inbound_state(
        call_id="call-in-13",
        caller_phone="+15558889900",
        caller_name="Karan",
        caller_email="karan@example.com",
    )
    state.lock_domain(DomainType.EDUSAAS)

    res = await engine.process_user_turn(state, "Can you send me the course details?")

    # Must NOT claim email was sent!
    assert "i've sent" not in res.response_text.lower()
    assert "i have sent" not in res.response_text.lower()
    # Reports honest failure / issue
    assert "couldn't send" in res.response_text.lower() or "system issue" in res.response_text.lower()
    # Pending action reset
    assert state.pending_action is None


# =============================================================================
# TEST 14: No Email Requested (Verbal Answer Only)
# =============================================================================

@pytest.mark.asyncio
async def test_scenario_14_no_email_requested_verbal_only(base_settings, mock_tool_provider, state_mgr):
    """TEST 14: No email requested -> answer verbally, no automatic email sent."""
    engine = make_engine(base_settings, mock_tool_provider)
    state = state_mgr.create_inbound_state(
        call_id="call-in-14",
        caller_phone="+15559990011",
        caller_name="Meera",
        caller_email="meera@example.com",
    )
    state.lock_domain(DomainType.EDUSAAS)

    # Caller asks a verbal question without requesting email
    res = await engine.process_user_turn(state, "Can you tell me about the Data Science course?")

    # NO email action executed
    assert len(mock_tool_provider.dispatched_requests) == 0
    # Answers verbally
    assert len(res.response_text) > 10
    assert "i've sent" not in res.response_text.lower()


# =============================================================================
# TEST 15: Caller Requests Vague Email (Asks What Information to Send)
# =============================================================================

@pytest.mark.asyncio
async def test_scenario_15_vague_email_request_asks_clarification(base_settings, mock_tool_provider, state_mgr):
    """TEST 15: Caller requests vague email -> asks what information to send."""
    engine = make_engine(base_settings, mock_tool_provider)
    state = state_mgr.create_inbound_state(
        call_id="call-in-15",
        caller_phone="+15550001122",
        caller_name="Rahul",
        caller_email="rahul@example.com",
    )
    state.lock_domain(DomainType.EDUSAAS)

    # Caller asks vaguely: "Can you email me the information?"
    res = await engine.process_user_turn(state, "Can you email me the information?")

    # Must NOT generate generic email immediately
    assert len(mock_tool_provider.dispatched_requests) == 0
    # Asks for clarification on topic
    assert "what information would you like me to send" in res.response_text.lower()
    assert state.pending_action == "send_email"

    # Turn 2: Caller provides specific topic
    res2 = await engine.process_user_turn(state, "The Data Science course details.")

    # Now email is generated specifically for Data Science
    assert len(mock_tool_provider.dispatched_requests) == 1
    req = mock_tool_provider.dispatched_requests[0]
    assert req.arguments.get("recipient") == "rahul@example.com"
    assert "Data Science" in req.arguments.get("subject", "")
    assert "Data Science" in res2.response_text or "course details" in res2.response_text


# =============================================================================
# TEST 16: Inbound Intent Changes While Domain Remains Locked
# =============================================================================

@pytest.mark.asyncio
async def test_scenario_16_inbound_intent_changes_domain_remains_locked(base_settings, mock_tool_provider, state_mgr):
    """TEST 16: Inbound intent can change dynamically while domain remains locked to EduSaaS."""
    engine = make_engine(base_settings, mock_tool_provider)
    state = state_mgr.create_inbound_state(
        call_id="call-in-16",
        caller_phone="+15551239876",
    )

    # Turn 1: Fresh inbound identifies and locks EduSaaS
    res1 = await engine.process_user_turn(state, "I want information about your courses.")
    assert state.current_domain == DomainType.EDUSAAS
    assert state.domain_locked is True

    # Turn 2: Caller asks pricing -> intent changes to pricing, domain remains EduSaaS
    res2 = await engine.process_user_turn(state, "What are the fees?")
    assert state.current_domain == DomainType.EDUSAAS
    assert state.domain_locked is True

    # Turn 3: Caller asks to schedule a consultation -> intent changes to schedule, domain remains EduSaaS
    res3 = await engine.process_user_turn(state, "Can we schedule a consultation for tomorrow at 2 PM?")
    assert state.current_domain == DomainType.EDUSAAS
    assert state.domain_locked is True
    assert len(mock_tool_provider.dispatched_requests) == 1
    assert mock_tool_provider.dispatched_requests[0].action_name == "create_calendar_event"
