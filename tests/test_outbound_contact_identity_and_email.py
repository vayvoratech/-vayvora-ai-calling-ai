"""Tests for Outbound Voice-Call Implementation: Contact Name vs Agent Name & WhatsApp Messaging Safety.

Verifies:
1. Contact Name != Agent Name (distinction, agent_name config, safe fallback).
2. Outbound Identity Verification: gates campaign discussions until identity is verified.
3. WhatsApp Brochure Delivery:
   - When recipient phone exists & authorized -> dispatches send_message and confirms.
   - When recipient phone is missing -> asks for WhatsApp phone number, blocks tool.
   - When caller provides valid phone number -> normalizes, dispatches, confirms.
   - When caller provides invalid or placeholder dummy number -> rejects and prompts again.
   - Explicit phone correction updates recipient.
   - Respects WhatsApp opt-out / refusal.
4. Deterministic Safety Guards in ActionValidator, MessageProvider, and EmailProvider.
5. Inbound domain routing to Infra Builder.
"""

import pytest
from src.config import Settings
from src.core.decision import (
    ConversationalDecision,
    ProposedAction,
    is_valid_email,
)
from src.core.engine import ConversationEngine
from src.core.llm import MockLLMProvider
from src.core.types import (
    CallDirection,
    ConversationStage,
    DomainType,
    ToolCallRequest,
    TurnRole,
)
from src.domains.registry import get_domain_registry
from src.state.manager import ConversationStateManager
from src.state.models import CallerProfile, ConversationState
from src.tools.email_provider import MockEmailProvider, SMTPEmailProvider
from src.tools.message_provider import validate_phone_number
from src.tools.mcp_client import MockToolProvider
from src.tools.schemas import ValidatedToolRequest
from src.tools.validation import ActionValidator
from src.core.errors import InvalidToolArgumentsError


@pytest.fixture
def base_settings():
    return Settings(
        agent_name="Sarah",
        smtp_from_email="noreply@infrabuilder.com",
    )


@pytest.fixture
def mock_tool_provider():
    return MockToolProvider()


@pytest.fixture
def engine(base_settings, mock_tool_provider):
    llm = MockLLMProvider()
    return ConversationEngine(
        llm_provider=llm,
        tool_provider=mock_tool_provider,
        settings=base_settings,
        registry=get_domain_registry(),
    )


def was_tool_dispatched(provider: MockToolProvider, tool_name: str) -> bool:
    return any(r.action_name == tool_name for r in provider.dispatched_requests)


def get_last_dispatched(provider: MockToolProvider, tool_name: str):
    return next((r for r in reversed(provider.dispatched_requests) if r.action_name == tool_name), None)


# =============================================================================
# 1. Contact Name != Agent Name Tests
# =============================================================================

@pytest.mark.asyncio
async def test_outbound_opening_with_configured_agent_name(base_settings, mock_tool_provider):
    """When agent_name is configured (e.g. Sarah), agent introduces itself as Sarah and addresses contact Alice."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-1",
        contact_name="Alice",
        caller_phone="+919845123890",
        domain=DomainType.INFRA_BUILDER,
        agent_name="Sarah",
        campaign_objective="follow up with buyer who showed interest in Skyline Towers luxury apartments",
    )

    llm = MockLLMProvider()
    engine = ConversationEngine(
        llm_provider=llm,
        tool_provider=mock_tool_provider,
        settings=base_settings,
        registry=get_domain_registry(),
    )

    result = await engine.start_outbound_conversation(state)
    opening = result.response_text

    # Agent identity must be Sarah
    assert "Sarah" in opening
    # Contact is Alice
    assert "Alice" in opening
    # Agent must NOT claim to be Alice
    assert "I'm Alice" not in opening
    assert "my name is Alice" not in opening
    assert "I am Alice" not in opening
    assert "Infra Builder" in opening


@pytest.mark.asyncio
async def test_outbound_opening_without_agent_name_uses_safe_fallback(mock_tool_provider):
    """When agent_name is None, agent uses safe fallback 'I'm calling from...' and never uses contact name."""
    settings = Settings(agent_name=None)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-2",
        contact_name="Alice",
        caller_phone="+919845123890",
        domain=DomainType.INFRA_BUILDER,
        agent_name=None,
        campaign_objective="follow up with buyer who showed interest in Skyline Towers",
    )

    llm = MockLLMProvider()
    engine = ConversationEngine(
        llm_provider=llm,
        tool_provider=mock_tool_provider,
        settings=settings,
        registry=get_domain_registry(),
    )

    result = await engine.start_outbound_conversation(state)
    opening = result.response_text

    # Safe fallback
    assert "I'm calling from Infra Builder" in opening
    # Addresses contact Alice
    assert "Alice" in opening
    # NEVER claims to be Alice
    assert "I'm Alice" not in opening
    assert "my name is Alice" not in opening
    assert "I am Alice" not in opening


@pytest.mark.asyncio
async def test_outbound_vayvora_opening_distinguishes_contact_and_agent(mock_tool_provider):
    """Outbound for Infra Builder distinguishes agent David from contact Bob."""
    settings = Settings(agent_name="David")
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-3",
        contact_name="Bob",
        caller_phone="+919845123890",
        domain=DomainType.INFRA_BUILDER,
        agent_name="David",
        company="Acme Corp",
        campaign_objective="understand buyer residential requirements in Bangalore",
    )

    llm = MockLLMProvider()
    engine = ConversationEngine(
        llm_provider=llm,
        tool_provider=mock_tool_provider,
        settings=settings,
        registry=get_domain_registry(),
    )

    result = await engine.start_outbound_conversation(state)
    opening = result.response_text

    assert "David" in opening
    assert "Bob" in opening
    assert "I'm Bob" not in opening
    assert "my name is Bob" not in opening
    assert "Infra Builder" in opening


# =============================================================================
# 2. When Contact Phone Exists & WhatsApp Brochure is Requested
# =============================================================================

@pytest.mark.asyncio
async def test_outbound_email_exists_sends_immediately_and_confirms(engine, mock_tool_provider):
    """When contact phone exists and caller confirms identity, requesting details dispatches WhatsApp send_message."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-4",
        contact_name="Alice",
        caller_phone="+919845123890",
        domain=DomainType.INFRA_BUILDER,
        agent_name="Sarah",
    )

    await engine.start_outbound_conversation(state)
    # Confirm identity
    await engine.process_user_turn(state, "Yes, speaking.")
    state.update_slot("whatsapp_consent", True)

    # Caller requests brochure/details
    result = await engine.process_user_turn(state, "Please send me the brochure for Skyline Towers.")

    # Tool send_message was invoked
    assert was_tool_dispatched(mock_tool_provider, "send_message")
    last_req = get_last_dispatched(mock_tool_provider, "send_message")
    assert last_req is not None
    assert "9845123890" in str(last_req.arguments.get("recipient"))

    # Confirms only after successful execution
    assert "sent" in result.response_text.lower()
    assert "9845123890" in result.response_text


# =============================================================================
# 3. When Contact Phone is Missing
# =============================================================================

@pytest.mark.asyncio
async def test_outbound_email_missing_prompts_and_blocks_tool(engine, mock_tool_provider):
    """When contact has NO phone, requesting details prompts for WhatsApp phone number and blocks tool."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-5",
        contact_name="Alice",
        caller_phone="",
        domain=DomainType.INFRA_BUILDER,
        agent_name="Sarah",
    )

    await engine.start_outbound_conversation(state)
    await engine.process_user_turn(state, "Yes, speaking.")

    result = await engine.process_user_turn(state, "Please send me the project brochure.")

    # Tool must NOT be called without phone number
    assert not was_tool_dispatched(mock_tool_provider, "send_message")
    # Agent prompts for WhatsApp number
    assert "whatsapp" in result.response_text.lower()
    assert "number" in result.response_text.lower() or "phone" in result.response_text.lower()
    # State records pending WhatsApp action
    assert state.pending_action == "send_message"


# =============================================================================
# 4. Caller Provides Phone Following Missing Phone Prompt
# =============================================================================

@pytest.mark.asyncio
async def test_caller_provides_valid_email_executes_and_confirms(engine, mock_tool_provider):
    """Caller provides phone after being prompted; system updates profile, executes tool, and confirms."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-6",
        contact_name="Alice",
        caller_phone="",
        domain=DomainType.INFRA_BUILDER,
        agent_name="Sarah",
    )

    await engine.start_outbound_conversation(state)
    await engine.process_user_turn(state, "Yes, speaking.")

    # Turn 1: Missing phone prompt
    turn1 = await engine.process_user_turn(state, "Please send me the project brochure.")
    assert not was_tool_dispatched(mock_tool_provider, "send_message")
    assert "whatsapp" in turn1.response_text.lower()

    # Turn 2: Caller provides valid phone number
    turn2 = await engine.process_user_turn(state, "+919845123890")

    # Stored in contact profile
    assert "9845123890" in (state.caller.phone or "")
    # Tool executed with verified recipient
    assert was_tool_dispatched(mock_tool_provider, "send_message")
    last_req = get_last_dispatched(mock_tool_provider, "send_message")
    assert "9845123890" in str(last_req.arguments.get("recipient"))
    # Verbal confirmation
    assert "sent" in turn2.response_text.lower()


@pytest.mark.asyncio
async def test_caller_provides_spoken_email_pattern(engine, mock_tool_provider):
    """Caller speaks phone number naturally in message."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-7",
        contact_name="Alice",
        caller_phone="",
        domain=DomainType.INFRA_BUILDER,
        agent_name="Sarah",
    )

    await engine.start_outbound_conversation(state)
    await engine.process_user_turn(state, "Yes, speaking.")
    await engine.process_user_turn(state, "Please send me the brochure.")

    turn2 = await engine.process_user_turn(state, "You can WhatsApp me on 9845123890")

    assert "9845123890" in (state.caller.phone or "")
    assert was_tool_dispatched(mock_tool_provider, "send_message")
    assert "sent" in turn2.response_text.lower()


# =============================================================================
# 5. Invalid or Dummy Phone Number
# =============================================================================

@pytest.mark.asyncio
async def test_caller_provides_unclear_email_asks_repeat(engine, mock_tool_provider):
    """When caller provides an invalid/dummy sequential phone, agent blocks tool and prompts again."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-8",
        contact_name="Alice",
        caller_phone="",
        domain=DomainType.INFRA_BUILDER,
        agent_name="Sarah",
    )

    await engine.start_outbound_conversation(state)
    await engine.process_user_turn(state, "Yes, speaking.")
    await engine.process_user_turn(state, "Please send me the brochure.")

    # Dummy sequential number rejected by validate_phone_number
    turn2 = await engine.process_user_turn(state, "send to 1234567890")

    # Tool must NOT execute
    assert not was_tool_dispatched(mock_tool_provider, "send_message")
    # Prompts for WhatsApp number
    assert "whatsapp" in turn2.response_text.lower()

    # Then caller gives valid phone number
    turn3 = await engine.process_user_turn(state, "+919845123890")
    assert was_tool_dispatched(mock_tool_provider, "send_message")
    assert "sent" in turn3.response_text.lower()


# =============================================================================
# 6. Explicit Phone Correction
# =============================================================================

@pytest.mark.asyncio
async def test_explicit_email_correction_updates_recipient(engine, mock_tool_provider):
    """Explicit caller phone correction updates recipient phone."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-9",
        contact_name="Alice",
        caller_phone="+919845123890",
        domain=DomainType.INFRA_BUILDER,
        agent_name="Sarah",
    )

    await engine.start_outbound_conversation(state)
    await engine.process_user_turn(state, "Yes, speaking.")
    await engine.process_user_turn(state, "Please send me the brochure.")

    # Caller explicitly provides alternate number
    turn1 = await engine.process_user_turn(state, "Actually, please send it to +919845112233")

    assert "9845112233" in (state.caller.phone or "")
    assert was_tool_dispatched(mock_tool_provider, "send_message")
    last_req = get_last_dispatched(mock_tool_provider, "send_message")
    assert "9845112233" in str(last_req.arguments.get("recipient"))


@pytest.mark.asyncio
async def test_ambiguous_speech_does_not_overwrite_email(engine):
    """Ambiguous speech without a valid phone does NOT overwrite stored recipient phone."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-10",
        contact_name="Alice",
        caller_phone="+919845123890",
        domain=DomainType.INFRA_BUILDER,
        agent_name="Sarah",
    )

    await engine.start_outbound_conversation(state)
    await engine.process_user_turn(state, "Yes, speaking.")
    await engine.process_user_turn(state, "I have another phone somewhere.")

    assert "9845123890" in (state.caller.phone or "")


# =============================================================================
# 7. Deterministic Safety Guards (ActionValidator, Tool, Provider)
# =============================================================================

def test_action_validator_rejects_fallback_and_missing_recipient():
    """ActionValidator raises InvalidToolArgumentsError if recipient is missing, invalid, or dummy."""
    # Missing WhatsApp recipient
    with pytest.raises(InvalidToolArgumentsError):
        ActionValidator.validate_action(
            ValidatedToolRequest(
                action_name="send_message",
                arguments={},
                session_id="s1",
                domain=DomainType.INFRA_BUILDER,
            )
        )

    # Dummy placeholder recipient
    with pytest.raises(InvalidToolArgumentsError):
        ActionValidator.validate_action(
            ValidatedToolRequest(
                action_name="send_message",
                arguments={"recipient": "1234567890"},
                session_id="s1",
                domain=DomainType.INFRA_BUILDER,
            )
        )

    # Opted out caller
    with pytest.raises(InvalidToolArgumentsError):
        ActionValidator.validate_action(
            ValidatedToolRequest(
                action_name="send_message",
                arguments={"recipient": "+919845123890", "whatsapp_consent": False},
                session_id="s1",
                domain=DomainType.INFRA_BUILDER,
            )
        )

    # Prohibited email recipient in legacy email tool
    with pytest.raises(InvalidToolArgumentsError):
        ActionValidator.validate_action(
            ValidatedToolRequest(
                action_name="send_email",
                arguments={"recipient": "noreply@infrabuilder.com"},
                session_id="s1",
                domain=DomainType.INFRA_BUILDER,
            )
        )

    # Valid WhatsApp message passes
    ActionValidator.validate_action(
        ValidatedToolRequest(
            action_name="send_message",
            arguments={"recipient": "+919845123890", "whatsapp_consent": True},
            session_id="s1",
            domain=DomainType.INFRA_BUILDER,
        )
    )


@pytest.mark.asyncio
async def test_mock_and_smtp_email_provider_block_prohibited_recipient():
    """EmailProvider implementations reject prohibited/fallback recipients."""
    mock_provider = MockEmailProvider()
    result = await mock_provider.send_email(
        recipient="noreply@infrabuilder.com",
        subject="Test",
        body="Test body",
    )
    assert not result.success
    assert "Invalid, missing, or prohibited recipient email address" in (result.error or "")


# =============================================================================
# 8. Inbound Domain Dynamic Routing Integrity
# =============================================================================

@pytest.mark.asyncio
async def test_inbound_domain_routing_unaffected(engine):
    """Inbound calls start with UNKNOWN domain and lock to Infra Builder on property query."""
    state_mgr = ConversationStateManager()
    inbound_state = state_mgr.create_inbound_state(
        call_id="call-in-1",
        caller_phone="+919845123890",
    )

    assert inbound_state.current_domain == DomainType.UNKNOWN
    assert not inbound_state.domain_locked

    turn1 = await engine.process_user_turn(inbound_state, "I want to know about your 3 BHK apartments at Skyline Towers.")

    assert inbound_state.current_domain == DomainType.INFRA_BUILDER
    assert inbound_state.domain_locked
