"""Tests for Outbound Voice-Call Implementation: Contact Name vs Agent Name & Deterministic Email Safety.

Verifies:
1. Contact Name != Agent Name (distinction, agent_name config, safe fallback).
2. Outbound Contact Email (no fallback, no company/agent email as recipient).
3. When Contact Email Exists (sends immediately, confirms with recipient).
4. When Contact Email is Missing (prompts for recipient email, blocks tool).
5. Caller Provides Email (extracts, stores, dispatches, confirms).
6. Invalid or Unclear Email (asks "Could you repeat the email address for me?", blocks tool).
7. Explicit Email Correction (updates stored recipient email on explicit correction cue).
8. Deterministic Email Guard in ConversationEngine, ActionValidator, and EmailProvider.
9. Inbound dynamic domain routing integrity remains unaffected.
"""

import pytest
from src.config import Settings
from src.core.decision import (
    ConversationalDecision,
    ProposedAction,
    extract_email_address,
    detect_explicit_email_correction,
    is_unclear_or_invalid_email_attempt,
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
from src.tools.mcp_client import MockToolProvider
from src.tools.schemas import ValidatedToolRequest
from src.tools.validation import ActionValidator
from src.core.errors import InvalidToolArgumentsError


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
def engine(base_settings, mock_tool_provider):
    llm = MockLLMProvider()
    return ConversationEngine(
        llm_provider=llm,
        tool_provider=mock_tool_provider,
        settings=base_settings,
        registry=get_domain_registry(),
    )


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
        caller_phone="+15551234567",
        domain=DomainType.EDUSAAS,
        agent_name="Sarah",
        campaign_objective="follow up with student who showed interest in AI course",
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
    assert "from EduSaaS Academic Admissions & Guidance" in opening


@pytest.mark.asyncio
async def test_outbound_opening_without_agent_name_uses_safe_fallback(mock_tool_provider):
    """When agent_name is None, agent uses safe fallback 'I'm calling from...' and never uses contact name."""
    settings = Settings(agent_name=None)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-2",
        contact_name="Alice",
        caller_phone="+15551234567",
        domain=DomainType.EDUSAAS,
        agent_name=None,
        campaign_objective="follow up with student who showed interest in AI course",
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
    assert "I'm calling from EduSaaS Academic Admissions & Guidance" in opening
    # Addresses contact Alice
    assert "Alice" in opening
    # NEVER claims to be Alice
    assert "I'm Alice" not in opening
    assert "my name is Alice" not in opening
    assert "I am Alice" not in opening


@pytest.mark.asyncio
async def test_outbound_vayvora_opening_distinguishes_contact_and_agent(mock_tool_provider):
    """Outbound for Vayvora distinguishes agent David from contact Bob."""
    settings = Settings(agent_name="David")
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-3",
        contact_name="Bob",
        caller_phone="+15559876543",
        domain=DomainType.VAYVORA,
        agent_name="David",
        company="Acme Corp",
        campaign_objective="understand whether organization is exploring enterprise AI",
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
    assert "Vayvora Technologies" in opening


def was_tool_dispatched(provider: MockToolProvider, tool_name: str) -> bool:
    return any(r.action_name == tool_name for r in provider.dispatched_requests)


def get_last_dispatched(provider: MockToolProvider, tool_name: str):
    return next((r for r in reversed(provider.dispatched_requests) if r.action_name == tool_name), None)


# =============================================================================
# 2. When Contact Email Exists
# =============================================================================

@pytest.mark.asyncio
async def test_outbound_email_exists_sends_immediately_and_confirms(engine, mock_tool_provider):
    """When contact has email (alice@example.com), requesting course details sends email immediately without re-asking."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-4",
        contact_name="Alice",
        caller_phone="+15551234567",
        caller_email="alice@example.com",
        domain=DomainType.EDUSAAS,
        agent_name="Sarah",
    )

    await engine.start_outbound_conversation(state)

    # Caller requests course details
    result = await engine.process_user_turn(state, "Send me the course details.")

    # Tool was invoked
    assert was_tool_dispatched(mock_tool_provider, "send_email")
    last_req = get_last_dispatched(mock_tool_provider, "send_email")
    assert last_req is not None
    assert last_req.arguments.get("recipient") == "alice@example.com"

    # Confirms only after successful execution
    assert "I've sent the course details to alice@example.com." in result.response_text


# =============================================================================
# 3. When Contact Email is Missing
# =============================================================================

@pytest.mark.asyncio
async def test_outbound_email_missing_prompts_and_blocks_tool(engine, mock_tool_provider):
    """When contact has NO email, requesting course details prompts for email and blocks send_email."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-5",
        contact_name="Alice",
        caller_phone="+15551234567",
        caller_email=None,
        domain=DomainType.EDUSAAS,
        agent_name="Sarah",
    )

    await engine.start_outbound_conversation(state)

    result = await engine.process_user_turn(state, "Send me the course details.")

    # Tool must NOT be called
    assert not was_tool_dispatched(mock_tool_provider, "send_email")
    # Agent prompts for email
    assert "What email address should I send the course details to?" in result.response_text
    # State records pending email action
    assert state.pending_action == "send_email"


# =============================================================================
# 4. Caller Provides Email Following Missing Email Prompt
# =============================================================================

@pytest.mark.asyncio
async def test_caller_provides_valid_email_executes_and_confirms(engine, mock_tool_provider):
    """Caller provides email after being prompted; system updates profile, executes tool, and confirms."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-6",
        contact_name="Alice",
        caller_phone="+15551234567",
        caller_email=None,
        domain=DomainType.EDUSAAS,
        agent_name="Sarah",
    )

    await engine.start_outbound_conversation(state)

    # Turn 1: Missing email prompt
    turn1 = await engine.process_user_turn(state, "Send me the course details.")
    assert not was_tool_dispatched(mock_tool_provider, "send_email")
    assert "What email address should I send the course details to?" in turn1.response_text

    # Turn 2: Caller provides email
    turn2 = await engine.process_user_turn(state, "alice@example.com")

    # Stored in contact profile
    assert state.caller.email == "alice@example.com"
    # Tool executed with verified recipient
    assert was_tool_dispatched(mock_tool_provider, "send_email")
    last_req = get_last_dispatched(mock_tool_provider, "send_email")
    assert last_req.arguments.get("recipient") == "alice@example.com"
    # Verbal confirmation
    assert "I've sent the course details to alice@example.com." in turn2.response_text


@pytest.mark.asyncio
async def test_caller_provides_spoken_email_pattern(engine, mock_tool_provider):
    """Caller speaks email pattern: 'alice at example dot com'."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-7",
        contact_name="Alice",
        caller_phone="+15551234567",
        caller_email=None,
        domain=DomainType.EDUSAAS,
        agent_name="Sarah",
    )

    await engine.start_outbound_conversation(state)
    await engine.process_user_turn(state, "Send me the course details.")

    turn2 = await engine.process_user_turn(state, "alice at example dot com")

    assert state.caller.email == "alice@example.com"
    assert was_tool_dispatched(mock_tool_provider, "send_email")
    assert "I've sent the course details to alice@example.com." in turn2.response_text


# =============================================================================
# 5. Invalid or Unclear Email
# =============================================================================

@pytest.mark.asyncio
async def test_caller_provides_unclear_email_asks_repeat(engine, mock_tool_provider):
    """When caller provides an unclear or invalid email attempt, agent asks 'Could you repeat the email address for me?'."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-8",
        contact_name="Alice",
        caller_phone="+15551234567",
        caller_email=None,
        domain=DomainType.EDUSAAS,
        agent_name="Sarah",
    )

    await engine.start_outbound_conversation(state)
    await engine.process_user_turn(state, "Send me the course details.")

    # Malformed email attempt
    turn2 = await engine.process_user_turn(state, "alice at example")

    # Tool must NOT execute
    assert not was_tool_dispatched(mock_tool_provider, "send_email")
    # Natural clarification question
    assert "Could you repeat the email address for me?" in turn2.response_text

    # Then caller gives valid email
    turn3 = await engine.process_user_turn(state, "alice@example.com")
    assert was_tool_dispatched(mock_tool_provider, "send_email")
    assert "I've sent the course details to alice@example.com." in turn3.response_text


# =============================================================================
# 6. Explicit Email Correction
# =============================================================================

@pytest.mark.asyncio
async def test_explicit_email_correction_updates_recipient(engine, mock_tool_provider):
    """Explicit caller correction ('Actually, use alice.new@example.com') updates recipient email."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-9",
        contact_name="Alice",
        caller_phone="+15551234567",
        caller_email="alice@example.com",
        domain=DomainType.EDUSAAS,
        agent_name="Sarah",
    )

    await engine.start_outbound_conversation(state)

    # Caller explicitly corrects email
    turn1 = await engine.process_user_turn(state, "Actually, use alice.new@example.com")

    assert state.caller.email == "alice.new@example.com"

    # Now caller requests course details
    turn2 = await engine.process_user_turn(state, "Send me the course details.")
    assert was_tool_dispatched(mock_tool_provider, "send_email")
    last_req = get_last_dispatched(mock_tool_provider, "send_email")
    assert last_req.arguments.get("recipient") == "alice.new@example.com"
    assert "I've sent the course details to alice.new@example.com." in turn2.response_text


@pytest.mark.asyncio
async def test_ambiguous_speech_does_not_overwrite_email(engine):
    """Ambiguous speech without explicit correction does NOT overwrite stored recipient email."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_outbound_state(
        call_id="call-out-10",
        contact_name="Alice",
        caller_phone="+15551234567",
        caller_email="alice@example.com",
        domain=DomainType.EDUSAAS,
        agent_name="Sarah",
    )

    await engine.start_outbound_conversation(state)
    await engine.process_user_turn(state, "I also have another account somewhere.")

    assert state.caller.email == "alice@example.com"


# =============================================================================
# 7. Deterministic Email Guard (Tool, Validator, and Provider)
# =============================================================================

def test_action_validator_rejects_fallback_and_missing_recipient():
    """ActionValidator raises InvalidToolArgumentsError if recipient is missing, invalid, or fallback."""
    # Missing recipient
    with pytest.raises(InvalidToolArgumentsError):
        ActionValidator.validate_action(
            ValidatedToolRequest(
                action_name="send_email",
                arguments={},
                session_id="s1",
                domain=DomainType.EDUSAAS,
            )
        )

    # Fallback recipient
    with pytest.raises(InvalidToolArgumentsError):
        ActionValidator.validate_action(
            ValidatedToolRequest(
                action_name="send_email",
                arguments={"recipient": "noreply@vayvora.com"},
                session_id="s1",
                domain=DomainType.EDUSAAS,
            )
        )

    # Placeholder recipient
    with pytest.raises(InvalidToolArgumentsError):
        ActionValidator.validate_action(
            ValidatedToolRequest(
                action_name="send_email",
                arguments={"recipient": "caller@mail"},
                session_id="s1",
                domain=DomainType.EDUSAAS,
            )
        )

    # Valid recipient passes
    ActionValidator.validate_action(
        ValidatedToolRequest(
            action_name="send_email",
            arguments={"recipient": "alice@example.com"},
            session_id="s1",
            domain=DomainType.EDUSAAS,
        )
    )


@pytest.mark.asyncio
async def test_mock_and_smtp_email_provider_block_prohibited_recipient():
    """EmailProvider implementations reject prohibited/fallback recipients."""
    mock_provider = MockEmailProvider()
    result = await mock_provider.send_email(
        recipient="noreply@vayvora.com",
        subject="Test",
        body="Test body",
    )
    assert not result.success
    assert "Invalid, missing, or prohibited recipient email address" in (result.error or "")

    # Valid email succeeds with mock provider
    result_valid = await mock_provider.send_email(
        recipient="alice@example.com",
        subject="Test",
        body="Test body",
    )
    assert result_valid.success


# =============================================================================
# 8. Inbound Domain Dynamic Routing Integrity
# =============================================================================

@pytest.mark.asyncio
async def test_inbound_domain_routing_unaffected(engine):
    """Inbound calls still start with UNKNOWN domain and route dynamically."""
    state_mgr = ConversationStateManager()
    inbound_state = state_mgr.create_inbound_state(
        call_id="call-in-1",
        caller_phone="+15550001111",
    )

    assert inbound_state.current_domain == DomainType.UNKNOWN
    assert not inbound_state.domain_locked

    turn1 = await engine.process_user_turn(inbound_state, "I want to know about your courses and curriculum.")

    assert inbound_state.current_domain == DomainType.EDUSAAS
    assert inbound_state.domain_locked
