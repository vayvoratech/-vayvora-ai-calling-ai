"""Comprehensive test suite for Inbound, Outbound, and SMTP Email behaviors.

Tests all 15 required scenarios:
INBOUND:
1. inbound session waits for caller
2. inbound caller greeting
3. inbound direct question
4. inbound conversation continues

OUTBOUND:
5. outbound session immediately generates agent opening
6. outbound uses caller context
7. outbound uses campaign objective when available
8. outbound consultative conversation
9. outbound latest intent overrides campaign objective
10. outbound remains active after initial greeting

EMAIL:
11. SMTP successful send
12. SMTP authentication failure
13. SMTP connection failure
14. missing recipient
15. send_email only reports success after verified SMTP success
"""

import smtplib
from unittest.mock import MagicMock, patch
import pytest

from src.config import Settings
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine, EngineTurnResult
from src.core.llm import MockLLMProvider
from src.core.types import (
    CallDirection,
    CallStatus,
    ConversationStage,
    DomainType,
    ToolCallRequest,
    TurnRole,
)
from src.domains.base import DomainConfig
from src.domains.registry import get_domain_registry
from src.rag.embeddings import MockEmbeddingProvider
from src.rag.retriever import GroundedKnowledgeProvider
from src.state.manager import ConversationStateManager
from src.state.models import CallerProfile, ConversationState
from src.telephony.adapter import TelephonyVoiceAdapter
from src.telephony.transport import MockTelephonyTransport
from src.tools.email_provider import (
    EmailSendResult,
    MockEmailProvider,
    SMTPEmailProvider,
)
from src.tools.mcp_client import MockToolProvider
from src.voice.adapters import (
    PipecatConversationAdapter,
    PipecatSTTAdapter,
    PipecatTTSAdapter,
    PipecatVADAdapter,
)
from src.audio.stt import MockSTTProvider
from src.audio.tts import MockTTSProvider
from src.audio.vad import MockVADProvider
from src.voice.context import TurnLifecycleState
from src.voice.pipeline import VoicePipeline
from src.voice.session import MockAudioInput, MockAudioOutput, VoiceSession


# =============================================================================
# Helper Fixtures
# =============================================================================

def build_engine(
    llm: MockLLMProvider = None,
    email_provider: MockEmailProvider = None,
    rag: GroundedKnowledgeProvider = None,
) -> tuple:
    """Helper to assemble a test conversation engine with controllable mock subsystems."""
    llm_prov = llm or MockLLMProvider()
    email_prov = email_provider or MockEmailProvider(force_success=True)
    tool_prov = MockToolProvider(email_provider=email_prov)
    rag_prov = rag or GroundedKnowledgeProvider(
        embedding_provider=MockEmbeddingProvider(dimension=64),
        in_memory=True,
    )
    engine = ConversationEngine(
        llm_provider=llm_prov,
        knowledge_provider=rag_prov,
        tool_provider=tool_prov,
    )
    return engine, tool_prov, email_prov, llm_prov


TupleEngine = tuple


# =============================================================================
# INBOUND CALL BEHAVIOR TESTS (Scenarios 1-4)
# =============================================================================

class TestInboundCallBehavior:
    """Validate reactive, non-intrusive inbound call behaviors."""

    def test_01_inbound_session_waits_for_caller(self):
        """1. Inbound session waits for caller: no unsolicited agent opening, state active, session listening."""
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state(
            call_id="call-inbound-01",
            caller_phone="+15551234567",
            domain=DomainType.EDUSAAS,
        )

        assert state.metadata.direction == CallDirection.INBOUND
        # Agent must NOT speak first: transcript history must be completely empty
        assert len(state.history) == 0
        assert state.conversation_active is True
        assert state.stage == ConversationStage.GREETING

        # Verify VoiceSession starts in LISTENING state waiting for caller audio
        engine, _, _, _ = build_engine()
        pipeline = VoicePipeline(
            session_id="session-in-01",
            vad_adapter=PipecatVADAdapter(MockVADProvider(sample_rate=16000)),
            stt_adapter=PipecatSTTAdapter(MockSTTProvider()),
            conversation_adapter=PipecatConversationAdapter(engine),
            tts_adapter=PipecatTTSAdapter(MockTTSProvider(sample_rate=16000)),
            audio_output=MockAudioOutput(),
        )
        session = VoiceSession(
            session_id="session-in-01",
            conversation_state=state,
            pipeline=pipeline,
        )
        session.start()

        assert session.is_active is True
        assert session.pipeline.state == TurnLifecycleState.LISTENING
        # History remains empty until caller produces audio
        assert len(session.conversation_state.history) == 0

    @pytest.mark.asyncio
    async def test_02_inbound_caller_greeting(self):
        """2. Inbound caller greeting: caller says 'Hello', agent responds naturally without sales pitch."""
        engine, _, _, llm = build_engine()
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state(
            call_id="call-inbound-02",
            caller_phone="+15551234567",
            domain=DomainType.EDUSAAS,
        )

        # Caller initiates with "Hello"
        result = await engine.process_user_turn(state, "Hello")

        # Natural, non-sales response
        assert result.response_text == "Hi, how can I help you today?"
        assert result.conversation_active is True
        assert result.termination_occurred is False
        assert len(state.history) == 2
        assert state.history[0].role == TurnRole.CALLER
        assert state.history[0].content == "Hello"
        assert state.history[1].role == TurnRole.AGENT
        assert state.history[1].content == "Hi, how can I help you today?"

    @pytest.mark.asyncio
    async def test_03_inbound_direct_question(self):
        """3. Inbound direct question: answers question directly without forced introduction."""
        engine, _, _, llm = build_engine()
        # Seed canned decision for a direct question
        llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.EDUSAAS,
                detected_intent="course_information",
                proposed_stage=ConversationStage.INFORMATION,
                user_facing_response="We offer comprehensive programs in AI Engineering, Data Science, and Machine Learning.",
                knowledge_required=False,
            )
        )

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state(
            call_id="call-inbound-03",
            caller_phone="+15551234567",
            domain=DomainType.EDUSAAS,
        )

        # Caller immediately asks a direct question
        result = await engine.process_user_turn(state, "What courses do you offer?")

        assert "AI Engineering" in result.response_text
        assert "Data Science" in result.response_text
        # No forced intro or outbound pitch
        assert "I'm reaching out" not in result.response_text
        assert result.conversation_active is True
        assert state.stage == ConversationStage.INFORMATION

    @pytest.mark.asyncio
    async def test_04_inbound_conversation_continues(self):
        """4. Inbound conversation continues: multi-turn dialogue advances seamlessly."""
        engine, _, _, llm = build_engine()
        # Turn 1: Greeting
        llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.EDUSAAS,
                detected_intent="greeting",
                proposed_stage=ConversationStage.GREETING,
                user_facing_response="Hi, how can I help you today?",
            )
        )
        # Turn 2: Question
        llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.EDUSAAS,
                detected_intent="scholarship_inquiry",
                proposed_stage=ConversationStage.INFORMATION,
                user_facing_response="Yes, we offer merit-based scholarships covering up to 40% of tuition.",
            )
        )

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state(
            call_id="call-inbound-04",
            caller_phone="+15551234567",
            domain=DomainType.EDUSAAS,
        )

        # Turn 1
        t1 = await engine.process_user_turn(state, "Hello")
        assert t1.response_text == "Hi, how can I help you today?"
        assert state.conversation_active is True

        # Turn 2
        t2 = await engine.process_user_turn(state, "Do you have any scholarships available?")
        assert "scholarships" in t2.response_text.lower()
        assert state.conversation_active is True
        assert not state.termination_requested
        assert len(state.history) == 4


# =============================================================================
# OUTBOUND CALL BEHAVIOR TESTS (Scenarios 5-10)
# =============================================================================

class TestOutboundCallBehavior:
    """Validate proactive, consultative outbound call behaviors."""

    @pytest.mark.asyncio
    async def test_05_outbound_session_immediately_generates_agent_opening(self):
        """5. Outbound session immediately generates agent opening without waiting for caller."""
        engine, _, _, _ = build_engine()
        mgr = ConversationStateManager()
        state = mgr.create_outbound_state(
            call_id="call-outbound-05",
            caller_phone="+15559876543",
            domain=DomainType.EDUSAAS,
            caller_name="Rahul",
            campaign_id="CAMP-EDUSAAS-01",
            campaign_objective="Follow up with student who showed interest in our courses",
        )

        # Session starts -> agent proactively generates opening
        opening = await engine.start_outbound_conversation(state)

        assert opening.response_text is not None
        assert len(opening.response_text) > 0
        assert opening.conversation_active is True
        assert opening.termination_occurred is False

        # Agent opening must be recorded as first turn in transcript
        assert len(state.history) == 1
        assert state.history[0].role == TurnRole.AGENT
        assert state.history[0].content == opening.response_text
        assert state.conversation_active is True
        assert state.stage == ConversationStage.GREETING

    @pytest.mark.asyncio
    async def test_06_outbound_uses_caller_context(self):
        """6. Outbound uses caller context: name, company, domain without fabricating details."""
        engine, _, _, _ = build_engine()
        mgr = ConversationStateManager()
        state = mgr.create_outbound_state(
            call_id="call-outbound-06",
            caller_phone="+15559876543",
            domain=DomainType.VAYVORA,
            caller_name="Rahul",
            caller_email="rahul@enterprise.com",
            company="Enterprise Corp",
            campaign_id="CAMP-AI-01",
            campaign_objective="Understand whether organization is exploring AI or automation solutions",
        )

        opening = await engine.start_outbound_conversation(state)

        # Uses caller name and company
        assert "Rahul" in opening.response_text
        assert "Vayvora" in opening.response_text
        assert "Enterprise Corp" in opening.response_text
        assert opening.conversation_active is True

    @pytest.mark.asyncio
    async def test_07_outbound_uses_campaign_objective_when_available(self):
        """7. Outbound uses campaign objective when available."""
        engine, _, _, _ = build_engine()
        mgr = ConversationStateManager()

        # EduSaaS example
        state_edu = mgr.create_outbound_state(
            call_id="call-outbound-07-edu",
            caller_phone="+15551112222",
            domain=DomainType.EDUSAAS,
            caller_name="Rahul",
            campaign_id="CAMP-EDU-01",
            campaign_objective="Follow up with student who showed interest in our courses",
        )
        opening_edu = await engine.start_outbound_conversation(state_edu)
        assert "Rahul" in opening_edu.response_text
        assert "EduSaaS" in opening_edu.response_text
        assert "interest in our courses" in opening_edu.response_text
        assert "good time to speak" in opening_edu.response_text

        # Vayvora example
        state_vay = mgr.create_outbound_state(
            call_id="call-outbound-07-vay",
            caller_phone="+15553334444",
            domain=DomainType.VAYVORA,
            caller_name="Rahul",
            campaign_id="CAMP-VAY-01",
            campaign_objective="Understand whether organization is exploring AI or automation solutions",
        )
        opening_vay = await engine.start_outbound_conversation(state_vay)
        assert "Rahul" in opening_vay.response_text
        assert "Vayvora" in opening_vay.response_text
        assert "AI or automation solutions" in opening_vay.response_text
        assert "quick conversation" in opening_vay.response_text

    @pytest.mark.asyncio
    async def test_08_outbound_consultative_conversation(self):
        """8. Outbound consultative conversation: opening -> caller accepts -> discover -> recommend -> handle objection."""
        engine, _, _, llm = build_engine()
        mgr = ConversationStateManager()
        state = mgr.create_outbound_state(
            call_id="call-outbound-08",
            caller_phone="+15555556666",
            domain=DomainType.EDUSAAS,
            caller_name="Rahul",
            campaign_id="CAMP-EDU-08",
            campaign_objective="Follow up with student who showed interest in our courses",
        )

        # 1. Opening
        opening = await engine.start_outbound_conversation(state)
        assert "Rahul" in opening.response_text
        assert state.conversation_active is True

        # 2. Caller confirms time & asks about program
        llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.EDUSAAS,
                detected_intent="curriculum_discovery",
                proposed_stage=ConversationStage.DISCOVERY,
                user_facing_response="Our flagship program is the AI Engineering Masterclass, covering LLMs and vector search.",
            )
        )
        t1 = await engine.process_user_turn(state, "Yes, I have a couple of minutes. What does the AI program cover?")
        assert state.stage == ConversationStage.DISCOVERY
        assert state.conversation_active is True

        # 3. Caller shares background & asks for recommendation
        llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.EDUSAAS,
                detected_intent="recommendation",
                proposed_stage=ConversationStage.RECOMMENDATION,
                user_facing_response="Given your Python background, the AI Agent Engineering track would be the ideal fit.",
            )
        )
        t2 = await engine.process_user_turn(state, "I have 2 years of Python experience. Which course do you recommend?")
        assert state.stage == ConversationStage.RECOMMENDATION
        assert state.conversation_active is True

        # 4. Caller raises objection (busy schedule)
        llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.EDUSAAS,
                detected_intent="objection_schedule",
                proposed_stage=ConversationStage.OBJECTION_HANDLING,
                user_facing_response="All our classes are recorded with flexible weekend cohorts designed for working professionals.",
            )
        )
        t3 = await engine.process_user_turn(state, "I work full time, so I'm worried I won't have enough time.")
        assert state.stage == ConversationStage.OBJECTION_HANDLING
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_09_outbound_latest_intent_overrides_campaign_objective(self):
        """9. Outbound latest intent overrides campaign objective: campaign was course follow-up, caller asks for corporate AI."""
        engine, _, _, llm = build_engine()
        mgr = ConversationStateManager()
        state = mgr.create_outbound_state(
            call_id="call-outbound-09",
            caller_phone="+15557778888",
            domain=DomainType.EDUSAAS,
            caller_name="Rahul",
            campaign_id="CAMP-EDU-STUDENT",
            campaign_objective="Follow up with student about AI course",
        )

        # Agent opening is about EduSaaS AI course
        opening = await engine.start_outbound_conversation(state)
        assert "EduSaaS" in opening.response_text

        # Caller completely pivots to corporate AI solutions (Vayvora domain)
        llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.VAYVORA,
                detected_intent="solutions_inquiry",
                proposed_stage=ConversationStage.INFORMATION,
                knowledge_required=False,
                user_facing_response="At Vayvora, we provide corporate AI solutions, custom agent architectures, and enterprise automation.",
            )
        )

        turn = await engine.process_user_turn(state, "Actually, I want to know if you provide corporate AI solutions.")

        # Intent preemption rule: domain switched to Vayvora, intent switched, campaign objective did not override
        assert state.current_domain == DomainType.VAYVORA
        assert state.current_intent == "solutions_inquiry"
        assert "corporate AI solutions" in turn.response_text or "Vayvora" in turn.response_text
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_10_outbound_remains_active_after_initial_greeting(self):
        """10. Outbound remains active after initial greeting and non-committal caller responses."""
        engine, _, _, llm = build_engine()
        mgr = ConversationStateManager()
        state = mgr.create_outbound_state(
            call_id="call-outbound-10",
            caller_phone="+15559991111",
            domain=DomainType.EDUSAAS,
            caller_name="Rahul",
            campaign_id="CAMP-EDU-10",
            campaign_objective="Follow up with student who showed interest in our courses",
        )

        opening = await engine.start_outbound_conversation(state)
        assert state.conversation_active is True
        assert state.stage == ConversationStage.GREETING

        # Caller says "No" or hesitant response — call must remain active
        llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.EDUSAAS,
                detected_intent="not_convenient_time",
                proposed_stage=ConversationStage.INFORMATION,
                user_facing_response="No problem at all! Would it be better if I sent you the information by email?",
            )
        )
        turn = await engine.process_user_turn(state, "No, not really a great time.")
        assert state.conversation_active is True
        assert not state.termination_requested


# =============================================================================
# SMTP EMAIL TESTS (Scenarios 11-15)
# =============================================================================

class TestSMTPEmailBehavior:
    """Validate real SMTP email dispatch, authentication/connection failure handling, and guards."""

    @pytest.mark.asyncio
    async def test_11_smtp_successful_send(self):
        """11. SMTP successful send: verifies real smtplib execution, message_id, and TLS."""
        settings = Settings(
            SMTP_HOST="smtp.testserver.com",
            SMTP_PORT=587,
            SMTP_USERNAME="testuser",
            SMTP_PASSWORD="testpassword",
            SMTP_FROM_EMAIL="noreply@vayvora.com",
            SMTP_USE_TLS=True,
        )
        provider = SMTPEmailProvider(settings=settings)

        mock_smtp = MagicMock()
        mock_smtp.__enter__.return_value = mock_smtp

        with patch("smtplib.SMTP", return_value=mock_smtp) as mock_cls:
            result = await provider.send_email(
                recipient="client@example.com",
                subject="Test Subject",
                body="Test plain body",
            )

            assert result.success is True
            assert result.status == "sent"
            assert result.message_id is not None
            assert "@" in result.message_id
            mock_cls.assert_called_once_with("smtp.testserver.com", 587, timeout=10.0)
            mock_smtp.ehlo.assert_called()
            mock_smtp.starttls.assert_called_once()
            mock_smtp.login.assert_called_once_with("testuser", "testpassword")
            mock_smtp.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_12_smtp_authentication_failure(self):
        """12. SMTP authentication failure: gracefully captures auth error and reports failure."""
        settings = Settings(
            SMTP_HOST="smtp.testserver.com",
            SMTP_PORT=587,
            SMTP_USERNAME="invaliduser",
            SMTP_PASSWORD="wrongpassword",
            SMTP_FROM_EMAIL="noreply@vayvora.com",
        )
        provider = SMTPEmailProvider(settings=settings)

        mock_smtp = MagicMock()
        mock_smtp.__enter__.return_value = mock_smtp
        mock_smtp.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Authentication credentials invalid")

        with patch("smtplib.SMTP", return_value=mock_smtp):
            result = await provider.send_email(
                recipient="client@example.com",
                subject="Test Subject",
                body="Test body",
            )

            assert result.success is False
            assert result.status == "auth_failed"
            assert "authentication" in result.error.lower()

    @pytest.mark.asyncio
    async def test_13_smtp_connection_failure(self):
        """13. SMTP connection failure: host unreachable or connection refused handled gracefully."""
        settings = Settings(
            SMTP_HOST="unreachable.host.example",
            SMTP_PORT=587,
        )
        provider = SMTPEmailProvider(settings=settings)

        with patch("smtplib.SMTP", side_effect=ConnectionRefusedError("Connection refused by server")):
            result = await provider.send_email(
                recipient="client@example.com",
                subject="Test",
                body="Body",
            )

            assert result.success is False
            assert result.status == "connection_failed"
            assert "connection failure" in result.error.lower()

    @pytest.mark.asyncio
    async def test_14_missing_recipient_asks_caller_and_avoids_dispatch(self):
        """14. Missing recipient: agent asks caller for email before attempting to send."""
        engine, tool_prov, email_prov, llm = build_engine()
        # LLM proposes sending email without recipient in args and caller has no email
        llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.EDUSAAS,
                detected_intent="request_brochure",
                proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                action_proposed=True,
                proposed_action=ProposedAction(
                    tool_name="send_email",
                    arguments={"subject": "EduSaaS AI Course Brochure"},
                ),
                user_facing_response="I will send the brochure.",
            )
        )

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state(
            call_id="call-email-missing",
            caller_phone="+15551234567",
            caller_email=None,  # Unknown email
            domain=DomainType.EDUSAAS,
        )

        turn = await engine.process_user_turn(state, "Can you email me the brochure?")

        # Guard: Agent must ask for email address before sending
        assert "email address" in turn.response_text.lower()
        assert state.pending_question == "Could you please share your email address?"
        # Tool was not executed
        assert state.last_action != "send_email"
        assert state.last_tool_result is None
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_15_send_email_only_reports_success_after_verified_smtp_success(self):
        """15. send_email only reports success after verified SMTP success, never claiming false success on failure."""
        # --- Scenario A: SMTP Fails ---
        mock_failing_email = MockEmailProvider(force_auth_failure=True)
        engine_fail, _, _, llm_fail = build_engine(email_provider=mock_failing_email)

        llm_fail.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.EDUSAAS,
                detected_intent="request_brochure",
                proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                action_proposed=True,
                proposed_action=ProposedAction(
                    tool_name="send_email",
                    arguments={"subject": "Curriculum", "recipient": "student@example.com"},
                ),
                user_facing_response="Sending email now.",
            )
        )

        mgr = ConversationStateManager()
        state_fail = mgr.create_inbound_state(
            call_id="call-email-fail",
            caller_phone="+15551234567",
            caller_email="student@example.com",
            domain=DomainType.EDUSAAS,
        )

        turn_fail = await engine_fail.process_user_turn(state_fail, "Please email me the brochure.")

        # On failure: NEVER say "I sent the email"
        assert "sent the email" not in turn_fail.response_text.lower()
        assert "system issue" in turn_fail.response_text.lower() or "attempted" in turn_fail.response_text.lower()
        assert state_fail.last_tool_result is not None
        assert state_fail.last_tool_result.success is False
        assert state_fail.conversation_active is True

        # --- Scenario B: SMTP Succeeds ---
        mock_success_email = MockEmailProvider(force_success=True)
        engine_succ, _, _, llm_succ = build_engine(email_provider=mock_success_email)

        llm_succ.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.EDUSAAS,
                detected_intent="request_brochure",
                proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                action_proposed=True,
                proposed_action=ProposedAction(
                    tool_name="send_email",
                    arguments={"subject": "Curriculum", "recipient": "student@example.com"},
                ),
                user_facing_response="Sending email now.",
            )
        )

        state_succ = mgr.create_inbound_state(
            call_id="call-email-succ",
            caller_phone="+15551234567",
            caller_email="student@example.com",
            domain=DomainType.EDUSAAS,
        )

        turn_succ = await engine_succ.process_user_turn(state_succ, "Please email me the brochure.")

        # On verified success: confirms sending
        assert "sent" in turn_succ.response_text.lower()
        assert "student@example.com" in turn_succ.response_text
        assert state_succ.last_action == "send_email"
        assert state_succ.last_tool_result.success is True
        assert state_succ.conversation_active is True
