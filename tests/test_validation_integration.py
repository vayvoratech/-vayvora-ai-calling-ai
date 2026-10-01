"""Comprehensive Integration and End-to-End Golden Tests for Unified AI Voice Agent.

Validates all 25 specific integration test scenarios from Section 15 and all 6
End-to-End Golden Test conversations from Section 16.
"""

import asyncio
from typing import List, Optional
import pytest

from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine, EngineTurnResult
from src.core.errors import LLMMalformedResponseError, LLMTimeoutError
from src.core.llm import MockLLMProvider
from src.core.types import (
    CallDirection,
    ConversationStage,
    DomainType,
    RAGChunk,
    RAGQuery,
    ToolCallRequest,
    ToolExecutionResult,
)
from src.audio import MockSTTProvider, MockTTSProvider, MockVADProvider
from src.state.models import CallerProfile
from src.rag.embeddings import MockEmbeddingProvider
from src.rag.retriever import GroundedKnowledgeProvider
from src.state.manager import ConversationStateManager
from src.tools.mcp_client import MockToolProvider
from src.voice.adapters.llm import PipecatConversationAdapter
from src.voice.adapters.stt import PipecatSTTAdapter
from src.voice.adapters.tts import PipecatTTSAdapter
from src.voice.adapters.vad import PipecatVADAdapter
from src.voice.pipeline import VoicePipeline
from src.voice.session import MockAudioInput, MockAudioOutput, VoiceSession


# =============================================================================
# Helper Fixtures & Setup
# =============================================================================

@pytest.fixture
def populated_knowledge_provider():
    """Deterministic in-memory knowledge provider seeded with verified facts."""
    provider = GroundedKnowledgeProvider(
        embedding_provider=MockEmbeddingProvider(dimension=384),
        in_memory=True,
    )
    # EduSaaS documents
    asyncio.run(
        provider.index_document(
            RAGChunk(
                doc_id="edu:courses:ai_eng",
                domain=DomainType.EDUSAAS,
                title="AI Engineering Bootcamp",
                content="The AI Engineering Bootcamp duration is 12 weeks with weekend batches and live capstone mentoring.",
                score=1.0,
            )
        )
    )
    # Vayvora documents
    asyncio.run(
        provider.index_document(
            RAGChunk(
                doc_id="vay:solutions:enterprise_ai",
                domain=DomainType.VAYVORA,
                title="Vayvora Enterprise AI Solutions",
                content="Vayvora enterprise AI solutions include conversational voice agents, custom RAG pipelines, and automated agent workflows.",
                score=1.0,
            )
        )
    )
    asyncio.run(
        provider.index_document(
            RAGChunk(
                doc_id="vay:company:location",
                domain=DomainType.VAYVORA,
                title="Vayvora Headquarters",
                content="Vayvora headquarters and primary research engineering center is located in Bangalore, India.",
                score=1.0,
            )
        )
    )
    asyncio.run(
        provider.index_document(
            RAGChunk(
                doc_id="vay:careers:ai_roles",
                domain=DomainType.VAYVORA,
                title="AI Engineering Careers",
                content="Vayvora is hiring Senior AI Engineers with experience in PyTorch, distributed inference, and low-latency voice pipelines.",
                score=1.0,
            )
        )
    )
    return provider


# =============================================================================
# Section 15: 25 Integration Tests
# =============================================================================

class TestTwentyFiveIntegrationScenarios:
    """Rigorous verification of scenarios 1 through 25 from Section 15."""

    @pytest.mark.asyncio
    async def test_01_edusaas_inbound_course_question(self, populated_knowledge_provider):
        """1. EduSaaS inbound course question retrieves knowledge and answers accurately."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="course_information",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Our AI Engineering Bootcamp runs for 12 weeks with live capstone mentoring.",
            knowledge_required=True,
            knowledge_query="AI Engineering Bootcamp duration weeks",
        )
        mock_llm = MockLLMProvider(
            canned_decisions=[decision],
            canned_responses=["The AI Engineering Bootcamp is a 12-week program with weekend batches."],
        )
        engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=populated_knowledge_provider)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s01", "+15551234567", domain=DomainType.EDUSAAS)

        res = await engine.process_user_turn(state, "What is the duration of the AI Engineering Bootcamp?")
        assert res.decision.detected_domain == DomainType.EDUSAAS
        assert "edu:courses:ai_eng" in res.grounded_citations
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_02_edusaas_outbound_conversation(self):
        """2. EduSaaS outbound conversation preserves campaign objective and respects caller priority."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="course_recommendation",
            proposed_stage=ConversationStage.DISCOVERY,
            user_facing_response="Hi Priya, I understand you're interested in AI. Are you looking for weekend or weekday batches?",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_outbound_state(
            call_id="call-s02",
            caller_phone="+15559876543",
            domain=DomainType.EDUSAAS,
            caller_name="Priya",
            campaign_id="CAMP-EDU-01",
            campaign_objective="Course enrollment follow-up",
        )

        res = await engine.process_user_turn(state, "Hi, I have a few questions about your AI course.")
        assert state.caller.name == "Priya"
        assert state.caller.campaign == "CAMP-EDU-01"
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_03_vayvora_inbound_company_question(self, populated_knowledge_provider):
        """3. Vayvora inbound company question retrieves company details."""
        decision = ConversationalDecision(
            detected_domain=DomainType.VAYVORA,
            detected_intent="company_location",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Vayvora is headquartered in Bangalore, India.",
            knowledge_required=True,
            knowledge_query="Vayvora headquarters Bangalore India",
        )
        mock_llm = MockLLMProvider(
            canned_decisions=[decision],
            canned_responses=["Vayvora Technologies is headquartered in Bangalore, India."],
        )
        engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=populated_knowledge_provider)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s03", "+15551234567", domain=DomainType.VAYVORA)

        res = await engine.process_user_turn(state, "Where is Vayvora headquartered?")
        assert res.decision.detected_domain == DomainType.VAYVORA
        assert "vay:company:location" in res.grounded_citations

    @pytest.mark.asyncio
    async def test_04_vayvora_inbound_career_question(self, populated_knowledge_provider):
        """4. Vayvora inbound career question routes to careers without recommending student courses."""
        decision = ConversationalDecision(
            detected_domain=DomainType.VAYVORA,
            detected_intent="career_information",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="We are currently hiring Senior AI Engineers experienced in PyTorch and voice pipelines.",
            knowledge_required=True,
            knowledge_query="Vayvora AI Engineering Careers PyTorch",
        )
        mock_llm = MockLLMProvider(
            canned_decisions=[decision],
            canned_responses=["We have open roles for Senior AI Engineers working on voice pipelines."],
        )
        engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=populated_knowledge_provider)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s04", "+15551234567", domain=DomainType.VAYVORA)

        res = await engine.process_user_turn(state, "Are you hiring for AI engineering roles?")
        assert res.decision.detected_domain == DomainType.VAYVORA
        assert res.decision.detected_intent == "career_information"
        assert "edu:" not in "".join(res.grounded_citations)

    @pytest.mark.asyncio
    async def test_05_vayvora_corporate_requirement(self):
        """5. Vayvora corporate requirement captures business entities."""
        decision = ConversationalDecision(
            detected_domain=DomainType.VAYVORA,
            detected_intent="client_requirement",
            extracted_slots={"company_name": "Apex Logistics", "technical_requirement": "automated dispatch voice agent"},
            proposed_stage=ConversationStage.DISCOVERY,
            user_facing_response="We can build custom voice agents for logistics. Would you like to schedule a technical discovery call?",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s05", "+15551234567", domain=DomainType.VAYVORA)

        res = await engine.process_user_turn(state, "We represent Apex Logistics and need an automated dispatch voice agent.")
        assert state.extracted_slots["company_name"] == "Apex Logistics"
        assert state.extracted_slots["technical_requirement"] == "automated dispatch voice agent"

    @pytest.mark.asyncio
    async def test_06_vayvora_outbound_corporate_call(self):
        """6. Vayvora outbound corporate call maintains client context."""
        decision = ConversationalDecision(
            detected_domain=DomainType.VAYVORA,
            detected_intent="meeting_request",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="Certainly, let us schedule a consultation for your team.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_outbound_state(
            call_id="call-s06",
            caller_phone="+15553334444",
            domain=DomainType.VAYVORA,
            caller_name="David",
            company="Nexis Corp",
            campaign_id="CAMP-VAY-02",
            campaign_objective="Enterprise AI consultation",
        )

        res = await engine.process_user_turn(state, "Yes, we would like to explore your enterprise AI solutions.")
        assert state.caller.company == "Nexis Corp"
        assert state.current_domain == DomainType.VAYVORA

    @pytest.mark.asyncio
    async def test_07_domain_switching(self):
        """7. Domain switches cleanly mid-conversation from EduSaaS to Vayvora."""
        dec1 = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="course_information",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="At EduSaaS, we teach AI Engineering.",
        )
        dec2 = ConversationalDecision(
            detected_domain=DomainType.VAYVORA,
            detected_intent="company_information",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Vayvora provides enterprise AI software services.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[dec1, dec2])
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s07", "+15551234567", domain=DomainType.EDUSAAS)

        await engine.process_user_turn(state, "Tell me about EduSaaS courses.")
        assert state.current_domain == DomainType.EDUSAAS

        await engine.process_user_turn(state, "Does your parent company Vayvora build custom software?")
        assert state.current_domain == DomainType.VAYVORA

    @pytest.mark.asyncio
    async def test_08_intent_switching(self):
        """8. New explicit caller intent preempts previous pending agent questions."""
        dec1 = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="course_information",
            proposed_stage=ConversationStage.DISCOVERY,
            user_facing_response="Would you like weekend or weekday batches?",
            needs_clarification=True,
            clarification_question="Would you like weekend or weekday batches?",
        )
        dec2 = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="pricing",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="The total fee for the program is $1,200.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[dec1, dec2])
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s08", "+15551234567", domain=DomainType.EDUSAAS)

        await engine.process_user_turn(state, "Tell me about the course.")
        assert state.pending_question == "Would you like weekend or weekday batches?"

        await engine.process_user_turn(state, "Actually, how much does it cost?")
        assert state.current_intent == "pricing"
        assert state.pending_question is None

    @pytest.mark.asyncio
    async def test_09_rag_retrieval(self, populated_knowledge_provider):
        """9. RAG retrieval extracts matching chunks above threshold."""
        query = RAGQuery(
            domain=DomainType.VAYVORA,
            query_text="Vayvora Enterprise AI Solutions",
            top_k=3,
            relevance_threshold=0.5,
        )
        res = await populated_knowledge_provider.retrieve_grounded_context(query)
        assert res.knowledge_available is True
        assert len(res.chunks) > 0
        assert res.chunks[0].doc_id == "vay:solutions:enterprise_ai"

    @pytest.mark.asyncio
    async def test_10_rag_no_result_fallback(self, populated_knowledge_provider):
        """10. RAG no-result fallback safely communicates lack of verified details."""
        query = RAGQuery(
            domain=DomainType.EDUSAAS,
            query_text="Underwater Basket Weaving Doctorate",
            top_k=3,
            relevance_threshold=0.70,
        )
        res = await populated_knowledge_provider.retrieve_grounded_context(query)
        assert res.knowledge_available is False
        assert len(res.chunks) == 0

    @pytest.mark.asyncio
    async def test_11_redis_unavailable(self):
        """11. Redis unavailable returns safe service maintenance message."""
        from src.config import Settings
        from src.rag.redis_client import RedisVectorStore

        bad_settings = Settings(redis_port=59999)
        bad_store = RedisVectorStore(settings=bad_settings)
        provider = GroundedKnowledgeProvider(vector_store=bad_store, in_memory=False)
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="course_information",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Checking course information.",
            knowledge_required=True,
            knowledge_query="Quantum Computing Program",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=provider)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s11", "+15551234567", domain=DomainType.EDUSAAS)

        res = await engine.process_user_turn(state, "Tell me about quantum computing.")
        assert "maintenance" in res.response_text.lower()
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_12_email_action(self):
        """12. Email action proposes and executes with caller email."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="send_brochure",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="I will email the brochure to you.",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_email",
                arguments={"recipient": "john@example.com", "subject": "Brochure"},
            ),
        )
        mock_llm = MockLLMProvider(
            canned_decisions=[decision],
            canned_responses=["I have sent the brochure to john@example.com."],
        )
        mock_tools = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tools)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s12", "+15551234567", domain=DomainType.EDUSAAS)

        res = await engine.process_user_turn(state, "Please email me the brochure at john@example.com")
        assert res.tool_result is not None
        assert res.tool_result.success is True
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_13_calendar_availability(self):
        """13. Finding available slots queries MCP calendar tool."""
        decision = ConversationalDecision(
            detected_domain=DomainType.VAYVORA,
            detected_intent="meeting_request",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="Let me check our calendar for openings.",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="find_available_slots",
                arguments={"preferred_date": "Tomorrow"},
            ),
        )
        mock_llm = MockLLMProvider(
            canned_decisions=[decision],
            canned_responses=["I found slots available tomorrow at 10:00 AM and 2:00 PM."],
        )
        mock_tools = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tools)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s13", "+15551234567", domain=DomainType.VAYVORA)

        res = await engine.process_user_turn(state, "What slots do you have open tomorrow?")
        assert res.tool_result is not None
        assert res.tool_result.success is True
        assert "slots" in res.tool_result.data

    @pytest.mark.asyncio
    async def test_14_calendar_creation(self):
        """14. Calendar event creation executes only with confirmed slot."""
        decision = ConversationalDecision(
            detected_domain=DomainType.VAYVORA,
            detected_intent="schedule_consultation",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="Booking that slot now.",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="create_calendar_event",
                arguments={"slot": "Tomorrow 10:00 AM", "confirmed": True},
            ),
        )
        mock_llm = MockLLMProvider(
            canned_decisions=[decision],
            canned_responses=["Your meeting is confirmed for Tomorrow at 10:00 AM."],
        )
        mock_tools = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tools)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s14", "+15551234567", domain=DomainType.VAYVORA)

        res = await engine.process_user_turn(state, "Yes, 10:00 AM tomorrow works perfectly.")
        assert res.tool_result is not None
        assert res.tool_result.success is True
        assert state.last_action == "create_calendar_event"

    @pytest.mark.asyncio
    async def test_15_tool_failure_keeps_call_active(self):
        """15. Simulated tool failure does not crash or terminate the conversation."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="send_brochure",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="Sending email.",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_email",
                arguments={"recipient": "fail@example.com", "subject": "Test"},
            ),
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        mock_tools = MockToolProvider(force_unavailable=True)
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tools)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s15", "+15551234567", domain=DomainType.EDUSAAS)

        res = await engine.process_user_turn(state, "Please email it to fail@example.com")
        assert res.tool_result is not None
        assert res.tool_result.success is False
        assert state.conversation_active is True
        assert "system issue" in res.response_text.lower()

    @pytest.mark.asyncio
    async def test_16_tool_verification_failure_prevents_fake_success(self):
        """16. Tool verification failure prevents claiming success."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="send_brochure",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="Sending email.",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_email",
                arguments={"recipient": "unverified@example.com", "subject": "Test"},
            ),
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        mock_tools = MockToolProvider(force_verification_failure=True)
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tools)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s16", "+15551234567", domain=DomainType.EDUSAAS)

        res = await engine.process_user_turn(state, "Send it to unverified@example.com")
        assert res.tool_result.success is False
        assert state.last_action is None  # Never completed
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_17_unsupported_action_rejected_safely(self):
        """17. Proposing an unsupported action rejects safely and conversation remains active."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="unknown_action",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="I will try that action.",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="unsupported_database_delete",
                arguments={},
            ),
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        mock_tools = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tools)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s17", "+15551234567", domain=DomainType.EDUSAAS)

        res = await engine.process_user_turn(state, "Delete the database.")
        assert res.tool_result is not None
        assert res.tool_result.success is False
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_18_explicit_termination(self):
        """18. Explicit caller farewell terminates conversation cleanly."""
        decision = ConversationalDecision(
            detected_domain=DomainType.GENERAL,
            detected_intent="farewell",
            proposed_stage=ConversationStage.COMPLETED,
            user_facing_response="Thank you for calling. Have a great day! Goodbye!",
            suggested_termination=True,
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s18", "+15551234567", domain=DomainType.EDUSAAS)

        res = await engine.process_user_turn(state, "That's all, thank you. Goodbye.")
        assert state.conversation_active is False
        assert state.termination_requested is True

    @pytest.mark.asyncio
    async def test_19_no_without_termination(self):
        """19. A simple 'No' does NOT terminate the call."""
        decision = ConversationalDecision(
            detected_domain=DomainType.GENERAL,
            detected_intent="decline",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Understood! Is there anything else I can help you with?",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s19", "+15551234567", domain=DomainType.EDUSAAS)

        res = await engine.process_user_turn(state, "No thanks.")
        assert state.conversation_active is True
        assert state.termination_requested is False

    @pytest.mark.asyncio
    async def test_20_barge_in_interruption(self):
        """20. Barge-in flushes assistant audio queue immediately."""
        vad_adapter = PipecatVADAdapter(MockVADProvider())
        stt_adapter = PipecatSTTAdapter(MockSTTProvider())
        conv_adapter = PipecatConversationAdapter(ConversationEngine(llm_provider=MockLLMProvider()))
        tts_adapter = PipecatTTSAdapter(MockTTSProvider())
        output_sink = MockAudioOutput()

        pipeline = VoicePipeline(
            session_id="voice-barge-in-01",
            vad_adapter=vad_adapter,
            stt_adapter=stt_adapter,
            conversation_adapter=conv_adapter,
            tts_adapter=tts_adapter,
            audio_output=output_sink,
        )

        output_sink.write(b"RIFF" + b"\x00" * 3200)
        assert len(output_sink.written_chunks) == 1

        pipeline.trigger_barge_in()
        assert len(output_sink.written_chunks) == 0
        assert output_sink.cleared_count > 0

    @pytest.mark.asyncio
    async def test_21_unknown_inbound_caller(self):
        """21. Unknown inbound caller starts with unpopulated profile."""
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s21", "+15559998888", domain=DomainType.EDUSAAS)
        assert state.caller.name is None
        assert state.caller.email is None
        assert state.caller.phone == "+15559998888"

    @pytest.mark.asyncio
    async def test_22_known_caller(self):
        """22. Known caller starts with pre-populated name and contact."""
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state(
            call_id="call-s22",
            caller_phone="+15551112222",
            domain=DomainType.VAYVORA,
            caller_name="Sarah Connor",
        )
        assert state.caller.name == "Sarah Connor"
        assert state.caller.phone == "+15551112222"

    @pytest.mark.asyncio
    async def test_23_outbound_caller_context(self):
        """23. Outbound caller context is fully initialized."""
        mgr = ConversationStateManager()
        state = mgr.create_outbound_state(
            call_id="call-s23",
            caller_phone="+15554445555",
            domain=DomainType.VAYVORA,
            caller_name="Michael Scott",
            campaign_id="CAMP-ENT-04",
            campaign_objective="AI workflow demo",
            company="Dunder Mifflin",
        )
        assert state.caller.name == "Michael Scott"
        assert state.caller.company == "Dunder Mifflin"
        assert state.caller.campaign == "CAMP-ENT-04"

    @pytest.mark.asyncio
    async def test_24_llm_timeout(self):
        """24. LLM timeout is translated into LLMTimeoutError."""
        from unittest.mock import AsyncMock

        mock_llm = MockLLMProvider()
        mock_llm.generate_decision = AsyncMock(side_effect=LLMTimeoutError("Request timed out"))
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s24", "+15551234567")

        with pytest.raises(LLMTimeoutError):
            await engine.process_user_turn(state, "Hello?")

    @pytest.mark.asyncio
    async def test_25_malformed_llm_json(self):
        """25. Malformed JSON response raises LLMMalformedResponseError."""
        from unittest.mock import AsyncMock

        mock_llm = MockLLMProvider()
        mock_llm.generate_decision = AsyncMock(side_effect=LLMMalformedResponseError("Invalid JSON"))
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s25", "+15551234567")

        with pytest.raises(LLMMalformedResponseError):
            await engine.process_user_turn(state, "Hello?")


# =============================================================================
# Section 16: End-to-End Golden Tests (Conversations 1 - 6)
# =============================================================================

class TestEndToEndGoldenConversations:
    """End-to-end multi-turn golden tests matching Section 16 exact conversation flows."""

    @pytest.mark.asyncio
    async def test_golden_01_vayvora_ai_solutions(self, populated_knowledge_provider):
        """TEST 1: Inbound question about Vayvora AI solutions with optimized retrieval."""
        decision = ConversationalDecision(
            detected_domain=DomainType.VAYVORA,
            detected_intent="ai_solution",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Vayvora provides enterprise AI solutions including conversational voice agents, custom RAG pipelines, and automated agent workflows.",
            knowledge_required=True,
            knowledge_query="Vayvora enterprise AI solutions voice agents custom RAG pipelines",
        )
        mock_llm = MockLLMProvider(
            canned_decisions=[decision],
            canned_responses=[
                "Vayvora provides enterprise AI solutions including conversational voice agents and custom RAG pipelines."
            ],
        )
        engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=populated_knowledge_provider)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("golden-01", "+15551234567", domain=DomainType.VAYVORA)

        res = await engine.process_user_turn(state, "Hello, what AI solutions does Vayvora provide?")

        assert state.current_domain == DomainType.VAYVORA
        assert state.current_intent == "ai_solution"
        assert res.knowledge_required is True
        assert len(res.grounded_citations) > 0
        assert "vay:solutions:enterprise_ai" in res.grounded_citations
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_golden_02_email_action_missing_then_provided(self):
        """TEST 2: Email requested without email address -> ask first -> execute upon receiving email."""
        # Turn 1: Propose email action but email is unknown
        dec1 = ConversationalDecision(
            detected_domain=DomainType.VAYVORA,
            detected_intent="send_brochure",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="I would be happy to email those details. Could you please share your email address?",
            action_proposed=True,
            proposed_action=ProposedAction(tool_name="send_email", arguments={}),
        )
        # Turn 2: Caller provides email
        dec2 = ConversationalDecision(
            detected_domain=DomainType.VAYVORA,
            detected_intent="send_brochure",
            extracted_slots={"email": "client@vayvora-partner.com"},
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="Sending details to your email now.",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_email",
                arguments={"recipient": "client@vayvora-partner.com", "subject": "Vayvora AI Overview"},
            ),
        )

        mock_llm = MockLLMProvider(
            canned_decisions=[dec1, dec2],
            canned_responses=["I have sent the details to client@vayvora-partner.com."],
        )
        mock_tools = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tools)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("golden-02", "+15551234567", domain=DomainType.VAYVORA)

        # Turn 1
        res1 = await engine.process_user_turn(state, "Can you send me the details by email?")
        assert res1.tool_result is None  # Guard prevented execution without email
        assert "share your email" in res1.response_text.lower()
        assert state.pending_question == "Could you please share your email address?"
        assert state.conversation_active is True

        # Turn 2
        res2 = await engine.process_user_turn(state, "My email is client@vayvora-partner.com")
        assert res2.tool_result is not None
        assert res2.tool_result.success is True
        assert state.last_action == "send_email"
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_golden_03_topic_pivot_to_location(self, populated_knowledge_provider):
        """TEST 3: Caller pivots to office location; prior pending action does NOT override."""
        dec = ConversationalDecision(
            detected_domain=DomainType.VAYVORA,
            detected_intent="company_location",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Vayvora is located in Bangalore, India.",
            knowledge_required=True,
            knowledge_query="Vayvora headquarters Bangalore India",
        )
        mock_llm = MockLLMProvider(
            canned_decisions=[dec],
            canned_responses=["Our headquarters is located in Bangalore, India."],
        )
        engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=populated_knowledge_provider)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("golden-03", "+15551234567", domain=DomainType.VAYVORA)
        state.set_pending_action("send_email")
        state.set_pending_question("Could you please share your email address?")

        res = await engine.process_user_turn(state, "Actually, what is your office location?")
        assert state.current_intent == "company_location"
        assert state.pending_question is None
        assert "vay:company:location" in res.grounded_citations
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_golden_04_no_does_not_terminate(self):
        """TEST 4: Caller says 'No' -> conversation remains active."""
        dec = ConversationalDecision(
            detected_domain=DomainType.VAYVORA,
            detected_intent="decline",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="No problem at all! Let me know if you have any other questions.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[dec])
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("golden-04", "+15551234567", domain=DomainType.VAYVORA)

        res = await engine.process_user_turn(state, "No.")
        assert state.conversation_active is True
        assert state.termination_requested is False

    @pytest.mark.asyncio
    async def test_golden_05_explicit_goodbye_terminates(self):
        """TEST 5: Caller explicitly says goodbye -> terminates conversation."""
        dec = ConversationalDecision(
            detected_domain=DomainType.GENERAL,
            detected_intent="farewell",
            proposed_stage=ConversationStage.COMPLETED,
            user_facing_response="Thank you for reaching out to Vayvora. Have a wonderful day. Goodbye!",
            suggested_termination=True,
        )
        mock_llm = MockLLMProvider(canned_decisions=[dec])
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("golden-05", "+15551234567", domain=DomainType.VAYVORA)

        res = await engine.process_user_turn(state, "That's all, thank you. Goodbye.")
        assert state.conversation_active is False
        assert state.termination_requested is True

    @pytest.mark.asyncio
    async def test_golden_06_connect_advisor_safely_handled_without_fake_success(self):
        """TEST 6: Caller asks for advisor -> no fake connect_advisor execution, safe handling."""
        # Scenario A: Phone unknown -> dropped from external dispatch, handled conversationally
        dec_no_phone = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="connect_advisor",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="I would be happy to have an advisor call you. What is your phone number?",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="connect_advisor",
                arguments={"notes": "Caller wants advisor"},
            ),
        )
        mock_llm = MockLLMProvider(canned_decisions=[dec_no_phone])
        mock_tools = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tools)
        mgr = ConversationStateManager()
        state_no_phone = mgr.create_inbound_state("golden-06a", "+15551234567", domain=DomainType.EDUSAAS)

        res_a = await engine.process_user_turn(state_no_phone, "Can you connect me with an advisor?")
        # Verified: No fake tool execution happened; conversation remains active
        assert res_a.tool_result is None
        assert state_no_phone.last_action is None
        assert state_no_phone.conversation_active is True

        # Scenario B: Phone known -> mapped to supported create_hr_followup
        dec_with_phone = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="connect_advisor",
            extracted_slots={"phone": "+15558889999", "name": "Alex"},
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="I have scheduled an advisor to call you at +15558889999.",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="connect_advisor",
                arguments={"phone": "+15558889999", "name": "Alex"},
            ),
        )
        mock_llm_b = MockLLMProvider(
            canned_decisions=[dec_with_phone],
            canned_responses=["Your advisor callback is confirmed."],
        )
        engine_b = ConversationEngine(llm_provider=mock_llm_b, tool_provider=mock_tools)
        state_with_phone = mgr.create_inbound_state("golden-06b", "+15558889999", domain=DomainType.EDUSAAS)

        res_b = await engine_b.process_user_turn(state_with_phone, "Please connect me with an advisor at +15558889999.")
        assert res_b.tool_result is not None
        assert res_b.tool_result.success is True
        assert state_with_phone.last_action == "create_hr_followup"
        assert state_with_phone.conversation_active is True
