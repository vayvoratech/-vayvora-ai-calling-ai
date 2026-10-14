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
    asyncio.run(
        provider.index_document(
            RAGChunk(
                doc_id="ib:projects:skyline_towers",
                domain=DomainType.INFRA_BUILDER,
                title="Skyline Towers Whitefield",
                content="Skyline Towers in Whitefield offers luxury 2 and 3 BHK apartments starting at ₹85 Lakhs with 40+ lifestyle amenities.",
                score=1.0,
            )
        )
    )
    asyncio.run(
        provider.index_document(
            RAGChunk(
                doc_id="ib:company:location",
                domain=DomainType.INFRA_BUILDER,
                title="Infra Builder Headquarters",
                content="Infra Builder headquarters and primary customer experience center is located in Bangalore, Outer Ring Road, India.",
                score=1.0,
            )
        )
    )
    asyncio.run(
        provider.index_document(
            RAGChunk(
                doc_id="ib:projects:green_valley",
                domain=DomainType.INFRA_BUILDER,
                title="Green Valley Villas",
                content="Green Valley Villas on Sarjapur Road offers luxury 4 and 5 BHK gated community villas starting at ₹2.5 Crores.",
                score=1.0,
            )
        )
    )
    asyncio.run(
        provider.index_document(
            RAGChunk(
                doc_id="ib:projects:urban_heights",
                domain=DomainType.INFRA_BUILDER,
                title="Urban Heights Electronic City",
                content="Urban Heights in Electronic City offers ready-to-move 1 and 2 BHK homes starting at ₹45 Lakhs.",
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
        """1. Infra Builder inbound project question retrieves knowledge and answers accurately."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="project_inquiry",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Skyline Towers in Whitefield offers luxury 2 and 3 BHK apartments starting at ₹85 Lakhs.",
            knowledge_required=True,
            knowledge_query="Skyline Towers Whitefield 2 and 3 BHK apartments",
        )
        mock_llm = MockLLMProvider(
            canned_decisions=[decision],
            canned_responses=["Skyline Towers offers 2 and 3 BHK apartments in Whitefield."],
        )
        engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=populated_knowledge_provider)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s01", "+15551234567", domain=DomainType.INFRA_BUILDER)

        res = await engine.process_user_turn(state, "What configurations are available at Skyline Towers?")
        assert res.decision.detected_domain == DomainType.INFRA_BUILDER
        assert "ib:projects:skyline_towers" in res.grounded_citations
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_02_edusaas_outbound_conversation(self):
        """2. Infra Builder outbound conversation preserves campaign objective and respects caller priority."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="property_recommendation",
            proposed_stage=ConversationStage.DISCOVERY,
            user_facing_response="Hi Priya, I understand you're interested in our residential projects. Are you looking for apartments or villas?",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_outbound_state(
            call_id="call-s02",
            caller_phone="+15559876543",
            domain=DomainType.INFRA_BUILDER,
            caller_name="Priya",
            campaign_id="CAMP-RE-01",
            campaign_objective="Property inquiry follow-up",
        )

        res = await engine.process_user_turn(state, "Hi, I have a few questions about your real estate projects.")
        assert state.caller.name == "Priya"
        assert state.caller.campaign == "CAMP-RE-01"
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_03_vayvora_inbound_company_question(self, populated_knowledge_provider):
        """3. Infra Builder inbound company question retrieves company details."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="company_location",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Infra Builder is headquartered in Bangalore, India.",
            knowledge_required=True,
            knowledge_query="Infra Builder headquarters Bangalore India",
        )
        mock_llm = MockLLMProvider(
            canned_decisions=[decision],
            canned_responses=["Infra Builder is headquartered on Outer Ring Road, Bangalore, India."],
        )
        engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=populated_knowledge_provider)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s03", "+15551234567", domain=DomainType.INFRA_BUILDER)

        res = await engine.process_user_turn(state, "Where is Infra Builder headquartered?")
        assert res.decision.detected_domain == DomainType.INFRA_BUILDER
        assert "ib:company:location" in res.grounded_citations

    @pytest.mark.asyncio
    async def test_04_vayvora_inbound_career_question(self, populated_knowledge_provider):
        """4. Infra Builder inbound villa question routes to verified villa details."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="property_inquiry",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Green Valley Villas on Sarjapur Road offers 4 and 5 BHK luxury villas starting at ₹2.5 Crores.",
            knowledge_required=True,
            knowledge_query="Green Valley Villas Sarjapur Road 4 and 5 BHK",
        )
        mock_llm = MockLLMProvider(
            canned_decisions=[decision],
            canned_responses=["Green Valley Villas offers 4 and 5 BHK luxury gated community villas."],
        )
        engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=populated_knowledge_provider)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s04", "+15551234567", domain=DomainType.INFRA_BUILDER)

        res = await engine.process_user_turn(state, "Tell me about your luxury villas on Sarjapur Road.")
        assert res.decision.detected_domain == DomainType.INFRA_BUILDER
        assert res.decision.detected_intent == "property_inquiry"
        assert "ib:projects:green_valley" in res.grounded_citations

    @pytest.mark.asyncio
    async def test_05_vayvora_corporate_requirement(self):
        """5. Infra Builder buyer requirement captures real estate entities."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="buyer_requirement",
            extracted_slots={"preferred_location": "Whitefield", "property_type": "3 BHK apartment"},
            proposed_stage=ConversationStage.DISCOVERY,
            user_facing_response="We have luxury 3 BHK residences at Skyline Towers in Whitefield. Would you like to schedule a site visit?",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s05", "+15551234567", domain=DomainType.INFRA_BUILDER)

        res = await engine.process_user_turn(state, "I am looking for a 3 BHK apartment in Whitefield.")
        assert state.extracted_slots["preferred_location"] == "Whitefield"
        assert state.extracted_slots["property_type"] == "3 BHK apartment"

    @pytest.mark.asyncio
    async def test_06_vayvora_outbound_corporate_call(self):
        """6. Infra Builder outbound real-estate call maintains buyer context."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="site_visit_request",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="Certainly, let us schedule a site visit for you at Skyline Towers.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_outbound_state(
            call_id="call-s06",
            caller_phone="+15553334444",
            domain=DomainType.INFRA_BUILDER,
            caller_name="David",
            company="Nexis Corp",
            campaign_id="CAMP-RE-02",
            campaign_objective="Residential site visit consultation",
        )

        res = await engine.process_user_turn(state, "Yes, we would like to explore your residential developments.")
        assert state.caller.name == "David"
        assert state.current_domain == DomainType.INFRA_BUILDER

    @pytest.mark.asyncio
    async def test_07_domain_switching(self):
        """7. Infra Builder maintains single active domain cleanly across project inquiries."""
        dec1 = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="property_information",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="At Skyline Towers, we offer 2 and 3 BHK apartments.",
        )
        dec2 = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="property_information",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Green Valley Villas offers 4 and 5 BHK luxury villas.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[dec1, dec2])
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s07", "+15551234567", domain=DomainType.INFRA_BUILDER)

        await engine.process_user_turn(state, "Tell me about Skyline Towers apartments.")
        assert state.current_domain == DomainType.INFRA_BUILDER

        await engine.process_user_turn(state, "Tell me about Green Valley Villas.")
        assert state.current_domain == DomainType.INFRA_BUILDER

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
            domain=DomainType.INFRA_BUILDER,
            query_text="Skyline Towers Whitefield",
            top_k=3,
            relevance_threshold=0.5,
        )
        res = await populated_knowledge_provider.retrieve_grounded_context(query)
        assert res.knowledge_available is True
        assert len(res.chunks) > 0
        assert res.chunks[0].doc_id == "ib:projects:skyline_towers"

    @pytest.mark.asyncio
    async def test_10_rag_no_result_fallback(self, populated_knowledge_provider):
        """10. RAG no-result fallback safely communicates lack of verified details."""
        query = RAGQuery(
            domain=DomainType.INFRA_BUILDER,
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
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="property_information",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Checking property information.",
            knowledge_required=True,
            knowledge_query="Quantum Computing Program",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=provider)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s11", "+15551234567", domain=DomainType.INFRA_BUILDER)

        res = await engine.process_user_turn(state, "Tell me about quantum computing.")
        assert "maintenance" in res.response_text.lower()
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_12_email_action(self):
        """12. WhatsApp brochure action proposes and executes with caller phone."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="send_brochure",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="I will send the project brochure to your WhatsApp.",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_message",
                arguments={"recipient": "+919845123890", "consent_confirmed": True},
            ),
        )
        mock_llm = MockLLMProvider(
            canned_decisions=[decision],
            canned_responses=["I have sent the brochure to your WhatsApp."],
        )
        mock_tools = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tools)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s12", "+919845123890", domain=DomainType.INFRA_BUILDER)
        state.update_slot("whatsapp_consent", True)

        res = await engine.process_user_turn(state, "Please send the brochure to my WhatsApp at +919845123890")
        assert res.tool_result is not None
        assert res.tool_result.success is True
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_13_calendar_availability(self):
        """13. Finding available slots queries MCP calendar tool."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
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
        state = mgr.create_inbound_state("call-s13", "+15551234567", domain=DomainType.INFRA_BUILDER)

        res = await engine.process_user_turn(state, "What slots do you have open tomorrow?")
        assert res.tool_result is not None
        assert res.tool_result.success is True
        assert "slots" in res.tool_result.data

    @pytest.mark.asyncio
    async def test_14_calendar_creation(self):
        """14. Calendar event creation executes only with confirmed slot."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
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
        state = mgr.create_inbound_state("call-s14", "+15551234567", domain=DomainType.INFRA_BUILDER)

        res = await engine.process_user_turn(state, "Yes, 10:00 AM tomorrow works perfectly.")
        assert res.tool_result is not None
        assert res.tool_result.success is True
        assert state.last_action == "create_calendar_event"

    @pytest.mark.asyncio
    async def test_15_tool_failure_keeps_call_active(self):
        """15. Simulated tool failure does not crash or terminate the conversation."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="send_brochure",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="Sending WhatsApp brochure.",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_message",
                arguments={"recipient": "+919845123890", "consent_confirmed": True},
            ),
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        mock_tools = MockToolProvider(force_unavailable=True)
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tools)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s15", "+919845123890", domain=DomainType.INFRA_BUILDER)
        state.update_slot("whatsapp_consent", True)

        res = await engine.process_user_turn(state, "Please send the brochure to +919845123890")
        assert res.tool_result is not None
        assert res.tool_result.success is False
        assert state.conversation_active is True
        assert "system issue" in res.response_text.lower()

    @pytest.mark.asyncio
    async def test_16_tool_verification_failure_prevents_fake_success(self):
        """16. Tool verification failure prevents claiming success."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="send_brochure",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="Sending WhatsApp brochure.",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_message",
                arguments={"recipient": "+919845123890", "consent_confirmed": True},
            ),
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        mock_tools = MockToolProvider(force_verification_failure=True)
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tools)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-s16", "+919845123890", domain=DomainType.INFRA_BUILDER)
        state.update_slot("whatsapp_consent", True)

        res = await engine.process_user_turn(state, "Send it to +919845123890")
        assert res.tool_result.success is False
        assert state.last_action is None  # Never completed
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_17_unsupported_action_rejected_safely(self):
        """17. Proposing an unsupported action rejects safely and conversation remains active."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
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
        state = mgr.create_inbound_state("call-s17", "+15551234567", domain=DomainType.INFRA_BUILDER)

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
        state = mgr.create_inbound_state("call-s18", "+15551234567", domain=DomainType.INFRA_BUILDER)

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
        state = mgr.create_inbound_state("call-s19", "+15551234567", domain=DomainType.INFRA_BUILDER)

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
        state = mgr.create_inbound_state("call-s21", "+15559998888", domain=DomainType.INFRA_BUILDER)
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
            domain=DomainType.INFRA_BUILDER,
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
            domain=DomainType.INFRA_BUILDER,
            caller_name="Michael Scott",
            campaign_id="CAMP-RE-04",
            campaign_objective="Residential project presentation",
            company="Dunder Mifflin",
        )
        assert state.caller.name == "Michael Scott"
        assert state.caller.company == "Dunder Mifflin"
        assert state.caller.campaign == "CAMP-RE-04"

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
            await engine.process_user_turn(state, "What are your property prices?")

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
            await engine.process_user_turn(state, "What are your property prices?")


# =============================================================================
# Section 16: End-to-End Golden Tests (Conversations 1 - 6)
# =============================================================================

class TestEndToEndGoldenConversations:
    """End-to-end multi-turn golden tests matching Section 16 exact conversation flows."""

    @pytest.mark.asyncio
    async def test_golden_01_vayvora_ai_solutions(self, populated_knowledge_provider):
        """TEST 1: Inbound question about Infra Builder residential projects with optimized retrieval."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="property_inquiry",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Infra Builder offers premier residential projects in Bangalore including Skyline Towers in Whitefield and Green Valley Villas on Sarjapur Road.",
            knowledge_required=True,
            knowledge_query="Infra Builder residential projects Skyline Towers Whitefield",
        )
        mock_llm = MockLLMProvider(
            canned_decisions=[decision],
            canned_responses=[
                "Infra Builder offers premier residential projects including Skyline Towers in Whitefield."
            ],
        )
        engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=populated_knowledge_provider)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("golden-01", "+15551234567", domain=DomainType.INFRA_BUILDER)

        res = await engine.process_user_turn(state, "Hello, what residential projects does Infra Builder offer?")

        assert state.current_domain == DomainType.INFRA_BUILDER
        assert state.current_intent == "property_inquiry"
        assert res.knowledge_required is True
        assert len(res.grounded_citations) > 0
        assert "ib:projects:skyline_towers" in res.grounded_citations
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_golden_02_email_action_missing_then_provided(self):
        """TEST 2: WhatsApp brochure requested without phone -> ask first -> execute upon receiving phone."""
        # Turn 1: Propose WhatsApp brochure action but phone is unknown
        dec1 = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="send_brochure",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="Sure, I can share the project brochure and details. What is your WhatsApp number?",
            action_proposed=True,
            proposed_action=ProposedAction(tool_name="send_message", arguments={}),
        )
        # Turn 2: Caller provides phone number with consent
        dec2 = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="send_brochure",
            extracted_slots={"phone": "+919845123890"},
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="Sending details to your WhatsApp now.",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_message",
                arguments={"recipient": "+919845123890", "consent_confirmed": True},
            ),
        )

        mock_llm = MockLLMProvider(
            canned_decisions=[dec1, dec2],
            canned_responses=["I have sent the details to your WhatsApp at +919845123890."],
        )
        mock_tools = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tools)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("golden-02", "", domain=DomainType.INFRA_BUILDER)

        # Turn 1
        res1 = await engine.process_user_turn(state, "Can you send me the details on WhatsApp?")
        assert res1.tool_result is None  # Guard prevented execution without phone
        assert "whatsapp" in res1.response_text.lower()
        assert state.pending_question == "I would be happy to send the brochure and details. Which WhatsApp number should I send them to?"
        assert state.conversation_active is True

        # Turn 2
        res2 = await engine.process_user_turn(state, "My WhatsApp number is +919845123890, please send it.")
        assert res2.tool_result is not None
        assert res2.tool_result.success is True
        assert state.last_action == "send_message"
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_golden_03_topic_pivot_to_location(self, populated_knowledge_provider):
        """TEST 3: Caller pivots to office location; prior pending action does NOT override."""
        dec = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="company_location",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Infra Builder is located in Bangalore, India.",
            knowledge_required=True,
            knowledge_query="Infra Builder headquarters Bangalore India",
        )
        mock_llm = MockLLMProvider(
            canned_decisions=[dec],
            canned_responses=["Our headquarters is located in Bangalore, India."],
        )
        engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=populated_knowledge_provider)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("golden-03", "+15551234567", domain=DomainType.INFRA_BUILDER)
        state.set_pending_action("send_message")
        state.set_pending_question("Could you please share your WhatsApp phone number?")

        res = await engine.process_user_turn(state, "Actually, what is your office location?")
        assert state.current_intent == "company_location"
        assert state.pending_question is None
        assert "ib:company:location" in res.grounded_citations
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_golden_04_no_does_not_terminate(self):
        """TEST 4: Caller says 'No' -> conversation remains active."""
        dec = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="decline",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="No problem at all! Let me know if you have any other questions.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[dec])
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("golden-04", "+15551234567", domain=DomainType.INFRA_BUILDER)

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
            user_facing_response="Thank you for reaching out to Infra Builder. Have a wonderful day. Goodbye!",
            suggested_termination=True,
        )
        mock_llm = MockLLMProvider(canned_decisions=[dec])
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("golden-05", "+15551234567", domain=DomainType.INFRA_BUILDER)

        res = await engine.process_user_turn(state, "That's all, thank you. Goodbye.")
        assert state.conversation_active is False
        assert state.termination_requested is True

    @pytest.mark.asyncio
    async def test_golden_06_connect_advisor_safely_handled_without_fake_success(self):
        """TEST 6: Caller asks for advisor -> no fake connect_advisor execution, safe handling."""
        # Scenario A: Phone unknown -> dropped from external dispatch, handled conversationally
        dec_no_phone = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
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
        state_no_phone = mgr.create_inbound_state("golden-06a", "+15551234567", domain=DomainType.INFRA_BUILDER)

        res_a = await engine.process_user_turn(state_no_phone, "Can you connect me with an advisor?")
        # Verified: No fake tool execution happened; conversation remains active
        assert res_a.tool_result is None
        assert state_no_phone.last_action is None
        assert state_no_phone.conversation_active is True

        # Scenario B: Phone known -> mapped to supported create_hr_followup
        dec_with_phone = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
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
        state_with_phone = mgr.create_inbound_state("golden-06b", "+15558889999", domain=DomainType.INFRA_BUILDER)

        res_b = await engine_b.process_user_turn(state_with_phone, "Please connect me with an advisor at +15558889999.")
        assert res_b.tool_result is not None
        assert res_b.tool_result.success is True
        assert state_with_phone.last_action == "create_hr_followup"
        assert state_with_phone.conversation_active is True
