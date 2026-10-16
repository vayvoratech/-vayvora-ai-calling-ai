"""Unit tests for GeminiLLMProvider and MockLLMProvider."""

import json
from unittest.mock import AsyncMock, patch
import httpx
import pytest
from pydantic import SecretStr

from src.config import Settings
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.errors import (
    LLMAuthenticationError,
    LLMError,
    LLMMalformedResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.core.llm import GeminiLLMProvider, MockLLMProvider, extract_json_block
from src.core.types import ConversationStage, DomainType


class TestExtractJsonBlock:
    """Test JSON block extraction from various LLM response formats."""

    def test_extract_clean_json(self):
        raw = '{"detected_domain": "edusaas"}'
        assert extract_json_block(raw) == '{"detected_domain": "edusaas"}'

    def test_extract_fenced_json(self):
        raw = '```json\n{"detected_domain": "vayvora"}\n```'
        assert extract_json_block(raw) == '{"detected_domain": "vayvora"}'

    def test_extract_json_with_surrounding_text(self):
        raw = 'Here is the decision:\n{"detected_domain": "general"}\nHope this helps!'
        assert extract_json_block(raw) == '{"detected_domain": "general"}'


class TestGeminiLLMProvider:
    """Test Gemini provider configuration, error handling, and generation."""

    def test_missing_api_key_raises_auth_error(self):
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY=None,
        )
        provider = GeminiLLMProvider(settings=settings)
        with pytest.raises(LLMAuthenticationError) as exc_info:
            _ = provider.api_key
        assert "Gemini API key is not configured" in str(exc_info.value)

    def test_model_configuration_preserved(self):
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY="test-key",
            GEMINI_MODEL="gemini-3.5-flash-lite",
        )
        provider = GeminiLLMProvider(settings=settings)
        assert provider.model == "gemini-3.5-flash-lite"

    def test_default_model_is_flash_lite(self):
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY="test-key",
        )
        provider = GeminiLLMProvider(settings=settings)
        assert provider.model == "gemini-3.5-flash-lite"

    def test_api_key_never_exposed_in_repr(self):
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY="super-secret-api-key",
        )
        provider = GeminiLLMProvider(settings=settings)
        assert "super-secret-api-key" not in repr(provider.settings.gemini_api_key)

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY="test-key",
            GEMINI_TIMEOUT_SECONDS=5.0,
        )
        provider = GeminiLLMProvider(settings=settings)

        with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Read timeout")):
            with pytest.raises(LLMTimeoutError):
                await provider.generate_response("Test prompt")

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self):
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY="test-key",
        )
        provider = GeminiLLMProvider(settings=settings)

        mock_resp = httpx.Response(status_code=429, text="Rate limit exceeded")
        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            with pytest.raises(LLMRateLimitError):
                await provider.generate_response("Test prompt")

    @pytest.mark.asyncio
    async def test_auth_failure_handling(self):
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY="invalid-key",
        )
        provider = GeminiLLMProvider(settings=settings)

        mock_resp = httpx.Response(status_code=401, text="API key not valid")
        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            with pytest.raises(LLMAuthenticationError):
                await provider.generate_response("Test prompt")

    @pytest.mark.asyncio
    async def test_successful_generate_response(self):
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY="test-key",
        )
        provider = GeminiLLMProvider(settings=settings)

        mock_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello, how can I help you today?"}]
                    }
                }
            ]
        }
        mock_resp = httpx.Response(status_code=200, json=mock_payload)
        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            result = await provider.generate_response("Hello")
            assert result == "Hello, how can I help you today?"

    @pytest.mark.asyncio
    async def test_successful_generate_decision(self):
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY="test-key",
        )
        provider = GeminiLLMProvider(settings=settings)

        decision_data = {
            "detected_domain": "infra_builder",
            "detected_intent": "project_information",
            "extracted_slots": {"preferred_project": "Skyline Towers"},
            "proposed_stage": "information",
            "user_facing_response": "Skyline Towers offers luxury 2 and 3 BHK apartments in Whitefield.",
            "knowledge_required": True,
            "knowledge_query": "Skyline Towers pricing and floor plans",
            "action_proposed": False,
            "needs_clarification": False,
            "suggested_termination": False,
        }
        mock_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps(decision_data)}]
                    }
                }
            ]
        }
        mock_resp = httpx.Response(status_code=200, json=mock_payload)
        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            decision = await provider.generate_decision("Tell me about Skyline Towers", "system prompt")
            assert decision.detected_domain == DomainType.INFRA_BUILDER
            assert decision.detected_intent == "project_information"
            assert decision.knowledge_required is True

    @pytest.mark.asyncio
    async def test_decision_request_includes_default_max_output_tokens_512(self):
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY="test-key",
        )
        provider = GeminiLLMProvider(settings=settings)
        captured_payloads = []

        async def fake_post(url, **kwargs):
            captured_payloads.append(kwargs.get("json"))
            decision_data = {
                "detected_domain": "infra_builder",
                "detected_intent": "project_information",
                "extracted_slots": {},
                "proposed_stage": "information",
                "user_facing_response": "We have several projects available.",
                "knowledge_required": False,
                "action_proposed": False,
                "needs_clarification": False,
                "suggested_termination": False,
            }
            return httpx.Response(
                status_code=200,
                json={"candidates": [{"content": {"parts": [{"text": json.dumps(decision_data)}]}}]},
            )

        with patch("httpx.AsyncClient.post", side_effect=fake_post):
            await provider.generate_decision("Tell me about projects", "system prompt")

        assert len(captured_payloads) == 1
        gen_config = captured_payloads[0]["generationConfig"]
        assert gen_config["maxOutputTokens"] == 512

    @pytest.mark.asyncio
    async def test_decision_request_respects_configured_token_override(self):
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY="test-key",
            GEMINI_MAX_DECISION_TOKENS=1024,
        )
        provider = GeminiLLMProvider(settings=settings)
        captured_payloads = []

        async def fake_post(url, **kwargs):
            captured_payloads.append(kwargs.get("json"))
            decision_data = {
                "detected_domain": "infra_builder",
                "detected_intent": "project_information",
                "extracted_slots": {},
                "proposed_stage": "information",
                "user_facing_response": "We have several projects available.",
                "knowledge_required": False,
                "action_proposed": False,
                "needs_clarification": False,
                "suggested_termination": False,
            }
            return httpx.Response(
                status_code=200,
                json={"candidates": [{"content": {"parts": [{"text": json.dumps(decision_data)}]}}]},
            )

        with patch("httpx.AsyncClient.post", side_effect=fake_post):
            await provider.generate_decision("Tell me about projects", "system prompt")

        assert len(captured_payloads) == 1
        gen_config = captured_payloads[0]["generationConfig"]
        assert gen_config["maxOutputTokens"] == 1024

    @pytest.mark.asyncio
    async def test_decision_request_includes_application_json_mime_type(self):
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY="test-key",
        )
        provider = GeminiLLMProvider(settings=settings)
        captured_payloads = []

        async def fake_post(url, **kwargs):
            captured_payloads.append(kwargs.get("json"))
            decision_data = {
                "detected_domain": "infra_builder",
                "detected_intent": "project_information",
                "extracted_slots": {},
                "proposed_stage": "information",
                "user_facing_response": "Hello",
                "knowledge_required": False,
                "action_proposed": False,
                "needs_clarification": False,
                "suggested_termination": False,
            }
            return httpx.Response(
                status_code=200,
                json={"candidates": [{"content": {"parts": [{"text": json.dumps(decision_data)}]}}]},
            )

        with patch("httpx.AsyncClient.post", side_effect=fake_post):
            await provider.generate_decision("Hello", "system prompt")

        assert len(captured_payloads) == 1
        assert captured_payloads[0]["generationConfig"]["responseMimeType"] == "application/json"

    @pytest.mark.asyncio
    async def test_decision_request_does_not_include_response_schema(self):
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY="test-key",
        )
        provider = GeminiLLMProvider(settings=settings)
        captured_payloads = []

        async def fake_post(url, **kwargs):
            captured_payloads.append(kwargs.get("json"))
            decision_data = {
                "detected_domain": "infra_builder",
                "detected_intent": "project_information",
                "extracted_slots": {},
                "proposed_stage": "information",
                "user_facing_response": "Hello",
                "knowledge_required": False,
                "action_proposed": False,
                "needs_clarification": False,
                "suggested_termination": False,
            }
            return httpx.Response(
                status_code=200,
                json={"candidates": [{"content": {"parts": [{"text": json.dumps(decision_data)}]}}]},
            )

        with patch("httpx.AsyncClient.post", side_effect=fake_post):
            await provider.generate_decision("Hello", "system prompt")

        assert len(captured_payloads) == 1
        assert "responseSchema" not in captured_payloads[0]["generationConfig"]

    @pytest.mark.asyncio
    async def test_decision_truncated_finish_reason_max_tokens_triggers_retry_and_raises_malformed(self):
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY="test-key",
        )
        provider = GeminiLLMProvider(settings=settings)
        attempt_count = 0

        async def fake_post(url, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            mock_payload = {
                "candidates": [
                    {
                        "finishReason": "MAX_TOKENS",
                        "content": {
                            "parts": [{"text": '{"detected_domain": "infra_builder", "user_facing_'}]
                        },
                    }
                ]
            }
            return httpx.Response(status_code=200, json=mock_payload)

        with patch("httpx.AsyncClient.post", side_effect=fake_post):
            with pytest.raises(LLMMalformedResponseError) as exc_info:
                await provider.generate_decision("Hello", "system prompt", max_retries=2)

        assert attempt_count == 3
        assert "finishReason=MAX_TOKENS" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_decision_truncated_output_cannot_execute_action(self):
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY="test-key",
        )
        provider = GeminiLLMProvider(settings=settings)

        mock_payload = {
            "candidates": [
                {
                    "finishReason": "MAX_TOKENS",
                    "content": {
                        "parts": [
                            {"text": '{"detected_domain": "infra_builder", "action_proposed": true, "proposed_action": {"tool_name": "send_message"'}
                        ]
                    },
                }
            ]
        }
        mock_resp = httpx.Response(status_code=200, json=mock_payload)

        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            with pytest.raises(LLMMalformedResponseError):
                await provider.generate_decision("Book site visit", "system prompt", max_retries=0)

    @pytest.mark.asyncio
    async def test_decision_retry_limits_preserved(self):
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY="test-key",
        )
        provider = GeminiLLMProvider(settings=settings)
        attempt_count = 0

        async def fake_post(url, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            return httpx.Response(
                status_code=200,
                json={"candidates": [{"content": {"parts": [{"text": "not valid json {"}]}}]},
            )

        with patch("httpx.AsyncClient.post", side_effect=fake_post):
            with pytest.raises(LLMMalformedResponseError):
                await provider.generate_decision("Hello", "system prompt", max_retries=2)

        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_decision_large_slot_dictionary_within_limit_succeeds(self):
        settings = Settings(
            _env_file=None,
            GEMINI_API_KEY="test-key",
        )
        provider = GeminiLLMProvider(settings=settings)

        decision_data = {
            "detected_domain": "infra_builder",
            "detected_intent": "project_information",
            "detected_sub_intent": "amenities_query",
            "extracted_slots": {
                "preferred_project": "Skyline Heights",
                "property_type": "apartment",
                "configuration": "3BHK",
                "budget": "1.5 Cr",
                "meeting_preference": "Tomorrow 3 PM",
                "name": "Jane Doe",
                "phone": "+919876543210",
                "email": "jane.doe@example.com",
            },
            "proposed_stage": "information",
            "user_facing_response": "Skyline Heights in Whitefield offers premium 3BHK residences with a clubhouse, pool, and sports facilities.",
            "knowledge_required": True,
            "knowledge_query": "Skyline Heights Whitefield amenities clubhouse pool",
            "action_proposed": False,
            "needs_clarification": False,
            "suggested_termination": False,
        }
        mock_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps(decision_data)}]
                    }
                }
            ]
        }
        mock_resp = httpx.Response(status_code=200, json=mock_payload)
        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            decision = await provider.generate_decision("Tell me about Skyline Heights 3BHK amenities", "system prompt")
            assert decision.detected_domain == DomainType.INFRA_BUILDER
            assert decision.detected_intent == "project_information"
            assert decision.extracted_slots["preferred_project"] == "Skyline Heights"
            assert decision.extracted_slots["configuration"] == "3BHK"
            assert decision.extracted_slots["name"] == "Jane Doe"


class TestMockLLMProvider:
    """Test MockLLMProvider functionality."""

    @pytest.mark.asyncio
    async def test_mock_canned_decision(self):
        canned = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="location_query",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Infra Builder properties are located across prime hubs in Bangalore.",
        )
        provider = MockLLMProvider(canned_decisions=[canned])

        decision = await provider.generate_decision("Where are your properties located?", "system")
        assert decision.detected_domain == DomainType.INFRA_BUILDER
        assert decision.detected_intent == "location_query"
        assert decision.user_facing_response == "Infra Builder properties are located across prime hubs in Bangalore."

    @pytest.mark.asyncio
    async def test_mock_streaming(self):
        provider = MockLLMProvider(canned_responses=["Hello from stream"])
        stream = provider.stream_response("test")
        chunks = [chunk async for chunk in stream]
        assert "".join(chunks).strip() == "Hello from stream"


class TestPhase5D3ASuggestedTerminationRemoval:
    """Focused tests for Phase 5D-3A verifying suggested_termination removal and safeguards."""

    @pytest.mark.asyncio
    async def test_decision_without_suggested_termination_validates_successfully(self):
        """Verify decisions generated without suggested_termination validate cleanly and default to False."""
        settings = Settings(_env_file=None, GEMINI_API_KEY="test-key")
        provider = GeminiLLMProvider(settings=settings)

        payload_without_term = {
            "detected_domain": "infra_builder",
            "detected_intent": "project_information",
            "extracted_slots": {"preferred_project": "Nature Valley"},
            "proposed_stage": "information",
            "user_facing_response": "Nature Valley offers serene villa plots near Sarjapur.",
            "knowledge_required": False,
            "action_proposed": False,
            "needs_clarification": False,
        }
        mock_resp = httpx.Response(
            status_code=200,
            json={"candidates": [{"content": {"parts": [{"text": json.dumps(payload_without_term)}]}}]},
        )
        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            decision = await provider.generate_decision("Tell me about Nature Valley", "system")
            assert decision.suggested_termination is False
            assert decision.detected_intent == "project_information"
            assert decision.user_facing_response == "Nature Valley offers serene villa plots near Sarjapur."

    @pytest.mark.asyncio
    async def test_decision_with_legacy_suggested_termination_retains_compatibility(self):
        """Verify legacy responses containing suggested_termination continue to parse without errors."""
        settings = Settings(_env_file=None, GEMINI_API_KEY="test-key")
        provider = GeminiLLMProvider(settings=settings)

        payload_with_legacy = {
            "detected_domain": "infra_builder",
            "detected_intent": "general_inquiry",
            "extracted_slots": {},
            "proposed_stage": "information",
            "user_facing_response": "Infra Builder specializes in premium residential plots.",
            "knowledge_required": False,
            "action_proposed": False,
            "needs_clarification": False,
            "suggested_termination": False,
        }
        mock_resp = httpx.Response(
            status_code=200,
            json={"candidates": [{"content": {"parts": [{"text": json.dumps(payload_with_legacy)}]}}]},
        )
        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            decision = await provider.generate_decision("Tell me about your company", "system")
            assert decision.suggested_termination is False
            assert decision.detected_intent == "general_inquiry"

    @pytest.mark.asyncio
    async def test_model_cannot_terminate_call_merely_by_proposing_completed_stage(self):
        """Verify that an LLM proposing proposed_stage=COMPLETED cannot terminate a call without explicit farewell."""
        from src.core.engine import ConversationEngine
        from src.state.manager import ConversationStateManager

        state_mgr = ConversationStateManager()
        state = state_mgr.create_inbound_state("term-test-1", "+15551234567", domain=DomainType.INFRA_BUILDER)

        # Canned decision where LLM suggests COMPLETED stage, but user said something non-terminating
        canned = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="project_information",
            proposed_stage=ConversationStage.COMPLETED,
            user_facing_response="No problem, let me know if you need anything else.",
            knowledge_required=False,
            action_proposed=False,
        )
        mock_llm = MockLLMProvider(canned_decisions=[canned])
        engine = ConversationEngine(llm_provider=mock_llm)

        result = await engine.process_user_turn(state, "No, I am looking for plots in Bangalore.")
        assert result.conversation_active is True
        assert result.termination_occurred is False
        assert state.termination_requested is False
        # Stage is safely reset to INFORMATION rather than terminating
        assert state.stage == ConversationStage.INFORMATION

    @pytest.mark.asyncio
    async def test_explicit_farewell_termination_still_terminates(self):
        """Verify explicit caller farewell triggers deterministic termination with synthetic decision."""
        from src.core.engine import ConversationEngine
        from src.state.manager import ConversationStateManager

        state_mgr = ConversationStateManager()
        state = state_mgr.create_inbound_state("term-test-2", "+15551234567", domain=DomainType.INFRA_BUILDER)

        mock_llm = MockLLMProvider()
        engine = ConversationEngine(llm_provider=mock_llm)

        result = await engine.process_user_turn(state, "Thank you, goodbye!")
        assert result.conversation_active is False
        assert result.termination_occurred is True
        assert state.termination_requested is True
        assert result.decision.suggested_termination is True
        assert result.decision.proposed_stage == ConversationStage.COMPLETED
        # Deterministic check bypassed calling LLM entirely
        assert len(mock_llm.invocations) == 0

    def test_synthetic_decision_and_ui_serialization_remain_valid(self):
        """Verify synthetic decision creation and serialization remain 100% valid with suggested_termination."""
        synthetic = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="end_call",
            proposed_stage=ConversationStage.COMPLETED,
            user_facing_response="Goodbye!",
            suggested_termination=True,
        )
        dumped = synthetic.model_dump()
        assert dumped["suggested_termination"] is True
        assert dumped["detected_domain"] == DomainType.INFRA_BUILDER

        json_str = synthetic.model_dump_json()
        assert '"suggested_termination":true' in json_str

    @pytest.mark.asyncio
    async def test_alternative_llm_provider_huggingface_generates_compatible_decision(self):
        """Verify HuggingFaceLLMProvider decision prompt omits suggested_termination and parses response."""
        from src.core.huggingface_llm import HuggingFaceLLMProvider

        hf_settings = Settings(_env_file=None, HF_API_KEY="test-hf-key")
        mock_hf_client = AsyncMock()
        hf_provider = HuggingFaceLLMProvider(settings=hf_settings, client=mock_hf_client)

        raw_hf_output = json.dumps({
            "detected_domain": "infra_builder",
            "detected_intent": "project_information",
            "extracted_slots": {"preferred_project": "JB Pride"},
            "proposed_stage": "information",
            "user_facing_response": "JB Pride is located near electronic city.",
            "knowledge_required": False,
            "action_proposed": False,
        })

        with patch.object(hf_provider, "generate_response", AsyncMock(return_value=raw_hf_output)):
            decision = await hf_provider.generate_decision("Tell me about JB Pride", "system prompt")
            assert decision.detected_domain == DomainType.INFRA_BUILDER
            assert decision.detected_intent == "project_information"
            assert decision.suggested_termination is False


class TestPhase5D3BDomainDerivation:
    """Verify Phase 5D-3B: detected_domain derivation from trusted application state."""

    def test_decision_without_detected_domain_defaults_to_infra_builder(self):
        """ConversationalDecision parses successfully without detected_domain and defaults to INFRA_BUILDER."""
        payload = {
            "detected_intent": "property_inquiry",
            "user_facing_response": "We have 2 and 3 BHK apartments available.",
        }
        decision = ConversationalDecision.model_validate(payload)
        assert decision.detected_domain == DomainType.INFRA_BUILDER
        assert decision.detected_intent == "property_inquiry"

    def test_decision_with_null_detected_domain_defaults_to_infra_builder(self):
        """ConversationalDecision parses with null detected_domain and normalizes to INFRA_BUILDER."""
        payload = {
            "detected_domain": None,
            "detected_intent": "property_inquiry",
            "user_facing_response": "We have 2 and 3 BHK apartments available.",
        }
        decision = ConversationalDecision.model_validate(payload)
        assert decision.detected_domain == DomainType.INFRA_BUILDER

    def test_decision_with_legacy_detected_domain_retains_compatibility(self):
        """ConversationalDecision continues to accept explicit detected_domain for backward compatibility."""
        payload = {
            "detected_domain": "infra_builder",
            "detected_intent": "property_inquiry",
            "user_facing_response": "We have 2 and 3 BHK apartments available.",
        }
        decision = ConversationalDecision.model_validate(payload)
        assert decision.detected_domain == DomainType.INFRA_BUILDER

    @pytest.mark.asyncio
    async def test_inbound_first_turn_and_locked_turn_domain_enforcement(self):
        """Inbound call enforces trusted backend domain regardless of LLM omitting detected_domain."""
        from src.core.engine import ConversationEngine
        from src.state.manager import ConversationStateManager

        mock_llm = MockLLMProvider(canned_responses=[json.dumps({
            "detected_intent": "project_information",
            "user_facing_response": "Welcome to Infra Builder!",
            "proposed_stage": "information",
        })])
        engine = ConversationEngine(llm_provider=mock_llm)
        state_mgr = ConversationStateManager()
        state = state_mgr.create_inbound_state("dom-test-1", "+15551234567")

        result = await engine.process_user_turn(state, "Hello, what projects do you have?")
        assert result.decision.detected_domain == DomainType.INFRA_BUILDER
        assert state.current_domain == DomainType.INFRA_BUILDER
        assert state.domain_locked is True

    @pytest.mark.asyncio
    async def test_outbound_domain_initialization_enforced(self):
        """Outbound call strictly enforces INFRA_BUILDER domain from state."""
        from src.core.engine import ConversationEngine
        from src.state.manager import ConversationStateManager

        mock_llm = MockLLMProvider(canned_responses=[json.dumps({
            "detected_intent": "project_information",
            "user_facing_response": "I am calling about Skyline Towers.",
            "proposed_stage": "information",
        })])
        engine = ConversationEngine(llm_provider=mock_llm)
        state_mgr = ConversationStateManager()
        state = state_mgr.create_outbound_state(
            call_id="dom-out-1",
            caller_phone="+15559876543",
            caller_name="Ananya Sharma",
            campaign_objective="Introduce Skyline Towers",
        )
        result = await engine.process_user_turn(state, "Hello?")
        assert result.decision.detected_domain == DomainType.INFRA_BUILDER
        assert state.current_domain == DomainType.INFRA_BUILDER

    def test_unknown_domain_fails_closed(self):
        """Unregistered domains raise DecisionValidationError and UNKNOWN domains suppress unauthorized actions."""
        from src.core.decision import DecisionValidator
        from src.core.errors import DecisionValidationError
        from src.domains.registry import DomainRegistry

        # 1. Unregistered domain raises DecisionValidationError
        custom_registry = DomainRegistry()
        custom_registry._domains.clear()
        validator = DecisionValidator(registry=custom_registry)
        decision = ConversationalDecision(
            user_facing_response="Test",
        )
        with pytest.raises(DecisionValidationError, match="Unknown domain detected"):
            validator.validate_and_filter(decision)

        # 2. UNKNOWN domain fails closed on actions and unsupported intents
        default_validator = DecisionValidator()
        decision_unknown = ConversationalDecision(
            detected_domain=DomainType.UNKNOWN,
            detected_intent="property_search",
            user_facing_response="I can check that for you.",
            action_proposed=True,
            proposed_action={"tool_name": "create_calendar_event"},
        )
        validated = default_validator.validate_and_filter(decision_unknown)
        assert validated.detected_domain == DomainType.UNKNOWN
        assert validated.action_proposed is False
        assert validated.proposed_action is None
        assert validated.detected_intent == "fallback"

    def test_domain_specific_tool_blocking_remains_effective_without_llm_domain(self):
        """Prohibited actions for INFRA_BUILDER (e.g. send_email) remain blocked when LLM omits detected_domain."""
        from src.core.decision import DecisionValidator

        validator = DecisionValidator()
        decision = ConversationalDecision.model_validate({
            "detected_intent": "property_search",
            "user_facing_response": "I will send an email.",
            "action_proposed": True,
            "proposed_action": {
                "tool_name": "send_email",
                "arguments": {"recipient": "user@example.com"},
            },
        })
        assert decision.detected_domain == DomainType.INFRA_BUILDER
        validated = validator.validate_and_filter(decision)
        assert validated.action_proposed is False
        assert validated.proposed_action is None

    @pytest.mark.asyncio
    async def test_huggingface_provider_omits_detected_domain_and_parses(self):
        """HuggingFace provider prompt omits detected_domain and parses decision without it."""
        from src.core.huggingface_llm import HuggingFaceLLMProvider

        hf_settings = Settings(_env_file=None, HF_API_KEY="test-hf-key")
        mock_hf_client = AsyncMock()
        hf_provider = HuggingFaceLLMProvider(settings=hf_settings, client=mock_hf_client)

        raw_hf_output = json.dumps({
            "detected_intent": "project_information",
            "extracted_slots": {"preferred_project": "Skyline Towers"},
            "proposed_stage": "information",
            "user_facing_response": "Skyline Towers offers 2 and 3 BHK residences.",
            "knowledge_required": False,
            "action_proposed": False,
        })

        with patch.object(hf_provider, "generate_response", AsyncMock(return_value=raw_hf_output)) as mock_gen:
            decision = await hf_provider.generate_decision("Tell me about Skyline Towers", "system prompt")
            assert decision.detected_domain == DomainType.INFRA_BUILDER
            assert decision.detected_intent == "project_information"

            call_args = mock_gen.call_args[1]
            assert '"detected_domain"' not in call_args["system_instruction"]

    def test_synthetic_decision_and_ui_serialization_retain_domain(self):
        """Synthetic decisions and serialization preserve detected_domain for UI consumers."""
        decision = ConversationalDecision(
            user_facing_response="Hello!",
        )
        assert decision.detected_domain == DomainType.INFRA_BUILDER
        dumped = decision.model_dump()
        assert dumped["detected_domain"] == DomainType.INFRA_BUILDER
        dumped_json = decision.model_dump_json()
        assert '"detected_domain":"infra_builder"' in dumped_json


class TestPhase5D3CActionProposedDerivation:
    """Verify Phase 5D-3C: action_proposed derivation from proposed_action."""

    def test_decision_without_action_proposed_with_valid_proposed_action(self):
        """ConversationalDecision parses without action_proposed and derives True from proposed_action."""
        payload = {
            "detected_intent": "site_visit_request",
            "user_facing_response": "Let me check available slots for Skyline Towers.",
            "proposed_action": {
                "tool_name": "find_available_slots",
                "arguments": {"project_name": "Skyline Towers"},
            },
        }
        decision = ConversationalDecision.model_validate(payload)
        assert decision.action_proposed is True
        assert decision.proposed_action is not None
        assert decision.proposed_action.tool_name == "find_available_slots"

    def test_decision_without_either_action_signal(self):
        """ConversationalDecision parses without any action fields and defaults to False/None."""
        payload = {
            "detected_intent": "greeting",
            "user_facing_response": "Hello, how can I assist you today?",
        }
        decision = ConversationalDecision.model_validate(payload)
        assert decision.action_proposed is False
        assert decision.proposed_action is None

    def test_legacy_payloads_containing_action_proposed(self):
        """Legacy payloads containing action_proposed remain compatible while preserving proposed_action truth."""
        # Legacy positive
        p1 = {
            "action_proposed": True,
            "proposed_action": {"tool_name": "find_available_slots", "arguments": {}},
            "user_facing_response": "Checking slots.",
        }
        d1 = ConversationalDecision.model_validate(p1)
        assert d1.action_proposed is True
        assert d1.proposed_action is not None

        # Legacy negative
        p2 = {
            "action_proposed": False,
            "proposed_action": None,
            "user_facing_response": "Hello.",
        }
        d2 = ConversationalDecision.model_validate(p2)
        assert d2.action_proposed is False
        assert d2.proposed_action is None

        # Legacy inconsistency: action_proposed=True but proposed_action=None -> derives False
        p3 = {
            "action_proposed": True,
            "proposed_action": None,
            "user_facing_response": "Hello.",
        }
        d3 = ConversationalDecision.model_validate(p3)
        assert d3.action_proposed is False
        assert d3.proposed_action is None

    def test_unsupported_intents_suppressing_proposed_actions(self):
        """Unsupported intents suppress proposed_action and reset action_proposed to False."""
        from src.core.decision import DecisionValidator

        validator = DecisionValidator()
        decision = ConversationalDecision.model_validate({
            "detected_intent": "completely_unsupported_intent_xyz",
            "user_facing_response": "Let me assist you.",
            "proposed_action": {
                "tool_name": "find_available_slots",
                "arguments": {},
            },
        })
        assert decision.action_proposed is True
        validated = validator.validate_and_filter(decision)
        assert validated.proposed_action is None
        assert validated.action_proposed is False
        assert validated.detected_intent == "fallback"

    def test_invalid_tool_names_and_arguments_remaining_rejected(self):
        """Invalid or domain-blocked tools are suppressed and reset action_proposed to False."""
        from src.core.decision import DecisionValidator

        validator = DecisionValidator()
        # Invalid tool name
        d_invalid = ConversationalDecision.model_validate({
            "detected_intent": "property_search",
            "user_facing_response": "Let me assist.",
            "proposed_action": {"tool_name": "unregistered_tool_name"},
        })
        assert d_invalid.action_proposed is True
        val_invalid = validator.validate_and_filter(d_invalid)
        assert val_invalid.proposed_action is None
        assert val_invalid.action_proposed is False

        # Blocked tool for INFRA_BUILDER
        d_blocked = ConversationalDecision.model_validate({
            "detected_intent": "property_search",
            "user_facing_response": "Let me email that.",
            "proposed_action": {
                "tool_name": "send_email",
                "arguments": {"recipient": "user@example.com"},
            },
        })
        assert d_blocked.action_proposed is True
        val_blocked = validator.validate_and_filter(d_blocked)
        assert val_blocked.proposed_action is None
        assert val_blocked.action_proposed is False

    def test_authorization_and_whatsapp_consent_gates(self):
        """WhatsApp refusal suppresses proposed_action and resets action_proposed to False."""
        from src.core.decision import DecisionValidator

        validator = DecisionValidator()
        decision = ConversationalDecision.model_validate({
            "detected_intent": "property_search",
            "user_facing_response": "Understood, no WhatsApp.",
            "extracted_slots": {"whatsapp_consent": False},
            "proposed_action": {
                "tool_name": "send_message",
                "arguments": {"recipient": "+919876543210"},
            },
        })
        assert decision.action_proposed is True
        validated = validator.validate_and_filter(decision)
        assert validated.proposed_action is None
        assert validated.action_proposed is False

    @pytest.mark.asyncio
    async def test_proposed_action_not_treated_as_executed_before_verification(self):
        """Proposed action is strictly distinct from executed action; state remains unmutated without execution."""
        from src.core.engine import ConversationEngine
        from src.state.manager import ConversationStateManager

        mock_llm = MockLLMProvider(canned_responses=[json.dumps({
            "detected_intent": "site_visit_request",
            "user_facing_response": "I can check available slots for you.",
            "proposed_stage": "action_confirmation",
            "proposed_action": {
                "tool_name": "find_available_slots",
                "arguments": {"project_name": "Skyline Towers"},
            },
        })])
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=None)
        state_mgr = ConversationStateManager()
        state = state_mgr.create_inbound_state("act-prop-1", "+15551234567")

        result = await engine.process_user_turn(state, "Can I visit Skyline Towers tomorrow?")
        assert result.conversation_active is True
        assert state.last_action is None
        assert state.last_tool_result is None

    @pytest.mark.asyncio
    async def test_gemini_and_huggingface_provider_contracts(self):
        """Prompt contracts for both Gemini and HuggingFace omit action_proposed and parse without it."""
        from src.core.prompts import PromptSynthesizer
        from src.domains import INFRA_BUILDER_DOMAIN_CONFIG
        from src.core.huggingface_llm import HuggingFaceLLMProvider
        from src.state.manager import ConversationStateManager

        # 1. Gemini / PromptSynthesizer system prompt omits action_proposed
        synth = PromptSynthesizer()
        state_mgr = ConversationStateManager()
        st = state_mgr.create_inbound_state("contract-test-1", "+15551234567")
        sys_prompt = synth.build_system_instruction(st, INFRA_BUILDER_DOMAIN_CONFIG)
        assert '"action_proposed"' not in sys_prompt

        # 2. Hugging Face provider prompt omits action_proposed and parses without it
        hf_settings = Settings(_env_file=None, HF_API_KEY="test-hf-key")
        mock_hf_client = AsyncMock()
        hf_provider = HuggingFaceLLMProvider(settings=hf_settings, client=mock_hf_client)

        raw_hf_output = json.dumps({
            "detected_intent": "site_visit_request",
            "proposed_stage": "action_confirmation",
            "user_facing_response": "Checking slots.",
            "proposed_action": {
                "tool_name": "find_available_slots",
                "arguments": {},
            },
        })
        with patch.object(hf_provider, "generate_response", AsyncMock(return_value=raw_hf_output)) as mock_gen:
            dec = await hf_provider.generate_decision("Book visit", "system prompt")
            assert dec.action_proposed is True
            assert dec.proposed_action.tool_name == "find_available_slots"
            call_args = mock_gen.call_args[1]
            assert '"action_proposed"' not in call_args["system_instruction"]

    def test_synthetic_decision_and_ui_serialization(self):
        """Synthetic decisions and serialization preserve action_proposed for UI consumers."""
        dec_with_act = ConversationalDecision(
            user_facing_response="Sending details.",
            proposed_action=ProposedAction(tool_name="send_message"),
        )
        assert dec_with_act.action_proposed is True
        dumped_with = dec_with_act.model_dump()
        assert dumped_with["action_proposed"] is True
        assert '"action_proposed":true' in dec_with_act.model_dump_json()

        dec_no_act = ConversationalDecision(
            user_facing_response="Hello.",
        )
        assert dec_no_act.action_proposed is False
        dumped_no = dec_no_act.model_dump()
        assert dumped_no["action_proposed"] is False
        assert '"action_proposed":false' in dec_no_act.model_dump_json()

    def test_direct_mutation_of_proposed_action_immediately_updates_flag(self):
        """Clearing or adding proposed_action directly updates action_proposed without drift."""
        dec = ConversationalDecision(
            user_facing_response="Checking slots.",
            proposed_action=ProposedAction(tool_name="find_available_slots"),
        )
        assert dec.action_proposed is True

        # Clearing proposed_action immediately makes action_proposed False
        dec.proposed_action = None
        assert dec.action_proposed is False

        # Adding proposed_action makes action_proposed True without a separate boolean update
        dec.proposed_action = ProposedAction(tool_name="send_message")
        assert dec.action_proposed is True

    def test_model_copy_update_maintains_consistent_action_flag(self):
        """model_copy(update={'proposed_action': None}) produces consistent False action_proposed."""
        dec = ConversationalDecision(
            user_facing_response="Checking slots.",
            proposed_action=ProposedAction(tool_name="find_available_slots"),
        )
        assert dec.action_proposed is True

        # Copy with proposed_action cleared
        dec_cleared = dec.model_copy(update={"proposed_action": None})
        assert dec_cleared.proposed_action is None
        assert dec_cleared.action_proposed is False

        # Copy with proposed_action added
        dec_empty = ConversationalDecision(user_facing_response="Hello.")
        assert dec_empty.action_proposed is False
        dec_added = dec_empty.model_copy(update={"proposed_action": ProposedAction(tool_name="send_message")})
        assert dec_added.proposed_action is not None
        assert dec_added.action_proposed is True

    def test_compatibility_setter_clears_action_and_never_fabricates_action(self):
        """Setting action_proposed=False clears proposed_action; setting True never fabricates an action."""
        dec = ConversationalDecision(
            user_facing_response="Checking slots.",
            proposed_action=ProposedAction(tool_name="find_available_slots"),
        )
        assert dec.action_proposed is True
        assert dec.proposed_action is not None

        # Setting action_proposed = False clears proposed_action
        dec.action_proposed = False
        assert dec.proposed_action is None
        assert dec.action_proposed is False

        # Setting action_proposed = True on None must NOT fabricate or authorize an action
        dec.action_proposed = True
        assert dec.proposed_action is None
        assert dec.action_proposed is False


class TestPhase5D3FDetectedSubIntentRemoval:
    """Verify Phase 5D-3F: Removal of detected_sub_intent from LLM output contract."""

    def test_gemini_and_huggingface_output_instructions_omit_detected_sub_intent(self):
        """Gemini and Hugging Face prompt instructions omit detected_sub_intent."""
        from src.core.prompts import PromptSynthesizer
        from src.domains import INFRA_BUILDER_DOMAIN_CONFIG
        from src.state.manager import ConversationStateManager
        from pathlib import Path

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-sub-1", "+15551234567", domain=DomainType.INFRA_BUILDER)
        synthesizer = PromptSynthesizer()
        system_prompt = synthesizer.build_system_instruction(state, INFRA_BUILDER_DOMAIN_CONFIG)

        # Gemini instructions (loaded via json_schema.txt and decision_instructions.txt)
        assert "detected_sub_intent" not in system_prompt
        assert "detected_intent" in system_prompt

        # Hugging Face provider contract
        hf_source = Path("src/core/huggingface_llm.py").read_text(encoding="utf-8")
        assert '"detected_sub_intent"' not in hf_source
        assert '"detected_intent"' in hf_source

    def test_model_response_omitting_sub_intent_defaults_to_none(self):
        """A model response omitting detected_sub_intent parses cleanly and defaults to None."""
        payload = {
            "detected_intent": "property_search",
            "extracted_slots": {"preferred_project": "Skyline Towers"},
            "proposed_stage": "information",
            "user_facing_response": "We have 2 and 3 BHK units at Skyline Towers.",
            "knowledge_required": False,
            "knowledge_query": None,
            "proposed_action": None,
            "needs_clarification": False,
        }
        dec = ConversationalDecision.model_validate(payload)
        assert dec.detected_intent == "property_search"
        assert dec.detected_sub_intent is None

    def test_legacy_payload_with_sub_intent_remains_accepted(self):
        """Legacy payloads containing detected_sub_intent remain accepted without error."""
        legacy_payload = {
            "detected_intent": "property_search",
            "detected_sub_intent": "amenities_query",
            "extracted_slots": {},
            "proposed_stage": "information",
            "user_facing_response": "Amenities include a pool and gym.",
            "knowledge_required": False,
            "knowledge_query": None,
            "proposed_action": None,
            "needs_clarification": False,
        }
        dec = ConversationalDecision.model_validate(legacy_payload)
        assert dec.detected_intent == "property_search"
        assert dec.detected_sub_intent == "amenities_query"

    def test_state_consumers_operate_cleanly_when_sub_intent_is_none(self):
        """State manager and set_intent handle sub_intent=None without issue."""
        from src.state.manager import ConversationStateManager
        from src.core.prompts import PromptSynthesizer
        from src.domains import INFRA_BUILDER_DOMAIN_CONFIG

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-sub-2", "+15551234567", domain=DomainType.INFRA_BUILDER)
        state.set_intent("property_search", sub_intent=None)
        assert state.current_intent == "property_search"
        assert state.current_sub_intent is None

        # Ensure prompt synthesis omits obsolete sub-intent context
        synthesizer = PromptSynthesizer()
        user_prompt = synthesizer.build_user_prompt(state, "Tell me about prices.")
        assert "Current sub-intent" not in user_prompt
        system_prompt = synthesizer.build_system_instruction(state, INFRA_BUILDER_DOMAIN_CONFIG)
        assert "Sub-intent:" not in system_prompt

    def test_no_other_required_decision_fields_were_removed(self):
        """All other required decision fields remain present in the system prompt."""
        from src.core.prompts import PromptSynthesizer
        from src.domains import INFRA_BUILDER_DOMAIN_CONFIG
        from src.state.manager import ConversationStateManager

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-sub-3", "+15551234567", domain=DomainType.INFRA_BUILDER)
        synthesizer = PromptSynthesizer()
        system_prompt = synthesizer.build_system_instruction(state, INFRA_BUILDER_DOMAIN_CONFIG)

        assert "detected_intent" in system_prompt
        assert "extracted_slots" in system_prompt
        assert "proposed_stage" in system_prompt
        assert "user_facing_response" in system_prompt
        assert "knowledge_required" in system_prompt
        assert "knowledge_query" in system_prompt
        assert "proposed_action" in system_prompt
        assert "needs_clarification" in system_prompt



