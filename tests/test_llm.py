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
            GEMINI_MODEL="gemini-3.5-flash",
        )
        provider = GeminiLLMProvider(settings=settings)
        assert provider.model == "gemini-3.5-flash"

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
            "detected_domain": "edusaas",
            "detected_intent": "course_information",
            "extracted_slots": {"target_course": "AI Engineering"},
            "proposed_stage": "information",
            "user_facing_response": "We have an extensive AI Engineering curriculum.",
            "knowledge_required": True,
            "knowledge_query": "AI Engineering syllabus prerequisites",
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
            decision = await provider.generate_decision("Tell me about AI course", "system prompt")
            assert decision.detected_domain == DomainType.EDUSAAS
            assert decision.detected_intent == "course_information"
            assert decision.knowledge_required is True


class TestMockLLMProvider:
    """Test MockLLMProvider functionality."""

    @pytest.mark.asyncio
    async def test_mock_canned_decision(self):
        canned = ConversationalDecision(
            detected_domain=DomainType.VAYVORA,
            detected_intent="company_location",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Vayvora is headquartered in Bangalore, India.",
        )
        provider = MockLLMProvider(canned_decisions=[canned])

        decision = await provider.generate_decision("Where are you located?", "system")
        assert decision.detected_domain == DomainType.VAYVORA
        assert decision.detected_intent == "company_location"
        assert decision.user_facing_response == "Vayvora is headquartered in Bangalore, India."

    @pytest.mark.asyncio
    async def test_mock_streaming(self):
        provider = MockLLMProvider(canned_responses=["Hello from stream"])
        stream = provider.stream_response("test")
        chunks = [chunk async for chunk in stream]
        assert "".join(chunks).strip() == "Hello from stream"
