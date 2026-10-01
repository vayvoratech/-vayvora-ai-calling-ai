"""Optional live smoke test for Google Gemini 3.5 Flash API.

This test is automatically skipped unless a valid GEMINI_API_KEY environment
variable is configured. It is clearly distinguished from offline mocked tests.
"""

import os
import pytest
from dotenv import load_dotenv

from src.config import Settings
from src.core.decision import ConversationalDecision
from src.core.llm import GeminiLLMProvider
from src.core.types import DomainType

load_dotenv()


def is_live_key_configured() -> bool:
    """Check if a non-placeholder GEMINI_API_KEY is present in the environment."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return False
    placeholder_signatures = ["your_gemini_api_key_here", "test-key", "placeholder"]
    return not any(sig in key.lower() for sig in placeholder_signatures)


@pytest.mark.skipif(
    not is_live_key_configured(),
    reason="Live Gemini API key not configured in environment. Skipping live smoke test.",
)
@pytest.mark.asyncio
async def test_live_gemini_decision_smoke():
    """Optional live integration smoke test with real Gemini 3.5 Flash API."""
    settings = Settings()
    provider = GeminiLLMProvider(settings=settings)

    prompt = 'Caller says: "Hi, I want to learn about your data science and AI courses."'
    system = "You are an assistant for EduSaaS. Output strict JSON with detected_domain, detected_intent, and user_facing_response."

    decision: ConversationalDecision = await provider.generate_decision(prompt, system)

    assert decision is not None
    assert decision.user_facing_response is not None
    assert len(decision.user_facing_response) > 5
    assert decision.detected_domain in (DomainType.EDUSAAS, DomainType.GENERAL)
