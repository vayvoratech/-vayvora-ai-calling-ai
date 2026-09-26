"""Gemini 3.5 Flash LLM provider and mock testing implementations.

Implements the LLMProvider interface using async httpx requests with streaming,
structured JSON decision parsing, rate-limiting safeguards, and bounded retries.
"""

import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx

from src.config import Settings, get_settings
from src.core.decision import ConversationalDecision, DecisionValidator
from src.core.errors import (
    LLMAuthenticationError,
    LLMError,
    LLMMalformedResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.core.interfaces import LLMProvider
from src.logging import get_logger

logger = get_logger("core.llm")


def extract_json_block(text: str) -> str:
    """Extract clean JSON content from potentially fenced LLM responses."""
    cleaned = text.strip()
    # Match markdown json block if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Or find outermost braces
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1].strip()
    return cleaned


class GeminiLLMProvider(LLMProvider):
    """Google Gemini 3.5 Flash provider communicating via async HTTP REST/SSE."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.gemini_model
        self.timeout = self.settings.gemini_timeout_seconds
        self.temperature = self.settings.gemini_temperature
        self._client = client
        self.validator = DecisionValidator()

    @property
    def api_key(self) -> str:
        """Fetch and validate Gemini API key without logging."""
        if not self.settings.gemini_api_key:
            raise LLMAuthenticationError(
                "Gemini API key is not configured. Set GEMINI_API_KEY in environment or .env."
            )
        return self.settings.gemini_api_key.get_secret_value()

    def _get_base_url(self) -> str:
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}"

    async def _send_request(
        self, payload: Dict[str, Any], stream: bool = False
    ) -> httpx.Response:
        """Internal helper to dispatch HTTP requests with standardized error handling."""
        action = "streamGenerateContent?alt=sse" if stream else "generateContent"
        url = f"{self._get_base_url()}:{action}"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        # Mask API key in any debug logs
        logger.debug("Dispatching request to Gemini model: %s", self.model)

        max_attempts = 4
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(max_attempts):
                try:
                    response = await client.post(url, json=payload, headers=headers)
                except httpx.TimeoutException as exc:
                    logger.error("Gemini API request timed out after %s seconds", self.timeout)
                    raise LLMTimeoutError(f"Request to Gemini timed out: {exc}") from exc
                except httpx.RequestError as exc:
                    logger.error("Network error communicating with Gemini API: %s", type(exc).__name__)
                    raise LLMError(f"Network error during Gemini API call: {exc}") from exc

                # Resilient fallback if selected model is unavailable or overloaded (e.g. 503 high demand or 404 deprecation)
                if response.status_code in (503, 404, 500) and self.model != "gemini-3.5-flash-lite":
                    logger.warning(
                        "Gemini model '%s' returned HTTP %d. Attempting fallback to 'gemini-3.5-flash-lite'...",
                        self.model,
                        response.status_code,
                    )
                    fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:{action}"
                    try:
                        fallback_resp = await client.post(fallback_url, json=payload, headers=headers)
                        if fallback_resp.status_code < 400:
                            return fallback_resp
                    except Exception as fb_exc:
                        logger.debug("Fallback to gemini-3.5-flash-lite failed: %s", fb_exc)

                if response.status_code == 401 or response.status_code == 403:
                    logger.error("Authentication failed for Gemini API (HTTP %d)", response.status_code)
                    raise LLMAuthenticationError("Gemini API authentication failed. Verify API key.")
                elif response.status_code == 429:
                    if attempt < max_attempts - 1:
                        sleep_time = 2.5 * (attempt + 1)
                        logger.warning(
                            "Gemini API rate limit (429) hit. Backing off for %.1fs (attempt %d/%d)...",
                            sleep_time,
                            attempt + 1,
                            max_attempts,
                        )
                        await asyncio.sleep(sleep_time)
                        continue
                    logger.warning("Gemini API rate limit exceeded after retries (HTTP 429)")
                    raise LLMRateLimitError("Gemini API rate limit reached. Retry later.")
                elif response.status_code >= 400:
                    logger.error("Gemini API error (HTTP %d)", response.status_code)
                    raise LLMError(f"Gemini API returned HTTP {response.status_code}: {response.text}")

                return response

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a complete text response using Gemini 3.5 Flash."""
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
        if tools:
            payload["tools"] = tools

        response = await self._send_request(payload, stream=False)
        data = response.json()

        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise LLMMalformedResponseError("No candidates returned by Gemini.")
            text = candidates[0]["content"]["parts"][0]["text"]
            return text
        except (KeyError, IndexError) as exc:
            raise LLMMalformedResponseError(f"Malformed response payload from Gemini: {data}") from exc

    async def stream_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[str]:
        """Stream response chunks incrementally via Server-Sent Events (SSE)."""
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if tools:
            payload["tools"] = tools

        action = "streamGenerateContent?alt=sse"
        url = f"{self._get_base_url()}:{action}"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        content = await response.aread()
                        raise LLMError(f"Streaming error HTTP {response.status_code}: {content.decode('utf-8')}")

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            raw_json = line[6:].strip()
                            if raw_json == "[DONE]":
                                break
                            try:
                                chunk_data = json.loads(raw_json)
                                candidates = chunk_data.get("candidates", [])
                                if candidates and "content" in candidates[0]:
                                    part = candidates[0]["content"]["parts"][0]["text"]
                                    yield part
                            except Exception:
                                continue
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError("Gemini streaming request timed out.") from exc

    async def generate_decision(
        self,
        prompt: str,
        system_instruction: str,
        max_retries: int = 2,
    ) -> ConversationalDecision:
        """Generate, parse, and validate a structured ConversationalDecision with retry."""
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
            },
        }

        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                response = await self._send_request(payload, stream=False)
                data = response.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                clean_json = extract_json_block(text)
                decision_dict = json.loads(clean_json)
                decision = ConversationalDecision.model_validate(decision_dict)
                return self.validator.validate_and_filter(decision)
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                last_error = LLMMalformedResponseError(f"Could not parse valid JSON from LLM: {exc}")
                logger.warning("Malformed JSON on attempt %d: %s", attempt + 1, exc)
            except Exception as exc:
                last_error = exc
                logger.warning("Decision generation error on attempt %d: %s", attempt + 1, exc)

        raise last_error or LLMMalformedResponseError("Failed to generate a valid conversational decision.")


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider for unit testing without API keys or network calls."""

    def __init__(
        self,
        canned_responses: Optional[List[str]] = None,
        canned_decisions: Optional[List[ConversationalDecision]] = None,
    ) -> None:
        self.canned_responses = list(canned_responses or [])
        self.canned_decisions = list(canned_decisions or [])
        self.invocations: List[Dict[str, Any]] = []
        self.validator = DecisionValidator()

    def add_response(self, response: str) -> None:
        """Queue a canned raw text response."""
        self.canned_responses.append(response)

    def add_decision(self, decision: ConversationalDecision) -> None:
        """Queue a canned structured ConversationalDecision."""
        self.canned_decisions.append(decision)

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        self.invocations.append({
            "prompt": prompt,
            "system_instruction": system_instruction,
            "tools": tools,
        })
        if self.canned_responses:
            return self.canned_responses.pop(0)
        return '{"user_facing_response": "Mock default response."}'

    async def stream_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[str]:
        response = await self.generate_response(prompt, system_instruction, tools)
        words = response.split(" ")
        for word in words:
            yield word + " "

    async def generate_decision(
        self,
        prompt: str,
        system_instruction: str,
        max_retries: int = 1,
    ) -> ConversationalDecision:
        self.invocations.append({
            "prompt": prompt,
            "system_instruction": system_instruction,
        })
        if self.canned_decisions:
            decision = self.canned_decisions.pop(0)
            return self.validator.validate_and_filter(decision)

        if self.canned_responses:
            raw = self.canned_responses.pop(0)
            clean_json = extract_json_block(raw)
            data = json.loads(clean_json)
            decision = ConversationalDecision.model_validate(data)
            return self.validator.validate_and_filter(decision)

        # Default fallback decision
        from src.core.types import ConversationStage, DomainType

        clean_user_text = ""
        if "Latest Caller Message:" in prompt:
            chunk = prompt.split("Latest Caller Message:")[-1]
            if "Analyze the latest caller message" in chunk:
                chunk = chunk.split("Analyze the latest caller message")[0]
            clean_user_text = chunk.strip().strip('"').strip().lower().rstrip(".!?")

        if clean_user_text in ("hello", "hi", "hey", "hello there", "hi there", "good morning", "good afternoon"):
            is_edusaas = "edusaas" in prompt.lower()
            decision = ConversationalDecision(
                detected_domain=DomainType.EDUSAAS if is_edusaas else DomainType.VAYVORA,
                detected_intent="greeting",
                proposed_stage=ConversationStage.GREETING,
                user_facing_response="Hi, how can I help you today?",
            )
            return self.validator.validate_and_filter(decision)

        if any(w in clean_user_text for w in ["corporate", "solutions", "enterprise", "vayvora", "custom ai"]):
            decision = ConversationalDecision(
                detected_domain=DomainType.VAYVORA,
                detected_intent="solutions_inquiry",
                proposed_stage=ConversationStage.INFORMATION,
                knowledge_required=True,
                knowledge_query=clean_user_text or "enterprise AI solutions",
                user_facing_response="Vayvora provides enterprise AI solutions, autonomous agent workflows, and intelligent software engineering.",
            )
            return self.validator.validate_and_filter(decision)

        if any(w in clean_user_text for w in ["course", "courses", "curriculum", "learn", "syllabus", "admissions", "degree"]):
            decision = ConversationalDecision(
                detected_domain=DomainType.EDUSAAS,
                detected_intent="course_information",
                proposed_stage=ConversationStage.INFORMATION,
                knowledge_required=True,
                knowledge_query=clean_user_text or "available courses curriculum",
                user_facing_response="At EduSaaS, we offer comprehensive programs in AI Engineering, Data Science, and Machine Learning.",
            )
            return self.validator.validate_and_filter(decision)

        decision = ConversationalDecision(
            detected_domain=DomainType.GENERAL,
            detected_intent="greeting",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Hello! How can I assist you with EduSaaS courses or Vayvora software services today?",
        )
        return self.validator.validate_and_filter(decision)
