"""Hugging Face / Qwen LLM provider.

Uses the OpenAI-compatible Hugging Face Router API while preserving
the existing LLMProvider and ConversationalDecision contracts.
"""

import json
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI

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
from pydantic import ValidationError

logger = get_logger("core.huggingface_llm")


def extract_json_block(text: str) -> str:
    """Extract JSON from a normal or markdown-fenced LLM response."""

    cleaned = text.strip()

    match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        cleaned,
        re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1].strip()

    return cleaned


class HuggingFaceLLMProvider(LLMProvider):
    """Qwen provider through the Hugging Face OpenAI-compatible API."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        self.settings = settings or get_settings()

        self.model = self.settings.hf_model
        self.base_url = self.settings.hf_base_url
        self.timeout = self.settings.hf_timeout_seconds
        self.temperature = self.settings.hf_temperature

        self.validator = DecisionValidator()

        api_key = self._get_api_key()

        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _get_api_key(self) -> str:
        """Return the HF API key without logging it."""

        if not self.settings.hf_api_key:
            raise LLMAuthenticationError(
                "Hugging Face API key is not configured. "
                "Set HF_API_KEY in .env."
            )

        return self.settings.hf_api_key.get_secret_value()

    def _build_messages(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build OpenAI-compatible chat messages."""

        messages: List[Dict[str, str]] = []

        if system_instruction:
            messages.append(
                {
                    "role": "system",
                    "content": system_instruction,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return messages

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a complete response from Qwen."""

        messages = self._build_messages(
            prompt=prompt,
            system_instruction=system_instruction,
        )

        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }

            if tools:
                kwargs["tools"] = tools

            response = await self._client.chat.completions.create(**kwargs)

            if not response.choices:
                raise LLMMalformedResponseError(
                    "Hugging Face returned no choices."
                )

            content = response.choices[0].message.content

            if not content:
                raise LLMMalformedResponseError(
                    "Hugging Face returned an empty response."
                )

            return content

        except LLMMalformedResponseError:
            raise

        except Exception as exc:
            self._raise_provider_error(exc)

            raise LLMError(
                f"Unexpected Hugging Face API error: {exc}"
            ) from exc

    async def stream_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[str]:
        """Stream Qwen response tokens."""

        messages = self._build_messages(
            prompt=prompt,
            system_instruction=system_instruction,
        )

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }

        if tools:
            kwargs["tools"] = tools

        try:
            stream = await self._client.chat.completions.create(
                **kwargs
            )

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                content = delta.content

                if content:
                    yield content

        except Exception as exc:
            self._raise_provider_error(exc)

            raise LLMError(
                f"Hugging Face streaming error: {exc}"
            ) from exc

    async def generate_decision(
        self,
        prompt: str,
        system_instruction: str,
        max_retries: int = 2,
    ) -> ConversationalDecision:
        """Generate and validate a structured conversational decision."""

        structured_instruction = f"""
            {system_instruction}

            You are generating a structured decision for a production conversational
            AI voice agent.

            RETURN ONLY VALID JSON.

            Do not return markdown.
            Do not use ```json.
            Do not add explanations.
            Do not add comments.
            Do not invent enum values.

            The JSON object MUST contain:

            "detected_domain":
                exactly one of:
                "edusaas"
                "vayvora"
                "general"

            "proposed_stage":
                exactly one of:
                "greeting"
                "identity"
                "purpose_discovery"
                "discovery"
                "information"
                "recommendation"
                "objection_handling"
                "action_confirmation"
                "follow_up"
                "closing"
                "completed"

            "detected_intent":
                a short lowercase intent identifier appropriate for the detected domain,
                or null if no specific intent can be determined.

            "detected_sub_intent":
                a short lowercase sub-intent identifier, or null.

            "extracted_slots":
                JSON object containing only entities actually present in the conversation.

            "knowledge_required":
                true only when factual information from the company's knowledge base is
                required.

            "knowledge_query":
                when knowledge_required is true, provide a concise, retrieval-optimized search query.
                Rules:
                - Preserve the caller's actual information need.
                - Include domain, product, service, and technical keywords implied by the caller.
                - Do NOT copy conversational wording (e.g. avoid 'What AI solutions does Vayvora provide?').
                - Do NOT invent facts or products.
                - If knowledge_required is false, set to null.

            "action_proposed":
                true only when an external action is genuinely needed and recipient details are available.

            "proposed_action":
                an action object or null.
                If present, "tool_name" MUST be strictly one of:
                "send_email"
                "find_available_slots"
                "create_calendar_event"
                "update_lead"
                "create_hr_followup"
                "send_message"
                "update_business_status"
                Never invent tool names like "connect_advisor". For advisor callbacks or consultations, use "create_hr_followup" (requires caller contact details) or assist conversationally.
                Never claim that an action has already executed.

            "needs_clarification":
                true only when the caller's request is genuinely ambiguous.

            "clarification_question":
                clarification question or null.

            "suggested_termination":
                true only when the caller explicitly indicates they want to end
                the conversation.

            "user_facing_response":
                the natural response that should be given to the caller.

            IMPORTANT DOMAIN RULES:

            - Vayvora is represented only by "vayvora".
            - EduSaaS is represented only by "edusaas".
            - General/unknown requests use "general".
            - Never use descriptions such as "Technology/AI Services" as a domain.
            - Never use descriptions such as "Initial information request" as a stage.
            - Use the exact enum strings above.

            EXAMPLE:

            {{
                "detected_domain": "vayvora",
                "detected_intent": "ai_solution",
                "detected_sub_intent": "general_ai_solution",
                "extracted_slots": {{}},
                "proposed_stage": "information",
                "user_facing_response": "Vayvora provides enterprise AI solutions, autonomous agent workflows, and intelligent software engineering.",
                "knowledge_required": true,
                "knowledge_query": "Vayvora enterprise AI solutions RAG structured entity extraction",
                "action_proposed": false,
                "proposed_action": null,
                "needs_clarification": false,
                "clarification_question": null,
                "suggested_termination": false
            }}
            """
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                raw_response = await self.generate_response(
                    prompt=prompt,
                    system_instruction=structured_instruction,
                )

                clean_json = extract_json_block(raw_response)

                decision_dict = json.loads(clean_json)

                decision = ConversationalDecision.model_validate(
                    decision_dict
                )

                return self.validator.validate_and_filter(decision)

            except json.JSONDecodeError as exc:
                last_error = LLMMalformedResponseError(
                    f"Could not parse JSON from Qwen response: {exc}"
                )

                logger.warning(
                    "Malformed Qwen JSON on attempt %d",
                    attempt + 1,
                )

            except ValidationError as exc:
                last_error = LLMMalformedResponseError(
                    f"Qwen returned an invalid ConversationalDecision: {exc}"
                )

                logger.warning(
                    "Qwen decision validation failed on attempt %d:\n%s",
                    attempt + 1,
                    exc,
                )

            except Exception as exc:
                last_error = exc

                logger.warning(
                    "Qwen decision generation failed on attempt %d: %s",
                    attempt + 1,
                    type(exc).__name__,
                )

        raise last_error or LLMMalformedResponseError(
            "Failed to generate a valid conversational decision."
        )

    @staticmethod
    def _raise_provider_error(exc: Exception) -> None:
        """Translate OpenAI/HF exceptions into project LLM errors."""

        error_name = type(exc).__name__
        error_text = str(exc)

        if "Authentication" in error_name:
            raise LLMAuthenticationError(
                "Hugging Face authentication failed. Verify HF_API_KEY."
            ) from exc

        if "RateLimit" in error_name:
            raise LLMRateLimitError(
                "Hugging Face rate limit reached. Retry later."
            ) from exc

        if "Timeout" in error_name:
            raise LLMTimeoutError(
                "Hugging Face request timed out."
            ) from exc

        if "APIConnection" in error_name:
            raise LLMError(
                "Could not connect to Hugging Face."
            ) from exc