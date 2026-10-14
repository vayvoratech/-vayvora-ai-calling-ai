"""
Configuration for the output guardrail layer.

The output guardrail runs after the LLM and before downstream
systems such as TTS.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OutputGuardrailConfig:
    """Configuration for validating and sanitizing LLM output."""

    # Response validation
    min_text_length: int = 1
    max_text_length: int = 2000

    # Voice-response limits
    max_sentence_count: int = 5

    # Formatting controls
    remove_markdown: bool = True
    remove_code_blocks: bool = True
    remove_urls: bool = False

    # PII protection
    enable_pii_redaction: bool = True
    block_on_pii: bool = False

    # Internal/system information protection
    block_system_prompt_leak: bool = True
    block_internal_error_details: bool = True

    # Fallback response
    fallback_message: str = (
        "I'm sorry, but I can't provide that information right now."
    )