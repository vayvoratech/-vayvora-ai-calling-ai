"""
Configuration for the input guardrail layer.

This module contains only static, low-latency configuration.
No model calls, network calls, or database access happen here.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InputGuardrailConfig:
    """Configuration used by the input guardrail."""

    # Basic input limits
    max_text_length: int = 4000
    min_text_length: int = 1

    # Prevent excessive repeated characters/words
    max_repeated_characters: int = 20
    max_repeated_words: int = 10

    # Prompt-injection detection
    detect_prompt_injection: bool = True

    # Dangerous instruction detection
    detect_unsafe_requests: bool = True

    # Transcript cleanup
    normalize_whitespace: bool = True
    strip_text: bool = True

    # Whether suspicious input should be blocked
    block_prompt_injection: bool = True
    block_unsafe_requests: bool = False

    # Detection sensitivity
    injection_threshold: int = 1

    # Custom blocked phrases can be added later
    blocked_phrases: tuple[str, ...] = field(
        default_factory=tuple
    )

    # Custom allowed phrases can be added later
    allowed_phrases: tuple[str, ...] = field(
        default_factory=tuple
    )