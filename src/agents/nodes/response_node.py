from __future__ import annotations

from typing import Any
from src.agents.state import AgentState


class ResponseNode:
    """
    Final Agent response preparation layer.

    Ensures every completed Agent execution produces a clean, spoken-ready
    plain-text string complying with Vayvora AI speech synthesis constraints.
    """

    FALLBACK_RESPONSE = "I'm sorry, I couldn't process that request right now."

    async def run(self, state: AgentState) -> AgentState:
        raw_response = state.get("response")
        response = self._extract_clean_text(raw_response)

        # If the LLM did not produce speech text, check tool output
        if not response:
            tool_result = state.get("tool_result")
            if tool_result is not None:
                response = self._extract_clean_text(tool_result)

        if not response:
            response = self.FALLBACK_RESPONSE

        # Apply voice speech guardrails: strip markdown formatting
        clean_voice_text = self._sanitize_for_voice(response)

        return {
            **state,
            "response": clean_voice_text,
            "is_complete": True,
            "error": state.get("error"),
        }

    @classmethod
    def _extract_clean_text(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = [cls._extract_clean_text(item) for item in value]
            return " ".join(p for p in parts if p).strip()
        if isinstance(value, dict):
            for key in ("message", "text", "response", "content", "summary"):
                if key in value and isinstance(value[key], str):
                    return value[key].strip()
            return str(value).strip()
        return str(value).strip()

    @staticmethod
    def _sanitize_for_voice(text: str) -> str:
        """Strip markdown markers (bold, italics, bullets, headers) for clean TTS."""
        import re
        t = text
        t = re.sub(r"[*_~`#>]", "", t)
        t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
        t = re.sub(r"\n+", " ", t)
        return " ".join(t.split()).strip()
