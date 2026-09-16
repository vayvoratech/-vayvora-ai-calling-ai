import json
import re
from typing import Any

from src.agents.prompts.router import LLM_ROUTER_PROMPT
from src.agents.state import AgentState


class LLMRouterNode:
    """
    LLM fallback router.

    This node is called ONLY when the fast deterministic router
    cannot confidently determine the user's intent.

    It does not answer the user.
    It only selects:

        direct
        rag
        mcp
        llm
    """

    ALLOWED_ROUTES = {
        "direct",
        "rag",
        "mcp",
        "llm",
    }

    DEFAULT_ROUTE = "llm"

    def __init__(self, llm: Any):
        self.llm = llm

    async def run(self, state: AgentState) -> AgentState:
        user_input = state.get("user_input", "").strip()

        if not user_input:
            return {
                **state,
                "route": "direct",
                "route_confidence": 1.0,
                "route_source": "system",
                "llm_router_required": False,
                "llm_required": False,
                "rag_required": False,
                "tool_required": False,
                "response": "I'm sorry, I didn't hear anything.",
                "is_complete": True,
            }

        messages = self._build_messages(state, user_input)

        try:
            response = await self.llm.ainvoke(messages)

            content = self._extract_content(response)

            decision = self._parse_decision(content)

            route = decision["route"]
            confidence = decision["confidence"]

            return {
                **state,
                "route": route,
                "route_confidence": confidence,
                "route_source": "llm_router",
                "llm_router_required": False,

                "llm_required": route in {
                    "llm",
                    "rag",
                    "mcp",
                },

                "rag_required": route == "rag",
                "tool_required": route == "mcp",

                "is_complete": False,
                "error": None,
            }

        except Exception as exc:
            # Safe fallback:
            # if the routing LLM fails, normal LLM conversation
            # is safer than guessing RAG or MCP.
            return {
                **state,
                "route": self.DEFAULT_ROUTE,
                "route_confidence": 0.0,
                "route_source": "llm_router_fallback",
                "llm_router_required": False,
                "llm_required": True,
                "rag_required": False,
                "tool_required": False,
                "is_complete": False,
                "error": f"LLM router failed: {exc}",
            }

    def _build_messages(
        self,
        state: AgentState,
        user_input: str,
    ) -> list[dict[str, str]]:

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": LLM_ROUTER_PROMPT,
            }
        ]

        conversation = state.get("messages", [])

        # Give the router recent conversation context.
        recent_messages = conversation[-8:]

        for message in recent_messages:
            role = message.get("role")

            if role not in {"user", "assistant"}:
                continue

            content = str(message.get("content", "")).strip()

            if not content:
                continue

            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        # Ensure current user input is present.
        if not any(
            message.get("role") == "user"
            and message.get("content") == user_input
            for message in messages
        ):
            messages.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

        return messages

    @staticmethod
    def _extract_content(response: Any) -> str:
        if hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)

        if isinstance(content, list):
            content = "".join(
                str(item)
                for item in content
            )

        return str(content).strip()

    def _parse_decision(
        self,
        content: str,
    ) -> dict[str, Any]:

        # First try direct JSON.
        try:
            data = json.loads(content)

            return self._validate_decision(data)

        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Handle accidental markdown fences.
        cleaned = re.sub(
            r"```(?:json)?",
            "",
            content,
            flags=re.IGNORECASE,
        ).replace("```", "").strip()

        try:
            data = json.loads(cleaned)

            return self._validate_decision(data)

        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Last-resort route extraction.
        route_match = re.search(
            r'"route"\s*:\s*"(direct|rag|mcp|llm)"',
            content,
            re.IGNORECASE,
        )

        confidence_match = re.search(
            r'"confidence"\s*:\s*([01](?:\.\d+)?)',
            content,
            re.IGNORECASE,
        )

        if route_match:
            route = route_match.group(1).lower()

            confidence = (
                float(confidence_match.group(1))
                if confidence_match
                else 0.50
            )

            return {
                "route": route,
                "confidence": confidence,
            }

        return {
            "route": self.DEFAULT_ROUTE,
            "confidence": 0.0,
        }

    def _validate_decision(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:

        route = str(
            data.get("route", self.DEFAULT_ROUTE)
        ).lower()

        if route not in self.ALLOWED_ROUTES:
            route = self.DEFAULT_ROUTE

        try:
            confidence = float(
                data.get("confidence", 0.0)
            )
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        return {
            "route": route,
            "confidence": confidence,
        }