import json
import re
from typing import Any

from src.agents.prompts.router import LLM_ROUTER_PROMPT
from src.agents.state import AgentState


class LLMRouterNode:
    """
    LLM-based routing controller.

    This node is the FIRST decision-making stage.

    It does NOT generate the final user response.

    Final routes:
        llm -> normal LLM response
        rag -> retrieve knowledge, then LLM
        mcp -> execute tool, then LLM
    """

    VALID_ROUTES = {
        "llm",
        "rag",
        "mcp",
    }

    MAX_HISTORY_MESSAGES = 8

    def _init_(self, llm: Any) -> None:
        self.llm = llm

    async def run(self, state: AgentState) -> AgentState:
        user_input = state.get("user_input", "").strip()

        if not user_input:
            return {
                **state,
                "route": "llm",
                "route_confidence": 1.0,
                "route_source": "llm_router",
                "llm_router_required": False,
                "rag_required": False,
                "tool_required": False,
                "llm_required": True,
            }

        messages = self._build_messages(state)

        try:
            result = await self.llm.ainvoke(messages)

            raw_content = self._extract_content(result)

            decision = self._parse_decision(raw_content)

            route = decision["route"]
            confidence = decision["confidence"]

            return self._apply_route(
                state=state,
                route=route,
                confidence=confidence,
            )

        except Exception as exc:
            # Safe fallback:
            # if the routing model fails, use the normal LLM.
            return {
                **state,
                "route": "llm",
                "route_confidence": 0.0,
                "route_source": "llm_router_fallback",
                "llm_router_required": False,
                "rag_required": False,
                "tool_required": False,
                "llm_required": True,
                "error": f"LLM router error: {exc}",
            }

    # ---------------------------------------------------------
    # Build router messages
    # ---------------------------------------------------------

    def _build_messages(
        self,
        state: AgentState,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": LLM_ROUTER_PROMPT,
            }
        ]

        conversation = state.get("messages", [])

        if conversation:
            recent_messages = conversation[
                -self.MAX_HISTORY_MESSAGES:
            ]

            for message in recent_messages:
                role = message.get("role")

                if role not in {
                    "user",
                    "assistant",
                }:
                    continue

                content = message.get("content", "")

                if not content:
                    continue

                messages.append(
                    {
                        "role": role,
                        "content": str(content),
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": state.get("user_input", ""),
            }
        )

        return messages

    # ---------------------------------------------------------
    # Extract model content
    # ---------------------------------------------------------

    @staticmethod
    def _extract_content(result: Any) -> str:
        content = getattr(result, "content", result)

        if isinstance(content, list):
            parts = []

            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")

                    if text:
                        parts.append(str(text))

                elif hasattr(item, "text"):
                    parts.append(str(item.text))

                else:
                    parts.append(str(item))

            return "\n".join(parts).strip()

        return str(content).strip()

    # ---------------------------------------------------------
    # Parse JSON decision
    # ---------------------------------------------------------

    def _parse_decision(
        self,
        content: str,
    ) -> dict[str, Any]:
        content = content.strip()

        # ---------------------------------------------
        # 1. Direct JSON
        # ---------------------------------------------

        try:
            data = json.loads(content)

            return self._validate_decision(data)

        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # ---------------------------------------------
        # 2. Markdown JSON block
        # ---------------------------------------------

        fenced_match = re.search(
            r"(?:json)?\s*(\{.*?\})\s*",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if fenced_match:
            try:
                data = json.loads(
                    fenced_match.group(1)
                )

                return self._validate_decision(data)

            except (
                json.JSONDecodeError,
                ValueError,
                TypeError,
            ):
                pass

        # ---------------------------------------------
        # 3. Extract JSON object
        # ---------------------------------------------

        json_match = re.search(
            r"\{.*\}",
            content,
            flags=re.DOTALL,
        )

        if json_match:
            try:
                data = json.loads(
                    json_match.group(0)
                )

                return self._validate_decision(data)

            except (
                json.JSONDecodeError,
                ValueError,
                TypeError,
            ):
                pass

        # ---------------------------------------------
        # 4. Route extraction fallback
        # ---------------------------------------------

        route_match = re.search(
            r"\b(llm|rag|mcp)\b",
            content.lower(),
        )

        if route_match:
            return {
                "route": route_match.group(1),
                "confidence": 0.5,
            }

        raise ValueError(
            "LLM router returned an invalid routing decision."
        )

    # ---------------------------------------------------------
    # Validate decision
    # ---------------------------------------------------------

    def _validate_decision(
        self,
        data: Any,
    ) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError(
                "Router response must be a JSON object."
            )

        route = str(
            data.get("route", "")
        ).strip().lower()

        if route not in self.VALID_ROUTES:
            raise ValueError(
                f"Invalid route: {route}"
            )

        try:
            confidence = float(
                data.get("confidence", 0.0)
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        return {
            "route": route,
            "confidence": confidence,
        }

    # ---------------------------------------------------------
    # Apply route to state
    # ---------------------------------------------------------

    @staticmethod
    def _apply_route(
        state: AgentState,
        route: str,
        confidence: float,
    ) -> AgentState:
        return {
            **state,

            "route": route,
            "route_confidence": confidence,
            "route_source": "llm_router",

            "llm_router_required": False,

            "llm_required": route == "llm",

            "rag_required": route == "rag",

            "tool_required": route == "mcp",
        }