from src.agents.state import AgentState


class ResponseNode:
    """
    Final Agent response preparation layer.

    Ensures every completed Agent execution produces a
    clean response that can be passed to output guardrails
    and then to streaming TTS.
    """

    FALLBACK_RESPONSE = (
        "I'm sorry, I couldn't process that request right now."
    )

    async def run(self, state: AgentState) -> AgentState:
        response = state.get("response", "").strip()

        # If the LLM did not produce a response but a tool did,
        # preserve the tool result for downstream processing.
        if not response:
            tool_result = state.get("tool_result")

            if tool_result is not None:
                response = str(tool_result)

        # Final fallback.
        if not response:
            response = self.FALLBACK_RESPONSE

        return {
            **state,
            "response": response,
            "is_complete": True,
            "error": state.get("error"),
        }