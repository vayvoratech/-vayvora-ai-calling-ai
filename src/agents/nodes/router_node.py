from src.agents.state import AgentState


class RouterNode:
    """
    Fast local router for Vayvora AI.

    This node performs deterministic routing without calling an LLM.

    Routes:
        direct -> fixed/common responses
        llm    -> normal conversation
        rag    -> knowledge retrieval + one LLM call
        mcp    -> live data/action + one LLM call
    """

    DIRECT_PHRASES = (
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
    )

    MCP_KEYWORDS = (
        "book",
        "booking",
        "schedule",
        "cancel",
        "reschedule",
        "send",
        "create",
        "update",
        "delete",
        "check my",
        "my appointment",
        "my calendar",
        "my email",
    )

    RAG_KEYWORDS = (
        "policy",
        "policies",
        "pricing",
        "price",
        "refund",
        "documentation",
        "docs",
        "information",
        "details",
        "explain",
        "what is",
        "how does",
        "how do",
        "eligibility",
        "terms",
        "conditions",
    )

    async def run(self, state: AgentState) -> AgentState:
        user_input = state.get("user_input", "").strip()

        if not user_input:
            return {
                **state,
                "route": "direct",
                "route_confidence": 1.0,
                "llm_required": False,
                "rag_required": False,
                "tool_required": False,
                "response": "I'm sorry, I didn't hear anything.",
                "is_complete": True,
            }

        text = user_input.lower()

        # 1. Fixed/common responses — fastest path.
        if self._is_direct(text):
            return {
                **state,
                "route": "direct",
                "route_confidence": 0.98,
                "llm_required": False,
                "rag_required": False,
                "tool_required": False,
                "is_complete": False,
            }

        # 2. Live actions / external data.
        if self._contains_keyword(text, self.MCP_KEYWORDS):
            return {
                **state,
                "route": "mcp",
                "route_confidence": 0.90,
                "llm_required": True,
                "rag_required": False,
                "tool_required": True,
                "is_complete": False,
            }

        # 3. Internal knowledge / documentation.
        if self._contains_keyword(text, self.RAG_KEYWORDS):
            return {
                **state,
                "route": "rag",
                "route_confidence": 0.88,
                "llm_required": True,
                "rag_required": True,
                "tool_required": False,
                "is_complete": False,
            }

        # 4. Normal conversation.
        return {
            **state,
            "route": "llm",
            "route_confidence": 0.80,
            "llm_required": True,
            "rag_required": False,
            "tool_required": False,
            "is_complete": False,
        }

    @staticmethod
    def _is_direct(text: str) -> bool:
        return any(
            text == phrase or text.startswith(f"{phrase} ")
            for phrase in RouterNode.DIRECT_PHRASES
        )

    @staticmethod
    def _contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)