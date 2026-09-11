from src.agents.state import AgentState


class RouterNode:
    """
    Fast local router for Vayvora AI.

    This node performs deterministic routing without calling an LLM.

    Routes:
        direct -> fixed/common responses
        llm    -> normal conversation
        rag    -> knowledge retrieval + LLM
        mcp    -> live data/action + LLM
    """

    # ---------------------------------------------------------
    # Direct / fixed responses
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # MCP / live actions
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Knowledge-base / RAG requests
    # ---------------------------------------------------------

    RAG_KEYWORDS = (
        "policy",
        "policies",
        "pricing",
        "price",
        "prices",
        "refund",
        "documentation",
        "document",
        "docs",
        "information",
        "details",
        "explain",
        "what is",
        "what are",
        "what does",
        "what do",
        "how does",
        "how do",
        "why",
        "eligibility",
        "eligible",
        "terms",
        "conditions",
        "services",
        "service",
        "company",
        "about vayvora",
        "about the company",
        "portfolio",
        "support",
        "contact",
        "career",
        "careers",
    )

    # ---------------------------------------------------------
    # Router
    # ---------------------------------------------------------

    async def run(self, state: AgentState) -> AgentState:
        user_input = state.get("user_input", "").strip()

        # -----------------------------------------------------
        # Empty input
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # 1. Direct / fixed responses
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # 2. MCP / live actions
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # 3. Knowledge-base / RAG
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # 4. Normal conversation
        # -----------------------------------------------------

        return {
            **state,
            "route": "llm",
            "route_confidence": 0.80,
            "llm_required": True,
            "rag_required": False,
            "tool_required": False,
            "is_complete": False,
        }

    # ---------------------------------------------------------
    # Direct phrase detection
    # ---------------------------------------------------------

    @staticmethod
    def _is_direct(text: str) -> bool:
        """
        Detect simple fixed/common conversational phrases.
        """

        normalized = text.strip().lower()

        return any(
            normalized == phrase
            or normalized.startswith(f"{phrase} ")
            or normalized.endswith(f" {phrase}")
            for phrase in RouterNode.DIRECT_PHRASES
        )

    # ---------------------------------------------------------
    # Keyword detection
    # ---------------------------------------------------------

    @staticmethod
    def _contains_keyword(
        text: str,
        keywords: tuple[str, ...],
    ) -> bool:
        """
        Detect whether any configured keyword/phrase
        occurs in the user input.
        """

        return any(
            keyword in text
            for keyword in keywords
        )