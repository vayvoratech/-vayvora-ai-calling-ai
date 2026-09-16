import re

from src.agents.state import AgentState


class RouterNode:
    """
    Fast deterministic router.

    This router handles only HIGH-CONFIDENCE intents.

    Ambiguous requests are sent to LLMRouterNode.

    Routes:
        direct
        rag
        mcp
        llm
        llm_router
    """

    # ─────────────────────────────────────────────
    # DIRECT
    # ─────────────────────────────────────────────

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

    # ─────────────────────────────────────────────
    # STRONG MCP ACTION PATTERNS
    # ─────────────────────────────────────────────

    MCP_PATTERNS = (
        r"\bsend\s+(?:a\s+)?whatsapp\b",
        r"\bsend\s+(?:a\s+)?message\s+to\b",
        r"\bsend\s+(?:an\s+)?email\s+to\b",
        r"\bcreate\s+(?:a\s+)?calendar\s+(?:event|appointment)\b",
        r"\badd\s+(?:a\s+)?(?:calendar\s+)?(?:event|appointment)\b",
        r"\bschedule\s+(?:a\s+)?(?:meeting|appointment|event)\b",
        r"\bbook\s+(?:a\s+)?(?:meeting|appointment)\b",
        r"\bcancel\s+(?:my\s+)?(?:meeting|appointment|event)\b",
        r"\breschedule\s+(?:my\s+)?(?:meeting|appointment|event)\b",
        r"\bupdate\s+(?:my\s+)?(?:calendar|event|appointment)\b",
        r"\bcheck\s+my\s+(?:calendar|email|appointments?)\b",
        r"\bshow\s+my\s+(?:calendar|events|appointments?)\b",
        r"\bcheck\s+(?:my\s+)?email\b",
    )

    # ─────────────────────────────────────────────
    # STRONG COMPANY / RAG PATTERNS
    # ─────────────────────────────────────────────

    COMPANY_TERMS = (
        "vayvora",
        "our company",
        "company",
        "company's",
        "company projects",
        "company policy",
        "company policies",
        "company services",
        "ai calling project",
        "ai calling",
        "edusaas",
        "ai summit",
        "employee",
        "employees",
        "team",
        "department",
        "departments",
        "portfolio",
        "refund policy",
        "expense policy",
        "office parking",
        "it support",
    )

    KNOWLEDGE_INTENT_TERMS = (
        "what",
        "who",
        "which",
        "where",
        "when",
        "how",
        "why",
        "tell me",
        "explain",
        "information",
        "details",
        "about",
    )

    async def run(self, state: AgentState) -> AgentState:
        user_input = state.get(
            "user_input",
            "",
        ).strip()

        # ─────────────────────────────────────────
        # EMPTY INPUT
        # ─────────────────────────────────────────

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

        text = self._normalize(user_input)

        # ─────────────────────────────────────────
        # 1. SIMPLE DIRECT RESPONSE
        # ─────────────────────────────────────────

        if self._is_direct(text):
            return {
                **state,
                "route": "direct",
                "route_confidence": 0.99,
                "route_source": "fast_router",
                "llm_router_required": False,
                "llm_required": False,
                "rag_required": False,
                "tool_required": False,
                "response": self._direct_response(text),
                "is_complete": True,
            }

        # ─────────────────────────────────────────
        # 2. HIGH-CONFIDENCE MCP
        # ─────────────────────────────────────────

        if self._is_strong_mcp(text):
            return {
                **state,
                "route": "mcp",
                "route_confidence": 0.96,
                "route_source": "fast_router",
                "llm_router_required": False,
                "llm_required": True,
                "rag_required": False,
                "tool_required": True,
                "is_complete": False,
            }

        # ─────────────────────────────────────────
        # 3. HIGH-CONFIDENCE COMPANY RAG
        # ─────────────────────────────────────────

        if self._is_strong_rag(text):
            return {
                **state,
                "route": "rag",
                "route_confidence": 0.96,
                "route_source": "fast_router",
                "llm_router_required": False,
                "llm_required": True,
                "rag_required": True,
                "tool_required": False,
                "is_complete": False,
            }

        # ─────────────────────────────────────────
        # 4. AMBIGUOUS → LLM ROUTER
        # ─────────────────────────────────────────

        return {
            **state,
            "route": "llm_router",
            "route_confidence": 0.50,
            "route_source": "fast_router",
            "llm_router_required": True,
            "llm_required": True,
            "rag_required": False,
            "tool_required": False,
            "is_complete": False,
        }

    # ─────────────────────────────────────────────
    # NORMALIZATION
    # ─────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            text.lower().strip(),
        )

    # ─────────────────────────────────────────────
    # DIRECT
    # ─────────────────────────────────────────────

    @classmethod
    def _is_direct(cls, text: str) -> bool:
        return any(
            text == phrase
            or text.startswith(f"{phrase} ")
            or text.endswith(f" {phrase}")
            for phrase in cls.DIRECT_PHRASES
        )

    @staticmethod
    def _direct_response(text: str) -> str:

        if text in {
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        }:
            return (
                "Hello! I'm Vayvora AI, the AI assistant "
                "for Vayvora Technologies. How can I help "
                "you today?"
            )

        if text in {
            "bye",
            "goodbye",
        }:
            return "Goodbye! Have a great day."

        return "You're welcome. How can I help you?"

    # ─────────────────────────────────────────────
    # MCP
    # ─────────────────────────────────────────────

    @classmethod
    def _is_strong_mcp(cls, text: str) -> bool:
        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
            for pattern in cls.MCP_PATTERNS
        )

    # ─────────────────────────────────────────────
    # RAG
    # ─────────────────────────────────────────────

    @classmethod
    def _is_strong_rag(cls, text: str) -> bool:

        has_company_term = any(
            term in text
            for term in cls.COMPANY_TERMS
        )

        if not has_company_term:
            return False

        # Company statement/question is enough to use
        # the company knowledge base.
        return True