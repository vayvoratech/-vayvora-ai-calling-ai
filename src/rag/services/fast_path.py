"""Deterministic Fast-Paths for High-Resonance Canonical Facts."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional

FAST_PATH_RULES = [
    {
        "topic": "office_headquarters",
        "source": "knowledge_base/vayvora/company/overview.md",
        "pattern": re.compile(
            r"\b(where\s+is\s+(?:your|the)\s+office|where\s+are\s+you\s+located|headquarters|office\s+location|company\s+address|where\s+is\s+vayvora(?:\s+located)?)\b",
            re.IGNORECASE,
        ),
        "response": (
            "Vayvora Technologies' primary engineering and research facility is located in Bangalore, India, "
            "with remote collaboration across international time zones."
        ),
    },
    {
        "topic": "official_email",
        "source": "knowledge_base/edusaas/courses/overview_catalog.md",
        "pattern": re.compile(
            r"\b(what\s+is\s+(?:your|the|company'?s?|vayvora'?s?)\s+email|(?:official|company|contact|support)\s+email(?:\s+address)?|how\s+to\s+email\s+vayvora)\b",
            re.IGNORECASE,
        ),
        "response": (
            "You can reach our team via email at info@edusaas.vayvora.com or visit our official portal at edusaas.vayvora.com."
        ),
    },
    {
        "topic": "company_leadership",
        "source": "knowledge_base/vayvora/company/overview.md",
        "pattern": re.compile(
            r"\b(who\s+is\s+(?:the\s+)?ceo|who\s+founded\s+vayvora|founder\s+of\s+vayvora|leadership\s+team|who\s+leads\s+vayvora)\b",
            re.IGNORECASE,
        ),
        "response": (
            "Vayvora Technologies specializes in enterprise artificial intelligence engineering, "
            "high-throughput voice agent architectures, and custom cloud software development."
        ),
    },
    {
        "topic": "service_catalog",
        "source": "knowledge_base/vayvora/solutions/enterprise_ai.md",
        "pattern": re.compile(
            r"\b(what\s+services\s+does\s+vayvora\s+(?:provide|offer)|what\s+does\s+(?:your\s+company|vayvora)\s+do|what\s+are\s+your\s+services|overview\s+of\s+(?:your\s+)?services|services\s+catalog)\b",
            re.IGNORECASE,
        ),
        "response": (
            "Vayvora provides enterprise AI solutions including RAG pipelines, structured entity extraction, "
            "multi-tenant conversational routing, vector search, and semantic verification guardrails."
        ),
    },
    {
        "topic": "ai_course_price",
        "source": "knowledge_base/edusaas/courses/overview_catalog.md",
        "pattern": re.compile(
            r"\b((?:price|fee|cost)\s+(?:of|for)\s+(?:the\s+)?ai(?:\s+engineering)?\s+course|how\s+much\s+(?:is|does)\s+the\s+ai(?:\s+engineering)?\s+course\s+cost|ai\s+course\s+price|ai\s+course\s+fee)\b",
            re.IGNORECASE,
        ),
        "response": (
            "The tuition fee for our AI & Machine Learning program track is five thousand rupees (₹5,000 INR)."
        ),
    },
    {
        "topic": "general_course_fee",
        "source": "knowledge_base/edusaas/courses/overview_catalog.md",
        "pattern": re.compile(
            r"\b((?:how\s+much\s+(?:is|are|do)\s+the\s+courses?|(?:course|courses|tuition)\s+(?:price|fee|cost)|fees\s+for\s+courses?))\b",
            re.IGNORECASE,
        ),
        "response": (
            "The tuition fee across our EduSaaS engineering specializations is five thousand rupees (₹5,000 INR) "
            "per program track, including hands-on capstones, automated coding assessments, and AI voice mock interviews."
        ),
    },
    {
        "topic": "all_courses_list",
        "source": "knowledge_base/edusaas/courses/overview_catalog.md",
        "pattern": re.compile(
            r"\b(what\s+courses\s+do\s+you\s+(?:offer|have|provide)|list\s+of\s+courses|course\s+catalog|all\s+courses|what\s+can\s+i\s+learn|all\s+programs)\b",
            re.IGNORECASE,
        ),
        "response": (
            "EduSaaS offers six comprehensive engineering programs: Artificial Intelligence & Machine Learning, "
            "Full-Stack Web & Software Engineering, Data Science & Business Analytics, Cloud Computing & DevOps, "
            "Cybersecurity & Ethical Hacking, and Data Structures, Algorithms & System Design."
        ),
    },
    {
        "topic": "sla_and_refund",
        "source": "knowledge_base/edusaas/policies/admissions_policy.md",
        "pattern": re.compile(
            r"\b(sla\s+policy|uptime\s+guarantee|refund\s+policy|service\s+level\s+agreement|guaranteed\s+uptime)\b",
            re.IGNORECASE,
        ),
        "response": (
            "EduSaaS financial arrangements including installment plans and fee policies are reviewed "
            "during the admissions interview based on cohort availability."
        ),
    },
]


def get_deterministic_fast_path(query: str) -> Optional[Dict[str, Any]]:
    """Evaluates high-resonance deterministic fast-paths in sub-millisecond time (< 3ms)."""
    t_start = time.perf_counter()
    q = query.strip()
    if not q:
        return None

    for rule in FAST_PATH_RULES:
        if rule["pattern"].search(q):
            latency_ms = round((time.perf_counter() - t_start) * 1000 + 0.5, 2)
            return {
                "response": rule["response"],
                "topic": rule["topic"],
                "source": rule["source"],
                "confidence": 0.99,
                "latency_ms": latency_ms,
                "bypassed_llm": True,
            }

    return None
