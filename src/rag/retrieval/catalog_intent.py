"""Lightweight deterministic intent detector for broad course catalog queries."""

from __future__ import annotations

import re
from src.core.types import DomainType


def is_catalog_query(query_text: str, domain: DomainType) -> bool:
    """Detect whether a query is requesting the broad/complete course catalog list.

    Returns True ONLY for catalog/list queries in the EduSaaS domain, while
    guaranteeing that queries about specific courses or details continue using
    semantic vector retrieval.
    """
    if domain != DomainType.EDUSAAS:
        return False

    text = query_text.lower().strip()

    # Negative triggers: if the user asks about specific course topics or details,
    # it is a specific query, NOT a broad catalog query.
    specific_topics = (
        "artificial intelligence",
        "ai/ml",
        "machine learning",
        "full stack",
        "full-stack",
        "web development",
        "react",
        "data science",
        "business analytics",
        "cloud computing",
        "devops",
        "aws",
        "kubernetes",
        "cybersecurity",
        "ethical hacking",
        "data structures",
        "dsa",
        "algorithms",
        "system design",
        "technolog",
        "fee for",
        "cost of",
        "price of",
        "syllabus",
        "prerequisite",
        "placement",
        "demo",
        "admission process",
        "interview",
        "refund",
    )
    if any(topic in text for topic in specific_topics):
        return False

    # Positive patterns matching broad catalog / course list intent
    catalog_patterns = [
        r"\b(list|show|give|tell)\b.*\b(courses?|programs?|tracks?|offerings?|classes)\b",
        r"\b(all|available|offered)\b.*\b(courses?|programs?|tracks?)\b",
        r"\b(courses?|programs?|tracks?)\b.*\b(available|offered|list|catalog|options)\b",
        r"\bwhat\b.*\b(courses?|programs?|tracks?)\b.*\b(offer|have|provide|available|teach|run)\b",
        r"\bwhat\s+(do\s+you|does\s+edusaas)\s+(offer|have|provide|teach)\b",
        r"\bwhich\b.*\b(courses?|programs?|tracks?)\b",
        r"\bcourse\s*list\b",
        r"\bprogram\s*list\b",
        r"\bcatalog\b",
        r"\bwhat\s+(are\s+)?(your|the)\s+(courses?|programs?)\b",
    ]

    return any(re.search(pat, text) for pat in catalog_patterns)
