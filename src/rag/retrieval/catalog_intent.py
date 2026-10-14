"""Intent detection for broad JB Infra project portfolio queries."""

from __future__ import annotations

import re

from src.core.types import DomainType


# Broad portfolio/list questions.
PORTFOLIO_PATTERNS = [
    r"\b(list|show|give|tell)\b.*\b(projects?|properties?|developments?)\b",
    r"\b(all|available|current)\b.*\b(projects?|properties?|developments?)\b",
    r"\b(projects?|properties?|developments?)\b.*\b("
    r"available|list|portfolio|options|offerings?"
    r")\b",
    r"\bwhat\b.*\b(projects?|properties?)\b.*\b("
    r"offer|have|provide|develop|available"
    r")\b",
    r"\bwhich\b.*\b(projects?|properties?|developments?)\b",
    r"\bproject\s*list\b",
    r"\bproperty\s*list\b",
    r"\bproject\s*portfolio\b",
    r"\bproperty\s*portfolio\b",
    r"\bwhat\s+(?:are\s+)?(?:your|the)\s+projects?\b",
]


# If the customer asks about a specific project/detail, normal semantic
# retrieval should handle the query instead of treating it as a portfolio
# request.
SPECIFIC_PROJECT_TOPICS = (
    "pristine city",
    "nature valley",
    "serene city",
    "jb pride",
    "hillside county",
    "serene villas",
    "serene county",
    "serene resort",
    "price of",
    "cost of",
    "rate",
    "availability",
    "plot number",
    "booking",
    "payment",
    "location of",
    "where is",
)


def is_portfolio_query(
    query_text: str,
    domain: DomainType,
) -> bool:
    """Return True for broad JB Infra project-list queries."""

    if domain != DomainType.GENERAL:
        return False

    text = query_text.lower().strip()

    if not text:
        return False

    if any(
        topic in text
        for topic in SPECIFIC_PROJECT_TOPICS
    ):
        return False

    return any(
        re.search(pattern, text)
        for pattern in PORTFOLIO_PATTERNS
    )


def is_catalog_query(
    query_text: str,
    domain: DomainType,
) -> bool:
    """Backward-compatible alias for portfolio intent detection."""

    return is_portfolio_query(
        query_text=query_text,
        domain=domain,
    )


__all__ = [
    "PORTFOLIO_PATTERNS",
    "is_portfolio_query",
    "is_catalog_query",
]