"""Lightweight deterministic intent detector for broad course catalog queries."""

from __future__ import annotations

import re
from src.core.types import DomainType


def is_catalog_query(query_text: str, domain: DomainType) -> bool:
    """Detect whether a query is requesting the broad/complete project or course catalog list.

    Returns True for catalog/list queries in INFRA_BUILDER (and legacy EDUSAAS), while
    guaranteeing that queries about specific properties/courses or details continue using
    semantic vector retrieval.
    """
    text = query_text.lower().strip()

    if domain == DomainType.INFRA_BUILDER:
        # Negative triggers: specific projects or specific detail inquiries
        specific_real_estate = (
            "skyline",
            "green valley",
            "urban heights",
            "whitefield",
            "sarjapur",
            "electronic city",
            "villa",
            "villas",
            "reschedule",
            "cancel",
            "book site visit",
            "schedule site visit",
            "whatsapp",
        )
        if any(topic in text for topic in specific_real_estate):
            return False

        catalog_patterns = [
            r"\b(list|show|give|tell)\b.*\b(projects?|properties|developments?|apartments?|communities|units)\b",
            r"\b(all|available|ongoing|active)\b.*\b(projects?|properties|developments?|options)\b",
            r"\b(projects?|properties)\b.*\b(available|offered|list|catalog|portfolio|options)\b",
            r"\bwhat\b.*\b(projects?|properties)\b.*\b(have|offer|provide|develop|available|build)\b",
            r"\bwhat\s+(do\s+you|does\s+infra\s*builder)\s+(offer|have|develop|build)\b",
            r"\bwhich\b.*\b(projects?|properties)\b",
            r"\bproject\s*list\b",
            r"\bproperty\s*list\b",
            r"\bcatalog\b",
            r"\bwhat\s+(are\s+)?(your|the)\s+(projects?|properties)\b",
        ]
        return any(re.search(pat, text) for pat in catalog_patterns)

    return False
