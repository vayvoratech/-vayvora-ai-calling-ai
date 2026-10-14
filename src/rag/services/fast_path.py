"""Deterministic Fast-Paths for High-Resonance Canonical Facts."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional

FAST_PATH_RULES = [
    {
        "topic": "office_headquarters",
        "source": "knowledge_base/infra_builder/company/overview.md",
        "pattern": re.compile(
            r"\b(where\s+is\s+(?:your|the)\s+office|where\s+are\s+you\s+located|headquarters|office\s+location|company\s+address|where\s+is\s+infra\s*builder(?:\s+located)?)\b",
            re.IGNORECASE,
        ),
        "response": (
            "Infra Builder's corporate headquarters is located at Infra Builder Tower, Outer Ring Road, Bangalore, Karnataka."
        ),
    },
    {
        "topic": "official_contact",
        "source": "knowledge_base/infra_builder/company/overview.md",
        "pattern": re.compile(
            r"\b(what\s+is\s+(?:your|the|company'?s?)\s+email|(?:official|company|contact|support)\s+email|contact\s+details|reach\s+infra\s*builder)\b",
            re.IGNORECASE,
        ),
        "response": (
            "You can reach our property advisory team at info@infrabuilder.com or via WhatsApp at our official number."
        ),
    },
    {
        "topic": "projects_catalog",
        "source": "knowledge_base/infra_builder/projects/overview_catalog.md",
        "pattern": re.compile(
            r"\b(what\s+projects\s+do\s+you\s+(?:offer|have|build)|list\s+of\s+projects|all\s+projects|current\s+projects|developments)\b",
            re.IGNORECASE,
        ),
        "response": (
            "Infra Builder has three premier residential developments in Bangalore: Skyline Towers in Whitefield (2 & 3 BHK apartments), Green Valley Villas on Sarjapur Road (4 & 5 BHK luxury villas), and Urban Heights in Electronic City (1 & 2 BHK ready-to-move homes)."
        ),
    },
    {
        "topic": "skyline_towers_pricing",
        "source": "knowledge_base/infra_builder/projects/skyline_towers.md",
        "pattern": re.compile(
            r"\b(price\s+of\s+skyline\s+towers|skyline\s+towers\s+cost|pricing\s+for\s+skyline|skyline\s+towers\s+price)\b",
            re.IGNORECASE,
        ),
        "response": (
            "At Skyline Towers in Whitefield, 2 BHK apartments start at ₹85 Lakhs and 3 BHK luxury residences start at ₹1.42 Crores."
        ),
    },
    {
        "topic": "green_valley_pricing",
        "source": "knowledge_base/infra_builder/projects/green_valley_villas.md",
        "pattern": re.compile(
            r"\b(price\s+of\s+green\s+valley|green\s+valley\s+cost|green\s+valley\s+villas\s+price|pricing\s+for\s+green\s+valley)\b",
            re.IGNORECASE,
        ),
        "response": (
            "At Green Valley Villas on Sarjapur Road, 4 BHK villas start at ₹2.5 Crores and 5 BHK grand estates start at ₹3.8 Crores."
        ),
    },
    {
        "topic": "urban_heights_pricing",
        "source": "knowledge_base/infra_builder/projects/urban_heights.md",
        "pattern": re.compile(
            r"\b(price\s+of\s+urban\s+heights|urban\s+heights\s+cost|urban\s+heights\s+price|pricing\s+for\s+urban\s+heights)\b",
            re.IGNORECASE,
        ),
        "response": (
            "Urban Heights in Electronic City offers ready-to-move homes with 1 BHK apartments from ₹45 Lakhs and 2 BHK units from ₹68 Lakhs."
        ),
    },
    {
        "topic": "site_visit_policy",
        "source": "knowledge_base/infra_builder/policies/site_visit_policy.md",
        "pattern": re.compile(
            r"\b(site\s+visit\s+timing|visiting\s+hours|when\s+can\s+i\s+visit|site\s+visit\s+policy|free\s+cab|metro\s+pickup)\b",
            re.IGNORECASE,
        ),
        "response": (
            "Site visits for all Infra Builder developments are conducted daily between 10:00 AM and 6:00 PM, with complimentary pickup and drop services from nearby metro stations."
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
