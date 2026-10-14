"""Deterministic fast-path responses for common JB Infra questions."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional


FAST_PATH_RULES = [
    {
        "topic": "company_address",
        "source": "knowledge_base/company/contact.md",
        "pattern": re.compile(
            r"\b("
            r"where\s+is\s+(?:your|the)\s+office|"
            r"where\s+are\s+you\s+located|"
            r"head\s*office|"
            r"headquarters|"
            r"office\s+location|"
            r"company\s+address|"
            r"jb\s*infra\s+(?:office|address|location)"
            r")\b",
            re.IGNORECASE,
        ),
        "response": (
            "JB Infra's head office is located at Plot No. 1, "
            "2nd and 3rd Floor, adjacent to B.D. Reddy Gardens, "
            "Chaitanya Nagar, B.N. Reddy Nagar, Hyderabad, "
            "Telangana 500070."
        ),
    },
    {
        "topic": "company_phone",
        "source": "knowledge_base/company/contact.md",
        "pattern": re.compile(
            r"\b("
            r"what\s+is\s+(?:your|the)\s+(?:phone|mobile|contact)\s+number|"
            r"contact\s+number|"
            r"phone\s+number|"
            r"how\s+can\s+i\s+call\s+jb\s*infra|"
            r"jb\s*infra\s+(?:phone|number|contact)"
            r")\b",
            re.IGNORECASE,
        ),
        "response": (
            "You can contact JB Infra on 040 2955 4577."
        ),
    },
    {
        "topic": "company_email",
        "source": "knowledge_base/company/contact.md",
        "pattern": re.compile(
            r"\b("
            r"what\s+is\s+(?:your|the|jb\s*infra'?s?)\s+email|"
            r"official\s+email|"
            r"company\s+email|"
            r"contact\s+email|"
            r"email\s+address"
            r")\b",
            re.IGNORECASE,
        ),
        "response": (
            "You can reach JB Infra by email at "
            "contact@jbinfraprojects.in."
        ),
    },
    {
        "topic": "company_overview",
        "source": "knowledge_base/company/about.md",
        "pattern": re.compile(
            r"\b("
            r"what\s+is\s+jb\s*infra|"
            r"tell\s+me\s+about\s+jb\s*infra|"
            r"about\s+jb\s*infra|"
            r"who\s+is\s+jb\s*infra|"
            r"what\s+does\s+jb\s*infra\s+do"
            r")\b",
            re.IGNORECASE,
        ),
        "response": (
            "JB Infra is a real-estate development company with "
            "more than 25 years of experience. The company says it "
            "has developed more than 2,500 acres and served more "
            "than 25,000 happy customers, with a focus on legally "
            "compliant real-estate projects."
        ),
    },
    {
        "topic": "projects",
        "source": "knowledge_base/company/about.md",
        "pattern": re.compile(
            r"\b("
            r"what\s+projects\s+do\s+you\s+have|"
            r"list\s+(?:your|the)\s+projects|"
            r"which\s+projects\s+do\s+you\s+have|"
            r"jb\s*infra\s+projects|"
            r"projects\s+available"
            r")\b",
            re.IGNORECASE,
        ),
        "response": (
            "JB Infra's project portfolio includes Pristine City, "
            "Nature Valley, Serene City, Hillside County, Serene "
            "Villas, Serene County, Serene Resort, and JB Pride."
        ),
    },
    {
        "topic": "pristine_city",
        "source": "knowledge_base/projects/pristine_city.md",
        "pattern": re.compile(
            r"\b("
            r"pristine\s+city|"
            r"tell\s+me\s+about\s+pristine|"
            r"what\s+is\s+pristine\s+city"
            r")\b",
            re.IGNORECASE,
        ),
        "response": (
            "Pristine City is a JB Infra project in Vikarabad "
            "spread across more than 150 acres. The project is "
            "described as DTCP and RERA approved and includes "
            "features such as black-top roads, underground drainage "
            "and rainwater harvesting."
        ),
    },
    {
        "topic": "nature_valley",
        "source": "knowledge_base/projects/nature_valley.md",
        "pattern": re.compile(
            r"\b("
            r"nature\s+valley|"
            r"tell\s+me\s+about\s+nature\s+valley|"
            r"what\s+is\s+nature\s+valley"
            r")\b",
            re.IGNORECASE,
        ),
        "response": (
            "JB Infra's Nature Valley is a large plotted development "
            "spread across about 700 acres, with more than 6,000 "
            "plots. The project includes 100-foot roads, around "
            "9 kilometres of cycling tracks and 30 percent green zones."
        ),
    },
    {
        "topic": "serene_city",
        "source": "knowledge_base/projects/serene_city.md",
        "pattern": re.compile(
            r"\b("
            r"serene\s+city|"
            r"tell\s+me\s+about\s+serene\s+city|"
            r"what\s+is\s+serene\s+city"
            r")\b",
            re.IGNORECASE,
        ),
        "response": (
            "Serene City is a JB Infra project in Ibrahimpatnam. "
            "It is described as HMDA approved and includes plots, "
            "villas and commercial spaces, along with features such "
            "as black-top roads, solar fencing and a clubhouse."
        ),
    },
    {
        "topic": "jb_pride",
        "source": "knowledge_base/projects/jb_pride.md",
        "pattern": re.compile(
            r"\b("
            r"jb\s+pride|"
            r"tell\s+me\s+about\s+jb\s+pride|"
            r"what\s+is\s+jb\s+pride"
            r")\b",
            re.IGNORECASE,
        ),
        "response": (
            "JB Pride is a JB Infra development in the Tandur and "
            "Mominpet area. The project includes features such as "
            "rainwater harvesting, herbal gardens, clubhouses, a "
            "sports arena, cricket ground and amphitheatre."
        ),
    },
    {
        "topic": "pricing",
        "source": "knowledge_base/agent_guidelines.md",
        "pattern": re.compile(
            r"\b("
            r"price|"
            r"pricing|"
            r"cost|"
            r"rate|"
            r"per\s+(?:sq|square)\s*(?:ft|feet)|"
            r"plot\s+price|"
            r"how\s+much\s+does\s+it\s+cost"
            r")\b",
            re.IGNORECASE,
        ),
        "response": (
            "Pricing depends on the specific project and current "
            "availability. I don't have a verified current price "
            "in my knowledge base. I can help you with the project "
            "details or connect you with the JB Infra team for the "
            "latest pricing."
        ),
    },
    {
        "topic": "availability",
        "source": "knowledge_base/agent_guidelines.md",
        "pattern": re.compile(
            r"\b("
            r"is\s+(?:it|this)\s+available|"
            r"are\s+(?:plots|sites)\s+available|"
            r"plot\s+availability|"
            r"available\s+plots|"
            r"which\s+plots\s+are\s+available"
            r")\b",
            re.IGNORECASE,
        ),
        "response": (
            "I don't have verified real-time plot availability. "
            "Please contact the JB Infra team for the latest "
            "availability."
        ),
    },
]


def get_deterministic_fast_path(
    query: str,
) -> Optional[Dict[str, Any]]:
    """Return a deterministic response for a recognized common query."""

    t_start = time.perf_counter()

    q = query.strip()

    if not q:
        return None

    for rule in FAST_PATH_RULES:
        if rule["pattern"].search(q):
            latency_ms = round(
                (time.perf_counter() - t_start) * 1000,
                2,
            )

            return {
                "response": rule["response"],
                "topic": rule["topic"],
                "source": rule["source"],
                "confidence": 0.99,
                "latency_ms": latency_ms,
                "bypassed_llm": True,
            }

    return None


__all__ = [
    "FAST_PATH_RULES",
    "get_deterministic_fast_path",
]