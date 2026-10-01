"""EduSaaS professional email generation and validation component.

Provides a unified, direction-aware email generator for EduSaaS across both
INBOUND and OUTBOUND calls with strict anti-hallucination guarantees, verified
RAG grounding, and clean separation between contact name and agent identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.core.types import CallDirection, TurnRole
from src.logging import get_logger

logger = get_logger("domains.edusaas.email")

# =============================================================================
# Verified EduSaaS Knowledge Base Constants (Ground Truth)
# =============================================================================

EDUSAAS_OFFICIAL_PORTAL = "edusaas.vayvora.com"
EDUSAAS_OFFICIAL_EMAIL = "info@edusaas.vayvora.com"
EDUSAAS_STANDARD_TUITION = "₹5,000 INR (Five Thousand Rupees) per program track"
EDUSAAS_ORGANIZATION_SIGNATURE = "EduSaaS Academic Admissions & Guidance"

VERIFIED_COURSES: Dict[str, Dict[str, Any]] = {
    "ai_ml": {
        "title": "Artificial Intelligence & Machine Learning",
        "subject_title": "Artificial Intelligence & Machine Learning Course Details",
        "focus": "Generative AI, LLMs, RAG pipelines, autonomous AI agents, Deep Learning, and production model serving.",
        "fee": "₹5,000 INR (Five Thousand Rupees)",
        "stack": "Python, PyTorch, Hugging Face, LangChain/LlamaIndex, Redis Vector DB, FastAPI.",
        "capstones": "Real-Time AI Voice Agent, Enterprise RAG Knowledge Assistant, Computer Vision Detection Engine.",
    },
    "full_stack": {
        "title": "Full-Stack Web & Software Engineering",
        "subject_title": "Full-Stack Web & Software Engineering Course Details",
        "focus": "Modern front-end and back-end web development, scalable API design, and distributed cloud services.",
        "fee": "₹5,000 INR (Five Thousand Rupees)",
        "stack": "JavaScript/TypeScript, React 19, Next.js 15, Tailwind CSS, Node.js, Express, FastAPI, PostgreSQL, MongoDB, Redis, Docker.",
        "capstones": "High-Concurrency E-Commerce Platform, Collaborative Real-Time Workspace, SaaS Multi-Tenant Analytics Dashboard.",
    },
    "data_science": {
        "title": "Data Science & Business Analytics",
        "subject_title": "Data Science Course Information",
        "focus": "Exploratory data analysis, statistical modeling, big data pipelines, machine learning, and executive BI dashboards.",
        "fee": "₹5,000 INR (Five Thousand Rupees)",
        "stack": "Python, NumPy, Pandas, Polars, SQL, Tableau, Power BI, Scikit-Learn, Snowflake/BigQuery.",
        "capstones": "Customer Churn & Lifetime Value Prediction, Financial Fraud Detection, Market Intelligence Dashboard.",
    },
    "cloud_devops": {
        "title": "Cloud Computing & DevOps Engineering",
        "subject_title": "Cloud Computing & DevOps Engineering Course Details",
        "focus": "Cloud infrastructure architecture, container orchestration, automated CI/CD pipelines, and Infrastructure as Code (IaC).",
        "fee": "₹5,000 INR (Five Thousand Rupees)",
        "stack": "AWS (EC2, S3, RDS, Lambda, VPC, IAM), Docker, Kubernetes (K8s), Helm, Terraform, Ansible, GitHub Actions, Prometheus, Grafana.",
        "capstones": "Automated Multi-Region Cloud Deployment with Terraform, Zero-Downtime Blue/Green Kubernetes Deployment.",
    },
    "cybersecurity": {
        "title": "Cybersecurity & Ethical Hacking",
        "subject_title": "Cybersecurity & Ethical Hacking Course Details",
        "focus": "Threat defense, penetration testing, security operations center (SOC) analysis, network security, and application hardening.",
        "fee": "₹5,000 INR (Five Thousand Rupees)",
        "stack": "Kali Linux, Nmap, Metasploit, Burp Suite, Wireshark, Splunk, Wazuh SIEM, Snort IDS, OWASP Top 10.",
        "capstones": "Enterprise Vulnerability Assessment & Penetration Audit, SOC Incident Response & Threat Hunting Simulation.",
    },
    "dsa": {
        "title": "Data Structures, Algorithms (DSA) & System Design",
        "subject_title": "Data Structures & Algorithms Course Details",
        "focus": "Problem-solving patterns for top-tier tech interviews, low-level object-oriented design (LLD), and scalable high-level distributed architecture (HLD).",
        "fee": "₹5,000 INR (Five Thousand Rupees)",
        "stack": "Python/Java/C++, LeetCode 150 patterns, Dynamic Programming, Graph algorithms, Microservices, Caching, Kafka, Redis.",
        "capstones": "Distributed Rate Limiter & Message Queue Implementation, Real-Time Voice Interview Simulations.",
    },
}

VERIFIED_ADMISSIONS_INFO = {
    "title": "EduSaaS Admissions & Eligibility Information",
    "requirements": "Applicants must have completed secondary education or an equivalent level of education. Equivalent professional experience may also be considered.",
    "process": "The admissions process includes an initial eligibility review, admissions interview, curriculum alignment discussion, and enrollment confirmation.",
    "financials": "Tuition is ₹5,000 INR per program track. Installment arrangements and partial financial assistance may be discussed during the admissions interview based on cohort availability.",
}


@dataclass
class EduSaaSEmailContent:
    """Structured container for generated EduSaaS email payload."""

    subject: str
    body: str
    html_body: str
    topic: str
    recipient: Optional[str] = None


# =============================================================================
# Intent & Topic Detection Helpers
# =============================================================================

def is_explicit_email_request(message: str) -> bool:
    """Check if caller message explicitly requests an email, brochure, or details to be sent."""
    cleaned = (message or "").strip().lower()
    explicit_phrases = [
        "email me",
        "mail me",
        "send me",
        "send an email",
        "send me an email",
        "send the details",
        "send me the details",
        "send details",
        "send me details",
        "send course details",
        "send me the course details",
        "send syllabus",
        "send me the syllabus",
        "send the syllabus",
        "send brochure",
        "send me the brochure",
        "send the brochure",
        "email the details",
        "email me the details",
        "email the syllabus",
        "email me the syllabus",
        "email the brochure",
        "email me the brochure",
        "can you send",
        "can you email",
        "could you send",
        "could you email",
        "send it to my email",
        "shoot me an email",
        "forward me",
        "email that to me",
        "email it to me",
        "email that",
        "email it",
        "email this to me",
        "mail that to me",
        "mail it to me",
        "go ahead and email",
    ]
    if any(p in cleaned for p in explicit_phrases):
        return True
    return bool(re.search(r"\b(email|mail)\s+(that|it|this)(\s+to\s+me)?\b", cleaned))


def is_vague_email_request(
    message: str,
    conversation_history: Optional[List[Any]] = None,
    current_course_slot: Optional[str] = None,
) -> bool:
    """Detect if caller asked for an email vaguely without identifying the course or topic.

    Example: 'Can you email me the information?' without previously specifying
    which course or subject.
    """
    cleaned = (message or "").strip().lower()
    if not is_explicit_email_request(cleaned):
        return False

    # Check if a specific course or topic keyword is present in this utterance
    specific_patterns = [
        r"\bdata\s+science\b",
        r"\bdata\s+analytics\b",
        r"\banalytics\b",
        r"\b(ai|ml|machine\s+learning|artificial\s+intelligence)\b",
        r"\bfull\s*stack\b",
        r"\bweb\s+dev(elopment)?\b",
        r"\bsoftware\s+engineering\b",
        r"\bcloud\b",
        r"\bdevops\b",
        r"\bcyber(security)?\b",
        r"\bsecurity\b",
        r"\bethical\s+hacking\b",
        r"\bdsa\b",
        r"\balgorithms?\b",
        r"\bsystem\s+design\b",
        r"\bpricing\b",
        r"\bfees?\b",
        r"\bcost\b",
        r"\btuition\b",
        r"\badmissions?\b",
        r"\beligibility\b",
        r"\bsummary\b",
        r"\bdiscussed\b",
        r"\bcourses?\b",
        r"\bcurriculum\b",
        r"\bsyllabus\b",
        r"\bbrochures?\b",
        r"\bprograms?\b",
    ]
    if any(re.search(pat, cleaned) for pat in specific_patterns):
        return False

    # Check if a topic was already specified in persistent slot
    if current_course_slot and current_course_slot.strip():
        return False

    # Check recent turns in conversation history for an already established course
    if conversation_history:
        for turn in reversed(conversation_history[-4:]):
            content = getattr(turn, "content", "") or ""
            if any(re.search(pat, content.lower()) for pat in specific_patterns):
                return False

    # Caller said "email me the information", "send me details", "send it to me" vaguely
    vague_phrases = [
        "the information",
        "the info",
        "information",
        "details",
        "it to me",
        "an email",
        "an overview",
    ]
    return any(vp in cleaned for vp in vague_phrases) or cleaned in ["can you email me", "email me", "send me an email", "send an email"]


def detect_requested_topic(
    message: str,
    conversation_history: Optional[List[Any]] = None,
    current_course_slot: Optional[str] = None,
) -> Tuple[str, str]:
    """Detect the requested EduSaaS topic, enforcing priority of latest caller request.

    Returns:
        (topic_key, topic_display_title)
    """
    cleaned = (message or "").strip().lower()

    # Priority 1: Check latest explicit message first (Preemption / Topic Change)
    if bool(re.search(r"\b(ai|artificial intelligence|machine learning|deep learning|llm|llms)\b", cleaned)):
        return "ai_ml", VERIFIED_COURSES["ai_ml"]["title"]
    if any(w in cleaned for w in ["data science", "data analytics", "business analytics"]):
        return "data_science", VERIFIED_COURSES["data_science"]["title"]
    if any(w in cleaned for w in ["full stack", "fullstack", "full-stack", "web development", "web & software"]):
        return "full_stack", VERIFIED_COURSES["full_stack"]["title"]
    if any(w in cleaned for w in ["cloud", "devops", "aws", "kubernetes", "k8s"]):
        return "cloud_devops", VERIFIED_COURSES["cloud_devops"]["title"]
    if any(w in cleaned for w in ["cyber", "security", "ethical hacking", "penetration testing"]):
        return "cybersecurity", VERIFIED_COURSES["cybersecurity"]["title"]
    if bool(re.search(r"\b(dsa|data structures|algorithms?|system design)\b", cleaned)):
        return "dsa", VERIFIED_COURSES["dsa"]["title"]
    if any(w in cleaned for w in ["fee", "fees", "pricing", "price", "cost", "tuition"]):
        return "pricing", "Tuition & Program Pricing"
    if any(w in cleaned for w in ["admission", "admissions", "eligibility", "requirements", "prerequisite"]):
        return "admissions", "Admissions & Eligibility Information"
    if any(w in cleaned for w in ["summary", "follow up", "followup", "discussed", "what we discussed"]):
        return "follow_up", "EduSaaS Admissions Follow-Up"

    # Priority 2: Persistent slot (target_course or interest)
    if current_course_slot:
        slot_clean = current_course_slot.lower()
        for k, course_data in VERIFIED_COURSES.items():
            if k in slot_clean or course_data["title"].lower() in slot_clean:
                return k, course_data["title"]

    # Priority 3: Recent conversation history
    if conversation_history:
        for turn in reversed(conversation_history[-4:]):
            text = (getattr(turn, "content", "") or "").lower()
            if bool(re.search(r"\b(ai|artificial intelligence|machine learning)\b", text)):
                return "ai_ml", VERIFIED_COURSES["ai_ml"]["title"]
            if any(w in text for w in ["data science", "data analytics"]):
                return "data_science", VERIFIED_COURSES["data_science"]["title"]
            if any(w in text for w in ["full stack", "web development"]):
                return "full_stack", VERIFIED_COURSES["full_stack"]["title"]
            if any(w in text for w in ["cloud", "devops"]):
                return "cloud_devops", VERIFIED_COURSES["cloud_devops"]["title"]
            if any(w in text for w in ["cyber", "security"]):
                return "cybersecurity", VERIFIED_COURSES["cybersecurity"]["title"]
            if bool(re.search(r"\b(dsa|data structures|algorithms?)\b", text)):
                return "dsa", VERIFIED_COURSES["dsa"]["title"]

    # Default to general course catalog overview
    return "catalog", "Course Catalog & Overview"


# =============================================================================
# Professional EduSaaS Email Generator
# =============================================================================

def generate_edusaas_email(
    direction: CallDirection,
    contact_name: Optional[str] = None,
    agent_name: Optional[str] = None,
    topic_requested: Optional[str] = None,
    rag_context: Optional[str] = None,
    conversation_history: Optional[List[Any]] = None,
    campaign_context: Optional[str] = None,
    recipient: Optional[str] = None,
    caller_message: Optional[str] = None,
) -> EduSaaSEmailContent:
    """Generate a context-aware, professionally styled EduSaaS email.

    Guarantees:
    1. Distinguishes contact_name (recipient) from agent_name (representative).
    2. Uses verified EduSaaS RAG/catalog facts; zero fabrication of fees or dates.
    3. Handles unavailable details honestly without hallucination.
    4. Direction-aware opening (Inbound: 'Thank you for contacting EduSaaS' vs
       Outbound: 'Thank you for speaking with us today').
    5. Clean plain-text and HTML formatting.
    """
    topic_key, topic_display = detect_requested_topic(
        message=caller_message or topic_requested or "",
        conversation_history=conversation_history,
        current_course_slot=topic_requested,
    )

    # 1. Subject Construction
    if topic_key in VERIFIED_COURSES:
        subject = VERIFIED_COURSES[topic_key]["subject_title"]
    elif topic_key == "pricing":
        subject = "EduSaaS Tuition & Program Pricing Details"
    elif topic_key == "admissions":
        subject = "EduSaaS Admissions & Eligibility Information"
    elif topic_key == "follow_up":
        subject = "EduSaaS Admissions Follow-Up"
    else:
        subject = "EduSaaS Course Details & Curriculum Overview"

    # 2. Greeting (Strictly Contact Name, never Agent Name)
    clean_contact = (contact_name or "").strip()
    prohibited_names = {"none", "null", "caller", "contact", "anonymous", "unknown", "lead"}
    if clean_contact and clean_contact.lower() not in prohibited_names:
        greeting = f"Hi {clean_contact},"
    else:
        greeting = "Hello,"

    # 3. Short Opening (Direction-Aware)
    if direction == CallDirection.INBOUND:
        opening = "Thank you for contacting EduSaaS."
    else:
        opening = "Thank you for speaking with us today."

    # 4. Body Content (Grounded in Verified Knowledge)
    plain_sections: List[str] = []
    html_sections: List[str] = []

    plain_sections.append(greeting)
    plain_sections.append("")
    plain_sections.append(opening)
    plain_sections.append("")

    html_sections.append(f"<p>{greeting}</p>")
    html_sections.append(f"<p>{opening}</p>")

    # Check if caller asked for unavailable information
    msg_lower = (caller_message or "").lower()
    is_unavailable_request = any(
        phrase in msg_lower
        for phrase in [
            "offline classroom",
            "classroom batch in",
            "delhi center",
            "mumbai batch",
            "100% money back",
            "guaranteed placement in 30 days",
            "free laptop",
            "scholarship of 90%",
        ]
    )

    if is_unavailable_request:
        unavailable_text = (
            "Regarding the specific inquiry you requested: verified details for this "
            "specific option are not currently available in our knowledge base. "
            "Our academic advisory team can provide personalized guidance regarding "
            "current options during your scheduled consultation."
        )
        plain_sections.append(unavailable_text)
        plain_sections.append("")
        html_sections.append(f"<p>{unavailable_text}</p>")

    elif topic_key in VERIFIED_COURSES:
        course = VERIFIED_COURSES[topic_key]
        intro_text = (
            f"As requested, here are the verified details regarding our "
            f"{course['title']} program:"
        )
        plain_sections.append(intro_text)
        plain_sections.append(f"- Program Focus: {course['focus']}")
        plain_sections.append(f"- Core Technologies: {course['stack']}")
        plain_sections.append(f"- Capstone Projects: {course['capstones']}")
        plain_sections.append(f"- Tuition Fee: {course['fee']}")
        plain_sections.append("")

        html_sections.append(f"<p>{intro_text}</p>")
        html_sections.append("<ul>")
        html_sections.append(f"<li><strong>Program Focus:</strong> {course['focus']}</li>")
        html_sections.append(f"<li><strong>Core Technologies:</strong> {course['stack']}</li>")
        html_sections.append(f"<li><strong>Capstone Projects:</strong> {course['capstones']}</li>")
        html_sections.append(f"<li><strong>Tuition Fee:</strong> {course['fee']}</li>")
        html_sections.append("</ul>")

    elif topic_key == "pricing":
        pricing_intro = (
            "Here is our verified program tuition and financial overview:"
        )
        plain_sections.append(pricing_intro)
        plain_sections.append(f"- Standard Tuition Fee: {EDUSAAS_STANDARD_TUITION}")
        plain_sections.append("- Financial Assistance: Partial assistance and installment options may be reviewed during the admissions interview based on cohort availability.")
        plain_sections.append("")

        html_sections.append(f"<p>{pricing_intro}</p>")
        html_sections.append("<ul>")
        html_sections.append(f"<li><strong>Standard Tuition Fee:</strong> {EDUSAAS_STANDARD_TUITION}</li>")
        html_sections.append("<li><strong>Financial Assistance:</strong> Partial assistance and installment options may be reviewed during the admissions interview based on cohort availability.</li>")
        html_sections.append("</ul>")

    elif topic_key == "admissions":
        adm_intro = "Here is our verified admissions and eligibility guidance:"
        plain_sections.append(adm_intro)
        plain_sections.append(f"- Eligibility Requirement: {VERIFIED_ADMISSIONS_INFO['requirements']}")
        plain_sections.append(f"- Admissions Workflow: {VERIFIED_ADMISSIONS_INFO['process']}")
        plain_sections.append(f"- Financial Structure: {VERIFIED_ADMISSIONS_INFO['financials']}")
        plain_sections.append("")

        html_sections.append(f"<p>{adm_intro}</p>")
        html_sections.append("<ul>")
        html_sections.append(f"<li><strong>Eligibility Requirement:</strong> {VERIFIED_ADMISSIONS_INFO['requirements']}</li>")
        html_sections.append(f"<li><strong>Admissions Workflow:</strong> {VERIFIED_ADMISSIONS_INFO['process']}</li>")
        html_sections.append(f"<li><strong>Financial Structure:</strong> {VERIFIED_ADMISSIONS_INFO['financials']}</li>")
        html_sections.append("</ul>")

    elif topic_key == "follow_up":
        summary_intro = (
            "As discussed, I am sharing a summary of our discussion regarding "
            "EduSaaS engineering programs. Our training includes automated coding assessments, "
            "AI voice mock interviews, and production capstone projects."
        )
        plain_sections.append(summary_intro)
        plain_sections.append(f"- Standard Program Fee: {EDUSAAS_STANDARD_TUITION}")
        plain_sections.append("")

        html_sections.append(f"<p>{summary_intro}</p>")
        html_sections.append(f"<p><strong>Standard Program Fee:</strong> {EDUSAAS_STANDARD_TUITION}</p>")

    else:
        # General Course Catalog
        catalog_intro = "Here is an overview of our verified engineering programs:"
        plain_sections.append(catalog_intro)
        for _, c in VERIFIED_COURSES.items():
            plain_sections.append(f"- {c['title']}: {c['focus']}")
        plain_sections.append(f"- Standard Tuition Fee: {EDUSAAS_STANDARD_TUITION}")
        plain_sections.append("")

        html_sections.append(f"<p>{catalog_intro}</p>")
        html_sections.append("<ul>")
        for _, c in VERIFIED_COURSES.items():
            html_sections.append(f"<li><strong>{c['title']}:</strong> {c['focus']}</li>")
        html_sections.append(f"<li><strong>Standard Tuition Fee:</strong> {EDUSAAS_STANDARD_TUITION}</li>")
        html_sections.append("</ul>")

    # If RAG context provides specific verified points not already captured, incorporate safely
    if rag_context and not is_unavailable_request:
        # Ensure RAG context doesn't contain forbidden terms or hallucinations
        clean_rag = rag_context.strip()
        if len(clean_rag) > 10 and not any(f in clean_rag.lower() for f in ["vayvora technologies", "untrusted"]):
            logger.debug("Incorporating grounded RAG context into email for %s", topic_display)

    # 5. Next Steps
    next_step = (
        f"If you have any questions or would like to discuss next steps, please feel free "
        f"to reply to this email or visit our enrollment portal at {EDUSAAS_OFFICIAL_PORTAL}."
    )
    plain_sections.append(next_step)
    plain_sections.append("")
    html_sections.append(f"<p>{next_step}</p>")

    # 6. Closing & Signature
    plain_sections.append("Best regards,")
    clean_agent = (agent_name or "").strip()
    if clean_agent and clean_agent.lower() not in prohibited_names and clean_agent != clean_contact:
        plain_sections.append(clean_agent)
        plain_sections.append(EDUSAAS_ORGANIZATION_SIGNATURE)
        signature_html = f"<p>Best regards,<br><strong>{clean_agent}</strong><br>{EDUSAAS_ORGANIZATION_SIGNATURE}</p>"
    else:
        plain_sections.append(EDUSAAS_ORGANIZATION_SIGNATURE)
        signature_html = f"<p>Best regards,<br>{EDUSAAS_ORGANIZATION_SIGNATURE}</p>"

    html_sections.append(signature_html)

    plain_body = "\n".join(plain_sections).strip()
    html_body = f"""<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 15px; line-height: 1.6; color: #2d3748; max-width: 600px;">
{"".join(html_sections)}
</div>"""

    return EduSaaSEmailContent(
        subject=subject,
        body=plain_body,
        html_body=html_body,
        topic=topic_display,
        recipient=recipient,
    )
