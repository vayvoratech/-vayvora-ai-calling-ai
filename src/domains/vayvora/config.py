"""Vayvora domain configuration, supported intents, and entity slots."""

from src.core.types import DomainType
from src.domains.base import DomainConfig, SlotDefinition

VAYVORA_INTENTS = [
    "company_information",
    "company_location",
    "career_information",
    "job_application",
    "job_roles",
    "job_requirements",
    "product_information",
    "ai_solution",
    "voice_ai",
    "client_requirement",
    "custom_solution",
    "meeting_request",
    "general_information",
]

VAYVORA_SLOTS = [
    SlotDefinition(
        name="contact_name",
        slot_type="string",
        description="Full name of the job applicant or corporate client representative",
        example="Priya Patel",
    ),
    SlotDefinition(
        name="email",
        slot_type="email",
        description="Email address for correspondence, job follow-up, or sales decks",
        example="priya.patel@acme-corp.com",
    ),
    SlotDefinition(
        name="company_name",
        slot_type="string",
        description="Name of the client organization or current employer",
        example="Acme Technologies Ltd.",
    ),
    SlotDefinition(
        name="caller_type",
        slot_type="string",
        description="Category of caller: job_seeker, corporate_client, or partner",
        example="corporate_client",
    ),
    SlotDefinition(
        name="desired_role",
        slot_type="string",
        description="Role applied for by a job seeker (e.g. Senior AI Engineer)",
        example="Senior AI Engineer",
    ),
    SlotDefinition(
        name="experience",
        slot_type="string",
        description="Candidate years of experience or technical background",
        example="4 years building LLM and voice pipelines",
    ),
    SlotDefinition(
        name="technical_requirement",
        slot_type="string",
        description="Corporate client's specific software or AI requirements",
        example="Real-time multi-agent customer service voice bot",
    ),
    SlotDefinition(
        name="product_interest",
        slot_type="string",
        description="Specific Vayvora product or AI solution of interest",
        example="Vayvora Voice AI Platform",
    ),
    SlotDefinition(
        name="meeting_preference",
        slot_type="string",
        description="Preferred date, time, or timezone for discovery or interview meeting",
        example="Next Tuesday at 3:00 PM IST",
    ),
]

VAYVORA_DOMAIN_CONFIG = DomainConfig(
    domain=DomainType.VAYVORA,
    name="Vayvora Technologies - AI & Software Engineering",
    description="Enterprise AI solutions, custom software development, and engineering career opportunities.",
    supported_intents=VAYVORA_INTENTS,
    supported_slots=VAYVORA_SLOTS,
    persona_guidelines=(
        "You are an articulate, professional, and courteous representative for Vayvora. "
        "For corporate clients, discover technical requirements and coordinate discovery meetings. "
        "For career seekers, explain engineering culture, roles, office locations, and capture application details."
    ),
)
