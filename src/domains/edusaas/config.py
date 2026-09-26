"""EduSaaS domain configuration, supported intents, and entity slots."""

from src.core.types import DomainType
from src.domains.base import DomainConfig, SlotDefinition

EDUSAAS_INTENTS = [
    "course_information",
    "course_recommendation",
    "course_comparison",
    "assessment_information",
    "enrollment",
    "eligibility",
    "pricing",
    "duration",
    "student_support",
    "objection",
    "general_information",
]

EDUSAAS_SLOTS = [
    SlotDefinition(
        name="student_name",
        slot_type="string",
        description="Full name of the prospective or enrolled student",
        example="Aarav Sharma",
    ),
    SlotDefinition(
        name="email",
        slot_type="email",
        description="Email address of the student for sending syllabus or enrollment details",
        example="aarav.sharma@example.com",
    ),
    SlotDefinition(
        name="education",
        slot_type="string",
        description="Highest academic qualification or current degree",
        example="B.Tech Computer Science",
    ),
    SlotDefinition(
        name="year",
        slot_type="string",
        description="Current academic year or graduation year",
        example="Final year (2025)",
    ),
    SlotDefinition(
        name="experience",
        slot_type="string",
        description="Prior coding, technical, or industry experience level",
        example="Beginner with basic Python knowledge",
    ),
    SlotDefinition(
        name="interest",
        slot_type="string",
        description="Primary technical domain or career aspiration",
        example="Machine Learning and AI Engineering",
    ),
    SlotDefinition(
        name="target_course",
        slot_type="string",
        description="Specific course or program the student is inquiring about",
        example="Applied AI Specialist Certification",
    ),
    SlotDefinition(
        name="preferred_schedule",
        slot_type="string",
        description="Preferred batch timing or mode (e.g. weekend, evening, self-paced)",
        example="Weekend batches",
    ),
    SlotDefinition(
        name="enrollment_interest",
        slot_type="string",
        description="Level of urgency or commitment to enrolling",
        example="Planning to start within 2 weeks",
    ),
]

EDUSAAS_DOMAIN_CONFIG = DomainConfig(
    domain=DomainType.EDUSAAS,
    name="EduSaaS Academic Admissions & Guidance",
    description="Educational advisory and course counseling domain for students and professionals.",
    supported_intents=EDUSAAS_INTENTS,
    supported_slots=EDUSAAS_SLOTS,
    persona_guidelines=(
        "You are an empathetic, encouraging, and knowledgeable educational counselor for EduSaaS. "
        "Your goal is to guide prospective students toward the best course matching their background, "
        "address doubts clearly, explain schedules and curriculum factually based on RAG knowledge, "
        "and assist with enrollment next steps."
    ),
)
