
"""Infra Builder domain configuration.

Defines the real-estate intents, buyer-information slots, and persona
guidelines used by the shared conversational AI voice agent.
"""

from src.core.types import DomainType
from src.domains.base import DomainConfig, SlotDefinition


INFRA_BUILDER_DOMAIN_CONFIG = DomainConfig(
    domain=DomainType.INFRA_BUILDER,
    name="Infra Builder",
    description=(
        "Real-estate customer assistance for inbound and outbound calls, "
        "including property inquiries, project information, buyer requirements, "
        "site visits, and authorized WhatsApp information sharing."
    ),
    supported_intents=[
        # Conversation
        "greeting",
        "identity_inquiry",
        "general_inquiry",
        "general_question",
        "purpose_discovery",
        "fallback",
        "help",
        "clarification",
        "confirmation",
        "correction",
        "refusal",
        "not_interested",
        "busy",
        "callback_request",
        "human_agent_request",
        "end_call",
        "termination",

        # Property information
        "property_search",
        "project_information",
        "property_type_query",
        "location_query",
        "price_query",
        "budget_discussion",
        "availability_query",
        "amenities_query",
        "configuration_query",
        "possession_date_query",
        "payment_plan_query",
        "property_comparison",
        "buying_requirements",
        "buying_timeline",

        # Site visits and appointments
        "site_visit_inquiry",
        "site_visit_request",
        "site_visit_reschedule",
        "site_visit_cancellation",
        "appointment_confirmation",
        "appointment_status",

        # WhatsApp
        "whatsapp_information_request",
        "whatsapp_consent",
        "whatsapp_refusal",
        "whatsapp_number_provided",
        "whatsapp_number_correction",

        # Follow-up
        "human_followup_request",
        "human_followup_cancellation",
    ],
    supported_slots=[
        SlotDefinition(
            name="name",
            slot_type="string",
            description="Caller name, only when explicitly provided or reliably known.",
        ),
        SlotDefinition(
            name="phone",
            slot_type="string",
            description=(
                "Caller or recipient phone number from the caller or an "
                "authorized trusted source. Never invent or infer a number."
            ),
        ),
        SlotDefinition(
            name="preferred_project",
            slot_type="string",
            description="Specific Infra Builder project the caller is interested in.",
        ),
        SlotDefinition(
            name="preferred_location",
            slot_type="string",
            description="Location or area preferred by the caller.",
        ),
        SlotDefinition(
            name="property_type",
            slot_type="string",
            description="Requested property category, such as apartment, villa, or plot.",
        ),
        SlotDefinition(
            name="configuration",
            slot_type="string",
            description="Requested configuration, such as 2 BHK, 3 BHK, or plot size.",
        ),
        SlotDefinition(
            name="budget",
            slot_type="string",
            description="Budget explicitly stated by the caller, preserving its amount and currency.",
        ),
        SlotDefinition(
            name="intended_use",
            slot_type="string",
            description="Caller-stated purpose, such as residential use or investment.",
        ),
        SlotDefinition(
            name="amenities",
            slot_type="string",
            description="Amenities requested by the caller.",
        ),
        SlotDefinition(
            name="purchase_timeline",
            slot_type="string",
            description="Caller-stated expected purchase timeline.",
        ),
        SlotDefinition(
            name="possession_preference",
            slot_type="string",
            description="Caller-stated possession or move-in preference.",
        ),
        SlotDefinition(
            name="meeting_preference",
            slot_type="string",
            description="Requested site-visit date or time preference.",
        ),
        SlotDefinition(
            name="whatsapp_consent",
            slot_type="string",
            description=(
                "Consent status for the specific WhatsApp message being requested. "
                "Do not infer consent from general property interest."
            ),
        ),
        SlotDefinition(
            name="whatsapp_message_purpose",
            slot_type="string",
            description=(
                "Exact information the caller requested through WhatsApp, "
                "such as a brochure, project details, or a verified link."
            ),
        ),
        SlotDefinition(
            name="site_visit_status",
            slot_type="string",
            description=(
                "Site-visit status only when supported by actual application "
                "or booking-system evidence."
            ),
        ),
        SlotDefinition(
            name="human_followup_requested",
            slot_type="string",
            description=(
                "Whether the caller explicitly requested human representative "
                "follow-up. Do not infer a request from general interest."
            ),
        ),
    ],
    persona_guidelines=(
        "Act as a professional, helpful, and natural real-estate representative "
        "for Infra Builder. Support inbound customer inquiries and outbound "
        "property follow-ups. Prioritize the caller's latest explicit intent, "
        "answer questions before qualifying the buyer, and use approved "
        "knowledge for all project-specific facts. Understand property needs "
        "without asking unnecessary questions. Be informative and persuasive "
        "without pressure, artificial urgency, or unsupported promises. "
        "Arrange site visits and share property information through WhatsApp "
        "only when requested or appropriately authorized and when the required "
        "tools are available. Never invent property details, contact numbers, "
        "availability, bookings, or action results."
    ),
)
