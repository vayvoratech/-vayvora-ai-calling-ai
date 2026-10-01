"""Unit tests for domain configurations and domain registry."""

import pytest
from src.core.types import DomainType
from src.domains import (
    EDUSAAS_DOMAIN_CONFIG,
    EDUSAAS_INTENTS,
    EDUSAAS_SLOTS,
    GENERAL_DOMAIN_CONFIG,
    VAYVORA_DOMAIN_CONFIG,
    VAYVORA_INTENTS,
    VAYVORA_SLOTS,
    DomainRegistry,
    get_domain_registry,
)


class TestEduSaaSDomain:
    """Verify EduSaaS domain metadata, intents, and slot definitions."""

    def test_edusaas_config_metadata(self):
        assert EDUSAAS_DOMAIN_CONFIG.domain == DomainType.EDUSAAS
        assert "EduSaaS" in EDUSAAS_DOMAIN_CONFIG.name
        assert len(EDUSAAS_DOMAIN_CONFIG.supported_intents) >= 11
        assert len(EDUSAAS_DOMAIN_CONFIG.supported_slots) >= 9

    def test_edusaas_required_intents(self):
        required_intents = [
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
        for intent in required_intents:
            assert EDUSAAS_DOMAIN_CONFIG.is_intent_supported(intent) is True

    def test_edusaas_required_slots(self):
        required_slots = [
            "student_name",
            "email",
            "education",
            "year",
            "experience",
            "interest",
            "target_course",
            "preferred_schedule",
            "enrollment_interest",
        ]
        slot_names = EDUSAAS_DOMAIN_CONFIG.get_slot_names()
        for slot in required_slots:
            assert slot in slot_names
            definition = EDUSAAS_DOMAIN_CONFIG.get_slot_definition(slot)
            assert definition is not None
            assert definition.name == slot


class TestVayvoraDomain:
    """Verify Vayvora domain metadata, intents, and slot definitions."""

    def test_vayvora_config_metadata(self):
        assert VAYVORA_DOMAIN_CONFIG.domain == DomainType.VAYVORA
        assert "Vayvora" in VAYVORA_DOMAIN_CONFIG.name
        assert len(VAYVORA_DOMAIN_CONFIG.supported_intents) >= 13
        assert len(VAYVORA_DOMAIN_CONFIG.supported_slots) >= 9

    def test_vayvora_required_intents(self):
        required_intents = [
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
        for intent in required_intents:
            assert VAYVORA_DOMAIN_CONFIG.is_intent_supported(intent) is True

    def test_vayvora_required_slots(self):
        required_slots = [
            "contact_name",
            "email",
            "company_name",
            "caller_type",
            "desired_role",
            "experience",
            "technical_requirement",
            "product_interest",
            "meeting_preference",
        ]
        slot_names = VAYVORA_DOMAIN_CONFIG.get_slot_names()
        for slot in required_slots:
            assert slot in slot_names
            definition = VAYVORA_DOMAIN_CONFIG.get_slot_definition(slot)
            assert definition is not None
            assert definition.name == slot


class TestDomainRegistry:
    """Test registry retrieval and lookups."""

    def test_registry_lookup(self):
        registry = get_domain_registry()
        assert registry.get(DomainType.EDUSAAS) == EDUSAAS_DOMAIN_CONFIG
        assert registry.get(DomainType.VAYVORA) == VAYVORA_DOMAIN_CONFIG
        assert registry.get(DomainType.GENERAL) == GENERAL_DOMAIN_CONFIG

    def test_unregistered_domain_handling(self):
        registry = DomainRegistry()
        with pytest.raises(KeyError):
            registry.get_or_raise("unregistered_domain_key")
