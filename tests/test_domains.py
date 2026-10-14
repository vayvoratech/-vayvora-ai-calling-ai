"""Unit tests for domain configurations and domain registry."""

import pytest
from src.core.types import DomainType
from src.domains import (
    INFRA_BUILDER_DOMAIN_CONFIG,
    GENERAL_DOMAIN_CONFIG,
    UNKNOWN_DOMAIN_CONFIG,
    DomainRegistry,
    get_domain_registry,
)


class TestInfraBuilderDomain:
    """Verify Infra Builder domain metadata, intents, and slot definitions."""

    def test_infra_builder_config_metadata(self):
        assert INFRA_BUILDER_DOMAIN_CONFIG.domain == DomainType.INFRA_BUILDER
        assert "Infra Builder" in INFRA_BUILDER_DOMAIN_CONFIG.name
        assert len(INFRA_BUILDER_DOMAIN_CONFIG.supported_intents) >= 25
        assert len(INFRA_BUILDER_DOMAIN_CONFIG.supported_slots) >= 10

    def test_infra_builder_required_intents(self):
        required_intents = [
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
            "site_visit_inquiry",
            "site_visit_request",
            "whatsapp_information_request",
            "whatsapp_consent",
        ]
        for intent in required_intents:
            assert INFRA_BUILDER_DOMAIN_CONFIG.is_intent_supported(intent) is True

    def test_infra_builder_required_slots(self):
        required_slots = [
            "name",
            "phone",
            "preferred_project",
            "preferred_location",
            "property_type",
            "configuration",
            "budget",
            "intended_use",
            "amenities",
            "purchase_timeline",
            "possession_preference",
            "meeting_preference",
            "whatsapp_consent",
            "whatsapp_message_purpose",
            "site_visit_status",
            "human_followup_requested",
        ]
        slot_names = INFRA_BUILDER_DOMAIN_CONFIG.get_slot_names()
        for slot in required_slots:
            assert slot in slot_names
            definition = INFRA_BUILDER_DOMAIN_CONFIG.get_slot_definition(slot)
            assert definition is not None
            assert definition.name == slot


class TestDomainRegistry:
    """Test registry retrieval, lookups, and legacy business exclusion."""

    def test_registry_lookup(self):
        registry = get_domain_registry()
        assert registry.get(DomainType.INFRA_BUILDER) == INFRA_BUILDER_DOMAIN_CONFIG
        assert registry.get(DomainType.GENERAL) == GENERAL_DOMAIN_CONFIG
        assert registry.get(DomainType.UNKNOWN) == UNKNOWN_DOMAIN_CONFIG

    def test_legacy_domains_not_registered(self):
        registry = get_domain_registry()
        # EduSaaS and Vayvora must NOT be registered as active domains
        assert registry.get(DomainType.EDUSAAS) is None
        assert registry.get(DomainType.VAYVORA) is None
        with pytest.raises(KeyError):
            registry.get_or_raise(DomainType.EDUSAAS)
        with pytest.raises(KeyError):
            registry.get_or_raise(DomainType.VAYVORA)

    def test_unregistered_domain_handling(self):
        registry = DomainRegistry()
        with pytest.raises(KeyError):
            registry.get_or_raise("unregistered_domain_key")
