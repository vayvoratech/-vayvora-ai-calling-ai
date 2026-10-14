
"""Registry mapping domain types to their metadata configurations."""

from typing import Dict, List, Optional

from src.core.types import DomainType
from src.domains.base import DomainConfig, SlotDefinition
from src.domains.infra_builder.config import INFRA_BUILDER_DOMAIN_CONFIG


GENERAL_DOMAIN_CONFIG = DomainConfig(
    domain=DomainType.GENERAL,
    name="General Support",
    description=(
        "General conversational support for callers contacting Infra Builder."
    ),
    supported_intents=[
        "greeting",
        "identity_inquiry",
        "capabilities",
        "general_inquiry",
        "general_question",
        "fallback",
        "help",
        "clarification",
        "end_call",
        "termination",
    ],
    supported_slots=[
        SlotDefinition(
            name="name",
            slot_type="string",
            description="Caller name when explicitly provided or reliably known.",
        ),
        SlotDefinition(
            name="phone",
            slot_type="string",
            description="Caller phone number when provided or available from a trusted source.",
        ),
    ],
    persona_guidelines=(
        "Act as a professional, concise, and helpful assistant for Infra Builder. "
        "Answer general questions when supported by available information. "
        "For property-specific facts, use the active Infra Builder knowledge source. "
        "Never invent company information or claim an action succeeded without "
        "confirmation from the relevant tool."
    ),
)


UNKNOWN_DOMAIN_CONFIG = DomainConfig(
    domain=DomainType.UNKNOWN,
    name="Initial Call Context",
    description=(
        "Temporary domain state used before the caller's intent or session context "
        "has been resolved. Infra Builder is the only configured business domain."
    ),
    supported_intents=[
        "greeting",
        "identity_inquiry",
        "purpose_discovery",
        "general_inquiry",
        "general_question",
        "clarification",
        "fallback",
        "help",
        "end_call",
        "termination",
    ],
    supported_slots=[
        SlotDefinition(
            name="name",
            slot_type="string",
            description="Caller name when explicitly provided or reliably known.",
        ),
        SlotDefinition(
            name="phone",
            slot_type="string",
            description="Caller phone number when provided or available from a trusted source.",
        ),
    ],
    persona_guidelines=(
        "You are a concise and welcoming assistant for Infra Builder. "
        "Understand the caller's actual request and respond to it directly. "
        "Ask a brief clarification only when necessary. Do not route the caller "
        "to unrelated businesses or assume details that have not been established."
    ),
)


class DomainRegistry:
    """Central registry of registered business domain configurations."""

    def __init__(self) -> None:
        self._domains: Dict[DomainType, DomainConfig] = {}

        # Infra Builder is the only configured business domain.
        self.register(INFRA_BUILDER_DOMAIN_CONFIG)

        # Retain shared support states for compatibility with the existing engine.
        self.register(GENERAL_DOMAIN_CONFIG)
        self.register(UNKNOWN_DOMAIN_CONFIG)

    def register(self, config: DomainConfig) -> None:
        """Register or override a domain configuration."""
        self._domains[config.domain] = config

    def get(self, domain_type: DomainType) -> Optional[DomainConfig]:
        """Fetch a domain configuration, returning None if it is not registered."""
        return self._domains.get(domain_type)

    def get_or_raise(self, domain_type: DomainType) -> DomainConfig:
        """Fetch a domain configuration, raising KeyError if it is not registered."""
        config = self.get(domain_type)

        if config is None:
            raise KeyError(
                f"Domain '{domain_type}' is not registered."
            )

        return config

    def list_domains(self) -> List[DomainType]:
        """Return all registered domain types."""
        return list(self._domains.keys())


# Global default registry instance.
_default_registry = DomainRegistry()


def get_domain_registry() -> DomainRegistry:
    """Retrieve the global domain registry instance."""
    return _default_registry
