"""Domain registry mapping domain types to their metadata configurations."""

from typing import Dict, List, Optional
from src.core.types import DomainType
from src.domains.base import DomainConfig, SlotDefinition
from src.domains.edusaas.config import EDUSAAS_DOMAIN_CONFIG
from src.domains.vayvora.config import VAYVORA_DOMAIN_CONFIG

GENERAL_DOMAIN_CONFIG = DomainConfig(
    domain=DomainType.GENERAL,
    name="General & Organizational Support",
    description="Cross-cutting initial greeting, organizational routing, and general inquiries.",
    supported_intents=[
        "greeting",
        "identity_inquiry",
        "capabilities",
        "fallback",
        "help",
        "end_call",
    ],
    supported_slots=[
        SlotDefinition(
            name="target_organization",
            slot_type="string",
            description="Which organization caller is trying to reach (EduSaaS or Vayvora)",
        )
    ],
    persona_guidelines="You are a helpful and polite conversational receptionist assisting callers in reaching either EduSaaS or Vayvora.",
)


class DomainRegistry:
    """Central registry of registered business domain configurations."""

    def __init__(self) -> None:
        self._domains: Dict[DomainType, DomainConfig] = {}
        # Register standard default domains
        self.register(EDUSAAS_DOMAIN_CONFIG)
        self.register(VAYVORA_DOMAIN_CONFIG)
        self.register(GENERAL_DOMAIN_CONFIG)

    def register(self, config: DomainConfig) -> None:
        """Register or override a domain configuration."""
        self._domains[config.domain] = config

    def get(self, domain_type: DomainType) -> Optional[DomainConfig]:
        """Fetch domain configuration for a given domain key."""
        return self._domains.get(domain_type)

    def get_or_raise(self, domain_type: DomainType) -> DomainConfig:
        """Fetch domain configuration, raising KeyError if not found."""
        config = self.get(domain_type)
        if not config:
            raise KeyError(f"Domain '{domain_type}' is not registered.")
        return config

    def list_domains(self) -> List[DomainType]:
        """Return list of all registered domain types."""
        return list(self._domains.keys())


# Global default instance
_default_registry = DomainRegistry()


def get_domain_registry() -> DomainRegistry:
    """Retrieve the global domain registry instance."""
    return _default_registry
