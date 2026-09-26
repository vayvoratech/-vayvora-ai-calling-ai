"""Domain configurations and registry package."""

from src.domains.base import DomainConfig, SlotDefinition
from src.domains.edusaas import (
    EDUSAAS_DOMAIN_CONFIG,
    EDUSAAS_INTENTS,
    EDUSAAS_SLOTS,
)
from src.domains.registry import (
    GENERAL_DOMAIN_CONFIG,
    DomainRegistry,
    get_domain_registry,
)
from src.domains.vayvora import (
    VAYVORA_DOMAIN_CONFIG,
    VAYVORA_INTENTS,
    VAYVORA_SLOTS,
)

__all__ = [
    "DomainConfig",
    "SlotDefinition",
    "EDUSAAS_DOMAIN_CONFIG",
    "EDUSAAS_INTENTS",
    "EDUSAAS_SLOTS",
    "VAYVORA_DOMAIN_CONFIG",
    "VAYVORA_INTENTS",
    "VAYVORA_SLOTS",
    "GENERAL_DOMAIN_CONFIG",
    "DomainRegistry",
    "get_domain_registry",
]
