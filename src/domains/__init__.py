"""Domain configurations and registry package."""

from src.domains.base import DomainConfig, SlotDefinition
from src.domains.infra_builder.config import INFRA_BUILDER_DOMAIN_CONFIG
from src.domains.registry import (
    GENERAL_DOMAIN_CONFIG,
    UNKNOWN_DOMAIN_CONFIG,
    DomainRegistry,
    get_domain_registry,
)

__all__ = [
    "DomainConfig",
    "SlotDefinition",
    "INFRA_BUILDER_DOMAIN_CONFIG",
    "GENERAL_DOMAIN_CONFIG",
    "UNKNOWN_DOMAIN_CONFIG",
    "DomainRegistry",
    "get_domain_registry",
]
