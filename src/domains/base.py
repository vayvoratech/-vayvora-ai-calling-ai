"""Base domain configuration and slot schemas.

Defines the structure for business domain definitions without hardcoding
factual knowledge or embedding runtime conversation flows.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from src.core.types import DomainType


class SlotDefinition(BaseModel):
    """Specification of an entity/slot extracted during conversation."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Unique identifier for the slot")
    slot_type: str = Field(default="string", description="Data type (string, integer, email, boolean)")
    description: str = Field(..., min_length=1, description="Human/LLM guidance on what this slot captures")
    required: bool = Field(default=False, description="Whether slot is mandatory for completing primary actions")
    example: Optional[str] = Field(default=None, description="Illustrative example value")


class DomainConfig(BaseModel):
    """Metadata and intent/slot registry for a business domain."""

    model_config = ConfigDict(extra="forbid")

    domain: DomainType = Field(..., description="Unique domain enumeration key")
    name: str = Field(..., min_length=1, description="Official domain title")
    description: str = Field(..., min_length=1, description="Overview of services and purpose")
    supported_intents: List[str] = Field(..., min_length=1, description="List of recognized intent names")
    supported_slots: List[SlotDefinition] = Field(default_factory=list, description="List of entity slots")
    persona_guidelines: str = Field(
        default="",
        description="High-level behavioral tone and persona guidance for LLM prompt synthesizer",
    )

    def is_intent_supported(self, intent: str) -> bool:
        """Check if an intent is recognized within this domain."""
        return intent.strip().lower() in [i.lower() for i in self.supported_intents]

    def get_slot_names(self) -> List[str]:
        """Return list of all registered slot names."""
        return [slot.name for slot in self.supported_slots]

    def get_slot_definition(self, slot_name: str) -> Optional[SlotDefinition]:
        """Retrieve the slot definition for a given slot name if registered."""
        for slot in self.supported_slots:
            if slot.name.lower() == slot_name.strip().lower():
                return slot
        return None
