"""Data models for RAG chunks, metadata, and document structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from src.core.types import DomainType, RAGChunk


@dataclass
class DocumentChunk:
    """Unified document chunk representation bridging old and new RAG architectures."""

    chunk_id: str
    text: str
    tenant_id: str = "default"
    document_id: str = ""
    source: str = ""
    category: str = "general"
    section: Optional[str] = None
    version: str = "1"
    chunk_index: int = 1
    total_chunks: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.document_id and self.source:
            self.document_id = self.source
        if "tenant_id" not in self.metadata:
            self.metadata["tenant_id"] = self.tenant_id
        if "category" not in self.metadata:
            self.metadata["category"] = self.category
        if "source" not in self.metadata and self.source:
            self.metadata["source"] = self.source
        if self.section and "section" not in self.metadata:
            self.metadata["section"] = self.section

    @property
    def domain(self) -> DomainType:
        if self.tenant_id.lower() == "edusaas":
            return DomainType.EDUSAAS
        elif self.tenant_id.lower() == "vayvora":
            return DomainType.VAYVORA
        return DomainType.GENERAL

    @property
    def source_file(self) -> str:
        return self.source

    @property
    def title(self) -> str:
        return self.metadata.get("title") or self.section or self.document_id or "Document"

    def to_rag_chunk(self, domain: Optional[DomainType] = None) -> RAGChunk:
        target_domain = domain or self.domain
        return RAGChunk(
            doc_id=self.chunk_id,
            domain=target_domain,
            title=self.title,
            content=self.text,
            score=1.0,
            metadata={
                **self.metadata,
                "source_file": self.source,
                "category": self.category,
                "section": self.section,
                "tenant_id": self.tenant_id,
                "version": self.version,
            },
        )
