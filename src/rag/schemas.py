"""Schemas for JB Infra RAG queries, retrieval results, and responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.types import DomainType, RAGChunk, RAGQuery
from src.rag.models import DocumentChunk


@dataclass
class RetrievalResult:
    """Represents a scored chunk returned from vector or keyword retrieval."""

    chunk_id: str
    text: str
    score: float
    source: str = ""
    category: str = ""
    section: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_rag_chunk(
        self,
        domain: Optional[DomainType] = None,
    ) -> RAGChunk:
        """Convert the retrieval result into a shared RAG chunk."""

        # JB Infra MVP uses one general RAG domain.
        target_domain = domain or DomainType.GENERAL

        title = (
            self.section
            or self.metadata.get("title")
            or self.source
            or "Reference Document"
        )

        return RAGChunk(
            doc_id=self.chunk_id,
            domain=target_domain,
            title=title,
            content=self.text,
            score=self.score,
            metadata={
                **self.metadata,
                "source_file": self.source,
                "category": self.category,
                "section": self.section,
                "tenant_id": "default",
            },
        )


@dataclass
class RAGResponse:
    """Unified response containing retrieved results and aggregated context."""

    query: str
    results: List[RetrievalResult] = field(default_factory=list)
    context: str = ""
    has_relevant_context: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "DocumentChunk",
    "RetrievalResult",
    "RAGResponse",
    "RAGQuery",
    "RAGChunk",
    "DomainType",
]