from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """
    A chunk of source knowledge prepared for indexing.
    """

    chunk_id: str
    document_id: str
    tenant_id: str
    text: str

    source: str
    category: str

    section: str | None = None
    version: str = "1"

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """
    A document chunk returned by the retrieval layer.
    """

    chunk_id: str
    text: str

    score: float

    source: str
    category: str

    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """
    Final response from the RAG service.
    """

    query: str

    results: list[RetrievalResult] = field(
        default_factory=list
    )

    context: str = ""

    has_relevant_context: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )