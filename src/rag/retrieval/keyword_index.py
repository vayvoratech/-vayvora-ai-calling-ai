"""Keyword index manager bridging chunk ingestion with BM25 lexical search."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from src.rag.ingestion.pipeline import IngestionPipeline
from src.rag.models import DocumentChunk
from src.rag.retrieval.hybrid_search import BM25Retriever, KeywordDocument
from src.rag.schemas import RetrievalResult


class KnowledgeBaseKeywordIndex:
    """Manages lexical BM25 index for domain/tenant knowledge base chunks."""

    def __init__(
        self,
        tenant_id: str = "default",
        initial_chunks: Optional[List[DocumentChunk]] = None,
    ):
        self.tenant_id = tenant_id
        self.documents: List[KeywordDocument] = []
        self._doc_map: dict[str, KeywordDocument] = {}

        chunks = initial_chunks
        if chunks is None:
            try:
                chunks = IngestionPipeline().run(tenant_id=tenant_id)
            except Exception:
                chunks = []

        if chunks:
            self.add_chunks(chunks)
        else:
            self.retriever = BM25Retriever([])

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add or update chunks in the keyword index."""
        for chunk in chunks:
            searchable_text = " ".join(
                filter(
                    None,
                    [
                        chunk.document_id.replace("_", " "),
                        chunk.category.replace("_", " "),
                        chunk.section or "",
                        chunk.text,
                    ],
                )
            )

            doc = KeywordDocument(
                chunk_id=chunk.chunk_id,
                text=searchable_text,
                source=chunk.source,
                category=chunk.category,
                section=chunk.section,
                metadata={
                    "document_id": chunk.document_id,
                    "tenant_id": chunk.tenant_id,
                    "version": chunk.version,
                    "original_text": chunk.text,
                    **chunk.metadata,
                },
            )
            self._doc_map[chunk.chunk_id] = doc

        self.documents = list(self._doc_map.values())
        self.retriever = BM25Retriever(self.documents)

    def remove_chunk(self, chunk_id: str) -> None:
        if chunk_id in self._doc_map:
            del self._doc_map[chunk_id]
            self.documents = list(self._doc_map.values())
            self.retriever = BM25Retriever(self.documents)

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Search lexical BM25 index and restore original clean chunk text."""
        results = self.retriever.search(query=query, top_k=top_k)

        for result in results:
            orig = result.metadata.get("original_text")
            if orig:
                result.text = orig

        return results


@lru_cache(maxsize=16)
def get_keyword_index(tenant_id: str = "default") -> KnowledgeBaseKeywordIndex:
    return KnowledgeBaseKeywordIndex(tenant_id=tenant_id)
