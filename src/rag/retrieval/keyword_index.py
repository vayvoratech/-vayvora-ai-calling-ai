from __future__ import annotations

from functools import lru_cache

from src.rag.ingestion.pipeline import IngestionPipeline
from src.rag.retrieval.hybrid_search import (
    BM25Retriever,
    KeywordDocument,
)


class KnowledgeBaseKeywordIndex:
    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id

        chunks = IngestionPipeline().run(
            tenant_id=tenant_id
        )

        documents: list[KeywordDocument] = []

        for chunk in chunks:
            # Give the BM25 index useful structural information.
            #
            # This helps exact queries such as:
            # "What is Vayvora?"
            # "Tell me about Vayvora"
            # "What company is Vayvora?"
            #
            # The original chunk text remains unchanged.
            searchable_text = " ".join(
                [
                    chunk.document_id.replace("_", " "),
                    chunk.category.replace("_", " "),
                    chunk.section or "",
                    chunk.text,
                ]
            )

            documents.append(
                KeywordDocument(
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
            )

        self.retriever = BM25Retriever(
            documents
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ):
        results = self.retriever.search(
            query=query,
            top_k=top_k,
        )

        # Restore the original chunk text.
        # The expanded searchable text above is only for BM25.
        for result in results:
            original_text = result.metadata.get(
                "original_text"
            )

            if original_text:
                result.text = original_text

        return results


@lru_cache(maxsize=16)
def get_keyword_index(
    tenant_id: str = "default",
) -> KnowledgeBaseKeywordIndex:
    return KnowledgeBaseKeywordIndex(
        tenant_id=tenant_id
    )