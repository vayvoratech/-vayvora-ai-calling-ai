from __future__ import annotations

import hashlib
from typing import Any

from src.rag.ingestion.chunker import MarkdownChunker
from src.rag.ingestion.loader import MarkdownDocumentLoader
from src.rag.schemas import DocumentChunk


class IngestionPipeline:
    """
    Knowledge-base ingestion pipeline.

    Responsibilities:
    - Load Markdown documents
    - Chunk documents
    - Generate deterministic content hashes
    - Attach ingestion metadata
    - Preserve tenant isolation
    """

    def __init__(
        self,
        loader: MarkdownDocumentLoader | None = None,
        chunker: MarkdownChunker | None = None,
    ):
        self.loader = loader or MarkdownDocumentLoader()
        self.chunker = chunker or MarkdownChunker()

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    def run(
        self,
        tenant_id: str = "default",
    ) -> list[DocumentChunk]:

        if not tenant_id:
            raise ValueError(
                "tenant_id is required"
            )

        documents = self.loader.load_documents()

        all_chunks: list[DocumentChunk] = []

        for document in documents:

            content_hash = self._content_hash(
                document["text"]
            )

            chunks = self.chunker.chunk_document(
                document=document,
                tenant_id=tenant_id,
            )

            for chunk in chunks:

                chunk.metadata.update(
                    {
                        "content_hash": content_hash,
                        "source_path": chunk.source,
                        "ingestion_version": "1",
                    }
                )

            all_chunks.extend(chunks)

        return all_chunks

    def run_with_stats(
        self,
        tenant_id: str = "default",
    ) -> dict[str, Any]:

        documents = self.loader.load_documents()

        chunks = self.run(
            tenant_id=tenant_id
        )

        return {
            "tenant_id": tenant_id,
            "documents": len(documents),
            "chunks": len(chunks),
            "sources": sorted(
                {
                    chunk.source
                    for chunk in chunks
                }
            ),
        }