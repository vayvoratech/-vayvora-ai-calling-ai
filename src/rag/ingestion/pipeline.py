from __future__ import annotations

import hashlib
from typing import Any

from src.rag.ingestion.chunker import MarkdownChunker
from src.rag.ingestion.loader import MarkdownDocumentLoader
from src.rag.schemas import DocumentChunk


class IngestionPipeline:
    def __init__(
        self,
        loader: MarkdownDocumentLoader | None = None,
        chunker: MarkdownChunker | None = None,
    ):
        self.loader = loader or MarkdownDocumentLoader()
        self.chunker = chunker or MarkdownChunker()

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def run(self, tenant_id: str = "default") -> list[DocumentChunk]:
        documents = self.loader.load_documents()
        all_chunks: list[DocumentChunk] = []

        for document in documents:
            content_hash = self._content_hash(document["text"])
            chunks = self.chunker.chunk_document(document=document, tenant_id=tenant_id)
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
