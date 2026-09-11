from __future__ import annotations

import hashlib

from src.rag.ingestion.chunker import MarkdownChunker
from src.rag.ingestion.loader import MarkdownDocumentLoader
from src.rag.ingestion.manifest import (
    DocumentManifest,
    ManifestStore,
)
from src.rag.schemas import DocumentChunk


class IncrementalIngestionService:
    """
    Detects new/changed/unchanged documents.

    This layer only determines what needs to be indexed.
    Redis insertion/deletion is handled by the vector store.
    """

    def __init__(
        self,
        tenant_id: str = "default",
        loader: MarkdownDocumentLoader | None = None,
        chunker: MarkdownChunker | None = None,
        manifest: ManifestStore | None = None,
    ):
        if not tenant_id:
            raise ValueError(
                "tenant_id is required"
            )

        self.tenant_id = tenant_id
        self.loader = (
            loader or MarkdownDocumentLoader()
        )
        self.chunker = (
            chunker or MarkdownChunker()
        )
        self.manifest = (
            manifest or ManifestStore()
        )

    @staticmethod
    def _content_hash(
        text: str,
    ) -> str:
        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    def prepare(
        self,
    ) -> dict:

        documents = self.loader.load_documents()
        existing = self.manifest.load()

        changed_documents = []
        unchanged_documents = []
        current_document_ids = set()

        for document in documents:

            document_id = document[
                "document_id"
            ]

            current_document_ids.add(
                document_id
            )

            content_hash = (
                self._content_hash(
                    document["text"]
                )
            )

            previous = existing.get(
                document_id
            )

            if (
                previous is not None
                and previous.content_hash
                == content_hash
            ):
                unchanged_documents.append(
                    document
                )
                continue

            chunks = (
                self.chunker.chunk_document(
                    document=document,
                    tenant_id=self.tenant_id,
                )
            )

            for chunk in chunks:
                chunk.metadata.update(
                    {
                        "content_hash": content_hash,
                        "source_path": chunk.source,
                        "ingestion_version": "1",
                    }
                )

            changed_documents.append(
                {
                    "document": document,
                    "content_hash": content_hash,
                    "chunks": chunks,
                }
            )

        deleted_document_ids = sorted(
            set(existing.keys())
            - current_document_ids
        )

        return {
            "tenant_id": self.tenant_id,
            "changed_documents": changed_documents,
            "unchanged_documents": unchanged_documents,
            "deleted_document_ids": deleted_document_ids,
        }

    def update_manifest(
        self,
        document: dict,
        content_hash: str,
        chunk_count: int,
        version: str = "1",
    ) -> DocumentManifest:

        manifest = DocumentManifest(
            document_id=document[
                "document_id"
            ],
            source=document[
                "source"
            ],
            tenant_id=self.tenant_id,
            content_hash=content_hash,
            version=version,
            chunk_count=chunk_count,
        )

        self.manifest.update(
            manifest
        )

        return manifest