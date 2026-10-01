"""Incremental ingestion service detecting changes and updating manifest records."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from src.rag.ingestion.chunker import MarkdownChunker
from src.rag.ingestion.loader import MarkdownDocumentLoader
from src.rag.ingestion.manifest import DocumentManifest, ManifestStore


class IncrementalIngestionService:
    """Detects modified and deleted documents to minimize redundant embedding and indexing."""

    def __init__(
        self,
        tenant_id: str = "default",
        loader: Optional[MarkdownDocumentLoader] = None,
        chunker: Optional[MarkdownChunker] = None,
        manifest: Optional[ManifestStore] = None,
    ):
        self.tenant_id = tenant_id
        self.loader = loader or MarkdownDocumentLoader()
        self.chunker = chunker or MarkdownChunker()
        self.manifest = manifest or ManifestStore()

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def prepare(self) -> Dict[str, Any]:
        """Compare documents on disk with manifest to determine ingestion plan."""
        documents = self.loader.load_documents(tenant_id=self.tenant_id)
        existing = self.manifest.load()

        changed_documents = []
        unchanged_documents = []
        current_document_ids = set()

        for document in documents:
            doc_id = document["document_id"]
            current_document_ids.add(doc_id)
            content_hash = self._content_hash(document["text"])
            previous = existing.get(doc_id)

            if previous is not None and previous.content_hash == content_hash:
                unchanged_documents.append(document)
                continue

            chunks = self.chunker.chunk_document(document=document, tenant_id=self.tenant_id)
            for chunk in chunks:
                chunk.metadata.update({
                    "content_hash": content_hash,
                    "source_path": chunk.source,
                    "ingestion_version": "1",
                })

            changed_documents.append({
                "document": document,
                "content_hash": content_hash,
                "chunks": chunks,
            })

        # Find deleted documents that exist in manifest but no longer on disk
        deleted_document_ids = sorted(
            doc_id
            for doc_id, item in existing.items()
            if (self.tenant_id in ("default", "all") or item.tenant_id == self.tenant_id)
            and doc_id not in current_document_ids
        )

        return {
            "tenant_id": self.tenant_id,
            "changed_documents": changed_documents,
            "unchanged_documents": unchanged_documents,
            "deleted_document_ids": deleted_document_ids,
        }

    def update_manifest(
        self,
        document: Dict[str, Any],
        content_hash: str,
        chunk_count: int,
        version: str = "1",
    ) -> DocumentManifest:
        manifest = DocumentManifest(
            document_id=document["document_id"],
            source=document["source"],
            tenant_id=self.tenant_id,
            content_hash=content_hash,
            version=version,
            chunk_count=chunk_count,
        )
        self.manifest.update(manifest)
        return manifest
