"""Manifest tracking for incremental knowledge base ingestion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from src.rag.config import rag_config


@dataclass
class DocumentManifest:
    """Tracks state and content hash of an ingested document."""

    document_id: str
    source: str
    tenant_id: str
    content_hash: str
    version: str = "1"
    chunk_count: int = 0
    updated_at: str = ""

    def __post_init__(self):
        if not self.updated_at:
            self.updated_at = datetime.now(timezone.utc).isoformat()


class ManifestStore:
    """Persists ingestion manifest to a JSON file to prevent redundant re-indexing."""

    def __init__(self, file_path: Optional[str] = None):
        self.file_path = Path(file_path or rag_config.manifest_path)

    def load(self) -> Dict[str, DocumentManifest]:
        if not self.file_path.is_file():
            return {}

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                doc_id: DocumentManifest(**meta)
                for doc_id, meta in data.items()
            }
        except Exception:
            return {}

    def save(self, manifests: Dict[str, DocumentManifest]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            doc_id: asdict(manifest)
            for doc_id, manifest in manifests.items()
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    def get(self, document_id: str) -> Optional[DocumentManifest]:
        return self.load().get(document_id)

    def update(self, manifest: DocumentManifest) -> None:
        manifests = self.load()
        manifests[manifest.document_id] = manifest
        self.save(manifests)

    def delete(self, document_id: str) -> None:
        manifests = self.load()
        if document_id in manifests:
            del manifests[document_id]
            self.save(manifests)
