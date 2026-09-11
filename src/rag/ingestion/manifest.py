from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class DocumentManifest:
    """
    Tracks the current indexed state of a knowledge-base document.
    """

    document_id: str
    source: str
    tenant_id: str
    content_hash: str
    version: str
    chunk_count: int


class ManifestStore:
    """
    Persistent manifest for RAG ingestion.

    The manifest records document hashes so ingestion can determine
    whether a document has changed since the previous indexing run.
    """

    def __init__(
        self,
        path: str = "data/rag_manifest.json",
    ):
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load(self) -> dict[str, DocumentManifest]:
        if not self.path.exists():
            return {}

        try:
            raw = json.loads(
                self.path.read_text(
                    encoding="utf-8"
                )
            )
        except (json.JSONDecodeError, OSError):
            return {}

        manifests: dict[str, DocumentManifest] = {}

        for document_id, data in raw.items():
            manifests[document_id] = (
                DocumentManifest(
                    document_id=data["document_id"],
                    source=data["source"],
                    tenant_id=data["tenant_id"],
                    content_hash=data["content_hash"],
                    version=data["version"],
                    chunk_count=int(
                        data["chunk_count"]
                    ),
                )
            )

        return manifests

    def save(
        self,
        manifests: dict[str, DocumentManifest],
    ) -> None:

        payload: dict[str, Any] = {
            document_id: asdict(manifest)
            for document_id, manifest in manifests.items()
        }

        self.path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def get(
        self,
        document_id: str,
    ) -> DocumentManifest | None:

        manifests = self.load()

        return manifests.get(
            document_id
        )

    def update(
        self,
        manifest: DocumentManifest,
    ) -> None:

        manifests = self.load()

        manifests[
            manifest.document_id
        ] = manifest

        self.save(
            manifests
        )

    def delete(
        self,
        document_id: str,
    ) -> None:

        manifests = self.load()

        if document_id in manifests:
            del manifests[
                document_id
            ]

        self.save(
            manifests
        )

    def has_changed(
        self,
        document_id: str,
        content_hash: str,
    ) -> bool:

        existing = self.get(
            document_id
        )

        if existing is None:
            return True

        return (
            existing.content_hash
            != content_hash
        )