"""Markdown document loader scanning knowledge base files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from src.rag.config import rag_config


class MarkdownDocumentLoader:
    """Recursively loads Markdown and text documents from knowledge base directory."""

    def __init__(self, base_dir: Optional[str | Path] = None):
        self.base_dir = Path(base_dir or rag_config.knowledge_base_dir)

    def load_documents(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load documents, optionally filtered by tenant/domain."""
        if not self.base_dir.exists():
            return []

        search_dir = self.base_dir
        if tenant_id and tenant_id not in ("default", "all"):
            tenant_path = self.base_dir / tenant_id
            if tenant_path.is_dir():
                search_dir = tenant_path

        documents: List[Dict[str, Any]] = []

        for path in search_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in (".md", ".txt"):
                continue

            try:
                text = path.read_text(encoding="utf-8").strip()
            except Exception:
                continue

            if not text:
                continue

            rel_path = path.relative_to(self.base_dir).as_posix()
            parts = rel_path.split("/")

            # Determine tenant and category
            if len(parts) >= 2 and parts[0] in ("edusaas", "vayvora"):
                doc_tenant = parts[0]
                category = parts[1] if len(parts) > 2 else "general"
            else:
                doc_tenant = tenant_id or "default"
                category = path.parent.name if path.parent != self.base_dir else "general"

            if tenant_id and tenant_id not in ("default", "all") and doc_tenant != tenant_id:
                continue

            doc_id = path.stem.lower().replace(" ", "_")

            documents.append(
                {
                    "document_id": doc_id,
                    "source": rel_path,
                    "category": category,
                    "tenant_id": doc_tenant,
                    "text": text,
                    "file_path": str(path),
                }
            )

        return documents
