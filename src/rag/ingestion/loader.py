from __future__ import annotations

from pathlib import Path
from src.rag.config import rag_config


class MarkdownDocumentLoader:
    """Loads Markdown documents from the knowledge base directory."""

    def __init__(self, base_path: str | None = None):
        self.base_path = Path(base_path or rag_config.knowledge_base_path)

    def load_documents(self) -> list[dict]:
        if not self.base_path.exists():
            return []

        documents: list[dict] = []
        for file_path in sorted(self.base_path.rglob("*.md")):
            relative_path = file_path.relative_to(self.base_path)
            source = str(relative_path).replace("\\", "/")
            document_id = str(relative_path.with_suffix("")).replace("\\", "/")
            category = relative_path.parts[0] if len(relative_path.parts) > 1 else "general"

            text = file_path.read_text(encoding="utf-8").strip()
            if not text:
                continue

            documents.append(
                {
                    "document_id": document_id,
                    "source": source,
                    "category": category,
                    "text": text,
                }
            )

        return documents
