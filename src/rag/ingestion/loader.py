from __future__ import annotations

from pathlib import Path

from src.rag.config import rag_config


class MarkdownDocumentLoader:
    """
    Loads Markdown documents from the knowledge base.

    Document IDs are based on the complete relative path,
    preventing collisions between files with the same filename.
    """

    def __init__(
        self,
        base_path: str | None = None,
    ):
        self.base_path = Path(
            base_path
            or rag_config.knowledge_base_path
        )

    def load_documents(self) -> list[dict]:

        if not self.base_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found: "
                f"{self.base_path}"
            )

        documents: list[dict] = []

        for file_path in sorted(
            self.base_path.rglob("*.md")
        ):

            relative_path = (
                file_path.relative_to(
                    self.base_path
                )
            )

            source = str(
                relative_path
            ).replace("\\", "/")

            # Use the complete relative path
            # without the .md extension.
            #
            # Example:
            # careers/overview.md
            # -> careers/overview
            #
            # legal/overview.md
            # -> legal/overview
            #
            # This prevents document ID collisions.
            document_id = str(
                relative_path.with_suffix("")
            ).replace("\\", "/")

            category = (
                relative_path.parts[0]
                if len(relative_path.parts) > 1
                else "general"
            )

            text = file_path.read_text(
                encoding="utf-8"
            ).strip()

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