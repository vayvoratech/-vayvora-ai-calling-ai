"""JB Infra knowledge-base ingestion entry points."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.logging import get_logger
from src.rag.ingestion.chunker import chunk_markdown
from src.rag.models import DocumentChunk

logger = get_logger("rag.ingestion")


def discover_knowledge_base(
    knowledge_base_dir: Optional[str] = None,
) -> List[Path]:
    """Discover Markdown documents in the JB Infra knowledge base."""

    if knowledge_base_dir:
        root = Path(knowledge_base_dir)
    else:
        root = (
            Path(__file__)
            .resolve()
            .parent.parent.parent
            / "knowledge_base"
        )

    if not root.exists():
        logger.warning(
            "Knowledge base directory does not exist: %s",
            root,
        )
        return []

    files = sorted(
        path
        for path in root.rglob("*.md")
        if path.is_file()
        and not path.name.startswith(".")
    )

    logger.info(
        "Discovered %s knowledge-base documents.",
        len(files),
    )

    return files


def _document_type(path: Path) -> str:
    """Infer a simple document type from the KB directory."""

    parts = {
        part.lower()
        for part in path.parts
    }

    if "projects" in parts:
        return "project"

    if "company" in parts:
        return "company"

    if path.name.lower() == "faq.md":
        return "faq"

    if path.name.lower() == "agent_guidelines.md":
        return "guidelines"

    return "general"


def _category(path: Path) -> str:
    """Infer the KB category from its parent directory."""

    parent = path.parent.name.lower()

    if parent in {
        "company",
        "projects",
    }:
        return parent

    return "general"


def _title_from_markdown(
    text: str,
    fallback: str,
) -> str:
    """Extract the first Markdown H1 title."""

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("# "):
            return stripped[2:].strip()

    return fallback


def load_document(
    path: Path,
) -> str:
    """Read one Markdown knowledge-base document."""

    return path.read_text(
        encoding="utf-8",
    )


def load_documents(
    knowledge_base_dir: Optional[str] = None,
) -> List[DocumentChunk]:
    """Load and chunk all JB Infra Markdown documents."""

    documents: List[DocumentChunk] = []

    for path in discover_knowledge_base(
        knowledge_base_dir
    ):
        try:
            text = load_document(path)

            if not text.strip():
                logger.warning(
                    "Skipping empty document: %s",
                    path,
                )
                continue

            document_id = path.stem
            title = _title_from_markdown(
                text,
                fallback=path.stem.replace(
                    "_",
                    " ",
                ).title(),
            )

            category = _category(path)
            document_type = _document_type(path)

            chunks = chunk_markdown(text)

            if not chunks:
                chunks = [text]

            total_chunks = len(chunks)

            for index, chunk_text in enumerate(
                chunks,
                start=1,
            ):
                chunk_id = (
                    f"{document_id}:chunk:{index}"
                )

                documents.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        tenant_id="general",
                        document_id=document_id,
                        source=str(path),
                        category=category,
                        document_type=document_type,
                        section=title,
                        version="1",
                        chunk_index=index,
                        total_chunks=total_chunks,
                        metadata={
                            "title": title,
                            "source": str(path),
                            "source_file": str(path),
                            "category": category,
                            "document_type": document_type,
                            "tenant_id": "general",
                        },
                    )
                )

        except Exception as exc:
            logger.error(
                "Failed to load knowledge-base document %s: %s",
                path,
                exc,
            )

    logger.info(
        "Loaded %s document chunks from the JB Infra knowledge base.",
        len(documents),
    )

    return documents


async def ingest_knowledge_base(
    knowledge_base_dir: Optional[str] = None,
    retriever=None,
) -> int:
    """Load and index the JB Infra knowledge base."""

    if retriever is None:
        from src.rag.retriever import GroundedKnowledgeProvider

        retriever = GroundedKnowledgeProvider()

    documents = load_documents(
        knowledge_base_dir
    )

    if not documents:
        logger.warning(
            "No knowledge-base documents found."
        )
        return 0

    indexed = await retriever.index_documents(
        documents
    )

    logger.info(
        "JB Infra ingestion completed: %s/%s chunks indexed.",
        indexed,
        len(documents),
    )

    return indexed


__all__ = [
    "discover_knowledge_base",
    "load_document",
    "load_documents",
    "ingest_knowledge_base",
]