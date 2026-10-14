"""Keyword retrieval index for the JB Infra RAG demo MVP."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

from src.core.types import DomainType
from src.rag.config import rag_config
from src.rag.ingestion.chunker import chunk_markdown
from src.rag.models import DocumentChunk
from src.rag.retrieval.hybrid_search import BM25Retriever, KeywordDocument
from src.rag.schemas import RetrievalResult


class KnowledgeBaseKeywordIndex:
    """BM25 keyword index built from the JB Infra knowledge base."""

    def __init__(
        self,
        tenant_id: str = "default",
        initial_chunks: Optional[Iterable[DocumentChunk]] = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.documents: List[KeywordDocument] = []
        self._doc_map: dict[str, KeywordDocument] = {}
        self.retriever = BM25Retriever([])

        if initial_chunks is not None:
            self.add_chunks(initial_chunks)

    def _to_keyword_document(
        self,
        chunk: DocumentChunk,
    ) -> KeywordDocument:
        """Convert a RAG chunk into a searchable BM25 document."""

        searchable_text = " ".join(
            part
            for part in [
                chunk.document_id,
                chunk.source,
                chunk.category,
                chunk.document_type,
                chunk.section or "",
                chunk.title,
                chunk.text,
            ]
            if part
        )

        return KeywordDocument(
            chunk_id=chunk.chunk_id,
            text=searchable_text,
            source=chunk.source,
            category=chunk.category,
            section=chunk.section,
            metadata={
                **chunk.metadata,
                "tenant_id": chunk.tenant_id,
                "domain": DomainType.GENERAL.value,
                "document_id": chunk.document_id,
                "document_type": chunk.document_type,
                "title": chunk.title,
                "original_text": chunk.text,
            },
        )

    def add_chunks(
        self,
        chunks: Iterable[DocumentChunk],
    ) -> None:
        """Add or replace chunks in the keyword index."""

        for chunk in chunks:
            if chunk.tenant_id != self.tenant_id:
                continue

            document = self._to_keyword_document(chunk)
            self._doc_map[chunk.chunk_id] = document

        self.documents = list(self._doc_map.values())
        self.retriever = BM25Retriever(self.documents)

    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove one chunk from the index."""

        if chunk_id not in self._doc_map:
            return False

        del self._doc_map[chunk_id]

        self.documents = list(self._doc_map.values())
        self.retriever = BM25Retriever(self.documents)

        return True

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Search the keyword index using BM25."""

        if not query.strip() or not self.documents:
            return []

        results = self.retriever.search(
            query=query,
            top_k=top_k,
        )

        for result in results:
            document = self._doc_map.get(result.chunk_id)

            if document is None:
                continue

            result.text = document.metadata.get(
                "original_text",
                result.text,
            )

            result.metadata = {
                **result.metadata,
                "tenant_id": self.tenant_id,
                "domain": DomainType.GENERAL.value,
            }

        return results

    def clear(self) -> None:
        """Remove all indexed documents."""

        self.documents.clear()
        self._doc_map.clear()
        self.retriever = BM25Retriever([])

    def rebuild(self) -> int:
        """Load and chunk the JB Infra knowledge base."""

        base_dir = Path(rag_config.knowledge_base_dir)

        if not base_dir.exists():
            return 0

        chunks: List[DocumentChunk] = []

        for path in sorted(base_dir.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() not in {".md", ".txt"}:
                continue

            try:
                text = path.read_text(
                    encoding="utf-8",
                ).strip()
            except Exception:
                continue

            if not text:
                continue

            relative_path = path.relative_to(
                base_dir,
            ).as_posix()

            parts = relative_path.split("/")

            category = (
                parts[0]
                if len(parts) > 1
                else "general"
            )

            document_type = (
                "project"
                if category == "projects"
                else category
            )

            document_id = path.stem.lower().replace(
                " ",
                "_",
            )

            title = self._extract_title(
                text,
                path.stem,
            )

            text_chunks = chunk_markdown(
                text,
                chunk_size=rag_config.chunk_size,
                chunk_overlap=rag_config.chunk_overlap,
            )

            total_chunks = len(text_chunks)

            for index, chunk_text in enumerate(
                text_chunks,
                start=1,
            ):
                chunk_id = (
                    f"{document_id}:{index}"
                )

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        tenant_id=self.tenant_id,
                        document_id=document_id,
                        source=relative_path,
                        category=category,
                        document_type=document_type,
                        section=title,
                        version="1",
                        chunk_index=index,
                        total_chunks=total_chunks,
                        metadata={
                            "title": title,
                            "source": relative_path,
                            "source_file": relative_path,
                            "category": category,
                            "document_type": document_type,
                            "tenant_id": self.tenant_id,
                        },
                    )
                )

        self.clear()
        self.add_chunks(chunks)

        return len(self.documents)

    @staticmethod
    def _extract_title(
        text: str,
        fallback: str,
    ) -> str:
        """Extract the first Markdown heading."""

        for line in text.splitlines():
            stripped = line.strip()

            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()

                if title:
                    return title

        return fallback.replace("_", " ").title()


@lru_cache(maxsize=8)
def get_keyword_index(
    tenant_id: str = "default",
) -> KnowledgeBaseKeywordIndex:
    """Return a cached keyword index."""

    index = KnowledgeBaseKeywordIndex(
        tenant_id=tenant_id,
    )

    try:
        index.rebuild()
    except Exception:
        # Keep startup resilient for the demo.
        pass

    return index


__all__ = [
    "KnowledgeBaseKeywordIndex",
    "get_keyword_index",
]