"""Grounded hybrid RAG provider for the JB Infra demo MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.core.types import DomainType, RAGChunk, RAGQuery
from src.rag.config import rag_config
from src.rag.embeddings import BaseEmbeddingProvider, get_embedding_provider
from src.rag.ingestion.chunker import chunk_markdown
from src.rag.models import DocumentChunk
from src.rag.retrieval.catalog_intent import is_catalog_query
from src.rag.retrieval.fusion import ReciprocalRankFusion
from src.rag.retrieval.keyword_index import (
    KnowledgeBaseKeywordIndex,
    get_keyword_index,
)
from src.rag.retrieval.reranker import Reranker, get_reranker
from src.rag.schemas import RetrievalResult


class GroundedContextResult(BaseModel):
    """Grounded retrieval response returned to the calling layer."""

    query: RAGQuery
    chunks: List[RAGChunk] = Field(default_factory=list)
    context: str = ""
    has_relevant_context: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GroundedKnowledgeProvider:
    """Hybrid RAG provider for the JB Infra knowledge base."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        keyword_index: Optional[KnowledgeBaseKeywordIndex] = None,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self.settings = settings or get_settings()

        self.embeddings = (
            embedding_provider
            or get_embedding_provider(settings=self.settings)
        )

        self.keyword_index = (
            keyword_index
            or get_keyword_index(tenant_id="default")
        )

        self.reranker = reranker or get_reranker()
        self.rrf = ReciprocalRankFusion()

        self._documents: Dict[str, DocumentChunk] = {}

        # Load the same JB Infra KB used by the keyword retriever.
        self._load_knowledge_base()

    def _load_knowledge_base(self) -> None:
        """Load JB Infra Markdown files into the dense-search store."""

        base_dir = Path(rag_config.knowledge_base_dir)

        if not base_dir.exists():
            return

        for path in sorted(base_dir.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() not in {".md", ".txt"}:
                continue

            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ).strip()

                if not text:
                    continue

                relative_path = path.relative_to(base_dir).as_posix()
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

                document_id = path.stem.lower().replace(" ", "_")

                title = self._extract_title(
                    text=text,
                    fallback=path.stem.replace("_", " ").title(),
                )

                chunks = chunk_markdown(
                    text,
                    chunk_size=rag_config.chunk_size,
                    chunk_overlap=rag_config.chunk_overlap,
                )

                total_chunks = len(chunks)

                for index, chunk_text in enumerate(chunks, start=1):
                    chunk_id = f"{document_id}:{index}"

                    document = DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
    
                    document_id=document_id,
    source=relative_path,
    category=category,
    document_type=document_type,
    
    section="",
    chunk_index=index,
    total_chunks=total_chunks,
    metadata={
        "source": relative_path,
        "category": category,
        "document_type": document_type,
        "title": title,
        "chunk_index": index,
        "total_chunks": total_chunks,
    },
)
                    

                    self._documents[chunk_id] = document

            except Exception:
                continue

    @staticmethod
    def _extract_title(
        text: str,
        fallback: str,
    ) -> str:
        """Extract the first Markdown heading."""

        for line in text.splitlines():
            line = line.strip()

            if line.startswith("# "):
                return line[2:].strip()

        return fallback

    async def search(
        self,
        query: RAGQuery,
    ) -> List[RAGChunk]:
        """Return grounded chunks for a query."""

        result = await self.retrieve_grounded_context(query)
        return result.chunks

    async def index_document(
        self,
        chunk: RAGChunk | DocumentChunk,
    ) -> bool:
        """Add a document to the dense and keyword stores."""

        if isinstance(chunk, DocumentChunk):
            document_chunk = chunk
        else:
            document_chunk = DocumentChunk.from_rag_chunk(chunk)

        self._documents[document_chunk.chunk_id] = document_chunk

        self.keyword_index.add_chunks(
            [document_chunk]
        )

        return True

    async def retrieve_grounded_context(
        self,
        query: RAGQuery,
    ) -> GroundedContextResult:
        """Retrieve and assemble grounded JB Infra context."""

        query_text = (query.query_text or "").strip()

        if not query_text:
            return GroundedContextResult(
                query=query,
                chunks=[],
                context="",
                has_relevant_context=False,
                metadata={
                    "retrieval_mode": "hybrid",
                    "vector_results": 0,
                    "keyword_results": 0,
                    "fused_results": 0,
                    "final_results": 0,
                    "domain": DomainType.GENERAL.value,
                },
            )

        enriched_query = query_text

        if is_catalog_query(
            query_text,
            domain=DomainType.GENERAL,
        ):
            enriched_query = (
                f"{query_text} "
                "JB's Pristine City "
                "JB's Nature Valley "
                "JB's Serene City "
                "JB Pride "
                "Hillside County "
                "Serene Villas "
                "Serene County "
                "Serene Resort"
            )

        vector_results = self._vector_search(
            enriched_query,
            top_k=self._top_k(),
        )

        keyword_results = self.keyword_index.search(
            enriched_query,
            top_k=self._top_k(),
        )

        fused_results = self._fuse_results(
            vector_results=vector_results,
            keyword_results=keyword_results,
            top_k=self._top_k(),
        )

        reranked_results = self._rerank(
            query=enriched_query,
            results=fused_results,
        )

        final_results = self._apply_threshold(
            reranked_results,
        )

        chunks = [
            result.to_rag_chunk(
                domain=DomainType.GENERAL,
            )
            for result in final_results
        ]

        context = self._build_context(chunks)

        return GroundedContextResult(
            query=query,
            chunks=chunks,
            context=context,
            has_relevant_context=bool(chunks),
            metadata={
                "retrieval_mode": "hybrid",
                "vector_results": len(vector_results),
                "keyword_results": len(keyword_results),
                "fused_results": len(fused_results),
                "final_results": len(final_results),
                "domain": DomainType.GENERAL.value,
            },
        )

    def _top_k(self) -> int:
        """Return configured retrieval depth."""

        return int(
            getattr(
                self.settings,
                "rag_top_k",
                getattr(
                    self.settings,
                    "top_k",
                    5,
                ),
            )
        )

    def _vector_search(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """Perform dense cosine-similarity search."""

        if not self._documents:
            return []

        try:
            query_vector = self.embeddings.embed_text(query)

            results: List[RetrievalResult] = []

            for document in self._documents.values():
                document_vector = self.embeddings.embed_text(
                    document.text
                )

                score = self._cosine_similarity(
                    query_vector,
                    document_vector,
                )

                results.append(
                    RetrievalResult(
                        chunk_id=document.chunk_id,
                        text=document.text,
                        score=float(score),
                        metadata={
                            "document_id": document.document_id,
                            "source": document.source,
                            "category": document.category,
                            "document_type": document.document_type,
                            "title": document.title,
                            "section": document.section,
                            **document.metadata,
                        },
                    )
                )

            results.sort(
                key=lambda item: item.score,
                reverse=True,
            )

            return results[:top_k]

        except Exception:
            return []

    def _fuse_results(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Fuse dense and keyword retrieval results."""

        if vector_results and keyword_results:
            return self.rrf.fuse(
                vector_results=vector_results,
                keyword_results=keyword_results,
                top_k=top_k,
            )

        if keyword_results:
            return keyword_results[:top_k]

        if vector_results:
            return vector_results[:top_k]

        return []

    def _rerank(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Apply the configured reranker."""

        if not results:
            return []

        try:
            return self.reranker.rerank(
                query=query,
                results=results,
            )
        except Exception:
            return results

    def _apply_threshold(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Apply relevance threshold."""

        if not results:
            return []

        threshold = float(
            getattr(
                self.settings,
                "rag_relevance_threshold",
                getattr(
                    self.settings,
                    "relevance_threshold",
                    0.60,
                ),
            )
        )

        filtered = [
            result
            for result in results
            if float(result.score) >= threshold
        ]

        if not filtered:
            filtered = [results[0]]

        final_top_k = int(
            getattr(
                self.settings,
                "rag_final_top_k",
                getattr(
                    self.settings,
                    "final_top_k",
                    3,
                ),
            )
        )

        return filtered[:final_top_k]

    @staticmethod
    def _cosine_similarity(
        left: Any,
        right: Any,
    ) -> float:
        """Calculate cosine similarity between vectors."""

        try:
            import numpy as np

            left_array = np.asarray(left, dtype=float)
            right_array = np.asarray(right, dtype=float)

            left_norm = np.linalg.norm(left_array)
            right_norm = np.linalg.norm(right_array)

            if left_norm == 0 or right_norm == 0:
                return 0.0

            return float(
                np.dot(left_array, right_array)
                / (left_norm * right_norm)
            )

        except Exception:
            return 0.0

    @staticmethod
    def _build_context(
        chunks: List[RAGChunk],
    ) -> str:
        """Build grounded context for the agent."""

        if not chunks:
            return ""

        sections: List[str] = []

        for index, chunk in enumerate(chunks, start=1):
            source = (
                chunk.metadata.get("source")
                or chunk.metadata.get("source_path")
                or "knowledge_base"
            )

            # RAGChunk uses `content`, not `text`.
            content = getattr(
                chunk,
                "content",
                "",
            )

            sections.append(
                f"[Source {index}: {source}]\n"
                f"{content.strip()}"
            )

        return "\n\n".join(sections)


RAGRetriever = GroundedKnowledgeProvider


def get_rag_retriever(
    settings: Optional[Settings] = None,
) -> GroundedKnowledgeProvider:
    """Return a configured JB Infra RAG provider."""

    return GroundedKnowledgeProvider(
        settings=settings,
    )


__all__ = [
    "GroundedContextResult",
    "GroundedKnowledgeProvider",
    "RAGRetriever",
    "get_rag_retriever",
]