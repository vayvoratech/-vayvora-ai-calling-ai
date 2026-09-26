"""Knowledge ingestion pipeline coordinating document loading, chunking, and indexing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.types import DomainType, RAGChunk
from src.logging import get_logger
from src.rag.config import rag_config
from src.rag.ingestion.chunker import MarkdownChunker
from src.rag.ingestion.incremental import IncrementalIngestionService
from src.rag.ingestion.loader import MarkdownDocumentLoader
from src.rag.ingestion.manifest import ManifestStore
from src.rag.models import DocumentChunk

logger = get_logger("rag.ingestion.pipeline")


class IngestionPipeline:
    """Pipelines documents from knowledge_base directory into structured chunks or vector stores."""

    def __init__(
        self,
        knowledge_provider: Any = None,
        chunker: Optional[MarkdownChunker] = None,
        loader: Optional[MarkdownDocumentLoader] = None,
        manifest: Optional[ManifestStore] = None,
        base_dir: Optional[Path | str] = None,
    ) -> None:
        self.provider = knowledge_provider
        self.chunker = chunker or MarkdownChunker()
        self.loader = loader or MarkdownDocumentLoader(base_dir=base_dir)
        self.manifest = manifest or ManifestStore()
        self.base_dir = Path(base_dir or rag_config.knowledge_base_dir)

    def run(self, tenant_id: str = "default") -> List[DocumentChunk]:
        """Synchronously load and chunk knowledge documents for a tenant/domain."""
        documents = self.loader.load_documents(tenant_id=tenant_id)
        all_chunks: List[DocumentChunk] = []

        for doc in documents:
            chunks = self.chunker.chunk_document(doc, tenant_id=tenant_id)
            all_chunks.extend(chunks)

        return all_chunks

    async def ingest_domain(self, domain: DomainType) -> Dict[str, int]:
        """Ingest all documents for a specific business domain into knowledge provider."""
        domain_name = domain.value
        domain_dir = self.base_dir / domain_name

        if not domain_dir.is_dir():
            logger.warning("Domain knowledge directory does not exist: %s", domain_dir)
            return {"files_processed": 0, "chunks_indexed": 0}

        documents = self.loader.load_documents(tenant_id=domain_name)
        files_processed = 0
        chunks_indexed = 0

        for doc in documents:
            chunks = self.chunker.chunk_document(doc, tenant_id=domain_name)
            for chunk in chunks:
                rag_chunk = chunk.to_rag_chunk(domain=domain)
                if self.provider is not None:
                    await self.provider.index_document(rag_chunk)
                chunks_indexed += 1
            files_processed += 1

        logger.info(
            "Ingestion complete for domain '%s': %d files, %d chunks.",
            domain_name,
            files_processed,
            chunks_indexed,
        )
        return {
            "files_processed": files_processed,
            "chunks_indexed": chunks_indexed,
        }

    async def ingest_all(self) -> Dict[DomainType, Dict[str, int]]:
        """Ingest documents across all supported domains."""
        summary: Dict[DomainType, Dict[str, int]] = {}
        for domain in [DomainType.EDUSAAS, DomainType.VAYVORA]:
            summary[domain] = await self.ingest_domain(domain)
        return summary


# Backward compatibility alias
KnowledgeIngestionPipeline = IngestionPipeline


async def async_main():
    from src.rag.retriever import GroundedKnowledgeProvider

    provider = GroundedKnowledgeProvider()
    pipeline = IngestionPipeline(knowledge_provider=provider)
    summary = await pipeline.ingest_all()

    print("\n" + "=" * 50)
    print("KNOWLEDGE BASE INGESTION COMPLETE")
    print("=" * 50)
    for domain, res in summary.items():
        print(f"Domain: {domain.value:<10} | Files: {res['files_processed']:<3} | Chunks: {res['chunks_indexed']}")
    print("=" * 50)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
