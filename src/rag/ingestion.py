"""Ingestion pipeline scanning knowledge_base directory and indexing documents."""

from src.rag.ingestion.pipeline import (
    IngestionPipeline,
    KnowledgeIngestionPipeline,
    main,
)

__all__ = [
    "IngestionPipeline",
    "KnowledgeIngestionPipeline",
    "main",
]

if __name__ == "__main__":
    main()