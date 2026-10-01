"""Deterministic document parsing and section-aware chunking (backward compatibility re-export)."""

from src.rag.ingestion.chunker import DocumentChunker, MarkdownChunker
from src.rag.models import DocumentChunk

__all__ = [
    "DocumentChunk",
    "DocumentChunker",
    "MarkdownChunker",
]