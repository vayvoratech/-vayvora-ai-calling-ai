"""Unit tests for document parsing, deterministic chunking, and metadata extraction."""

from pathlib import Path
import pytest
from src.core.types import DomainType
from src.rag.chunker import DocumentChunker


class TestDocumentChunker:
    """Test deterministic text and markdown chunking."""

    def test_chunk_size_greater_than_overlap_validation(self):
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=150)

    def test_deterministic_text_chunking(self):
        chunker = DocumentChunker(chunk_size=150, chunk_overlap=30)
        sample_text = (
            "# Introduction to Applied AI\n\n"
            "This course covers the foundations of transformers, self-attention, and large language models.\n\n"
            "Students will build interactive voice agents using modern open-source neural components.\n\n"
            "Prerequisites include basic familiarity with Python and linear algebra."
        )

        chunks = chunker.chunk_text(
            text=sample_text,
            domain=DomainType.EDUSAAS,
            category="courses",
            source_file="applied_ai.md",
        )

        assert len(chunks) >= 2
        # Title extracted from header
        assert chunks[0].title == "Introduction to Applied AI"
        assert chunks[0].domain == DomainType.EDUSAAS
        assert chunks[0].category == "courses"
        assert chunks[0].source_file == "applied_ai.md"
        assert chunks[0].chunk_index == 1
        assert chunks[0].total_chunks == len(chunks)

        # Re-running chunking produces identical chunk IDs and lengths
        chunks_repeat = chunker.chunk_text(
            text=sample_text,
            domain=DomainType.EDUSAAS,
            category="courses",
            source_file="applied_ai.md",
        )
        assert [c.chunk_id for c in chunks] == [c.chunk_id for c in chunks_repeat]

    def test_empty_text_returns_empty_list(self):
        chunker = DocumentChunker()
        assert chunker.chunk_text("", DomainType.EDUSAAS, "general", "empty.md") == []
        assert chunker.chunk_text("   \n\n   ", DomainType.EDUSAAS, "general", "empty.md") == []

    def test_malformed_nonexistent_file_raises_error(self):
        chunker = DocumentChunker()
        with pytest.raises(FileNotFoundError):
            chunker.chunk_file(
                file_path=Path("non_existent_directory/fake_file.md"),
                domain=DomainType.EDUSAAS,
            )

    def test_large_single_paragraph_chunking(self):
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        long_paragraph = "Word " * 60  # ~300 characters without line breaks

        chunks = chunker.chunk_text(
            text=long_paragraph,
            domain=DomainType.VAYVORA,
            category="services",
            source_file="long_doc.md",
        )
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c.text) <= 200
