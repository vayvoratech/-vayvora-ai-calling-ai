"""Structure-aware Markdown chunking with heading preservation and deterministic chunk IDs."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from src.core.types import DomainType
from src.rag.config import rag_config
from src.rag.models import DocumentChunk


class MarkdownChunker:
    """Splits Markdown and text documents while preserving section hierarchy and context."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        self.chunk_size = chunk_size or rag_config.chunk_size
        self.chunk_overlap = chunk_overlap or rag_config.chunk_overlap
        if self.chunk_size <= self.chunk_overlap:
            raise ValueError("chunk_size must be strictly greater than chunk_overlap.")

    def chunk_document(
        self,
        document: Dict[str, Any],
        tenant_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Chunk a document dictionary (document_id, source, category, text, tenant_id)."""
        doc_tenant = tenant_id or document.get("tenant_id", "default")
        source = document.get("source", document.get("document_id", "document.md"))
        category = document.get("category", "general")
        text = document.get("text", "")
        document_id = document.get("document_id", "")

        return self.chunk_text(
            text=text,
            domain_or_tenant=doc_tenant,
            category=category,
            source_file=source,
            document_id=document_id,
        )

    def chunk_file(
        self,
        file_path: Path | str,
        domain: DomainType | str,
        category: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Parse a file into structured DocumentChunk objects."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {file_path}")

        raw_text = path.read_text(encoding="utf-8")
        inferred_category = category or path.parent.name
        tenant_val = domain.value if isinstance(domain, DomainType) else str(domain)

        return self.chunk_text(
            text=raw_text,
            domain_or_tenant=tenant_val,
            category=inferred_category,
            source_file=path.name,
            document_id=path.stem.lower().replace(" ", "_"),
        )

    def chunk_text(
        self,
        text: str,
        domain_or_tenant: Optional[DomainType | str] = None,
        category: str = "general",
        source_file: str = "",
        document_id: str = "",
        domain: Optional[DomainType | str] = None,
        tenant_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Chunk Markdown while preserving title and section context."""
        cleaned = text.strip()
        if not cleaned:
            return []

        target = domain if domain is not None else (domain_or_tenant if domain_or_tenant is not None else (tenant_id or "default"))
        tenant_str = (
            target.value
            if isinstance(target, DomainType)
            else str(target)
        )

        document_title = self._extract_document_title(cleaned)
        sections = self._split_into_sections(cleaned)
        chunks_text: List[Dict[str, str]] = []

        for section in sections:
            heading = section["heading"]
            content = section["content"]

            if not content.strip():
                continue

            section_context = self._build_section_context(
                document_title=document_title,
                heading=heading,
            )

            full_section_text = f"{section_context}\n\n{content.strip()}"

            if len(full_section_text) <= self.chunk_size:
                chunks_text.append({
                    "text": full_section_text,
                    "section": heading,
                })
                continue

            sub_chunks = self._split_large_section(
                section_text=full_section_text,
                section_heading=heading,
            )
            chunks_text.extend(sub_chunks)

        if not chunks_text:
            chunks_text = [{
                "text": cleaned,
                "section": document_title,
            }]

        return self._build_document_chunks(
            chunks_text=chunks_text,
            document_title=document_title,
            tenant_id=tenant_str,
            category=category,
            source_file=source_file,
            document_id=document_id,
        )

    def _extract_document_title(self, text: str) -> str:
        match = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return "Untitled Document"

    def _split_into_sections(self, text: str) -> List[Dict[str, str]]:
        lines = text.splitlines()
        sections: List[Dict[str, str]] = []
        current_heading = ""
        current_content: List[str] = []

        for line in lines:
            if re.match(r"^#\s+", line):
                continue

            heading_match = re.match(r"^(#{2,6})\s+(.+?)\s*$", line)
            if heading_match:
                if current_heading or current_content:
                    sections.append({
                        "heading": current_heading,
                        "content": "\n".join(current_content).strip(),
                    })
                current_heading = heading_match.group(2).strip()
                current_content = []
                continue

            current_content.append(line)

        if current_heading or current_content:
            sections.append({
                "heading": current_heading,
                "content": "\n".join(current_content).strip(),
            })

        return sections

    def _build_section_context(self, document_title: str, heading: str) -> str:
        if heading:
            return f"Document: {document_title}\nSection: {heading}"
        return f"Document: {document_title}"

    def _split_large_section(
        self,
        section_text: str,
        section_heading: str,
    ) -> List[Dict[str, str]]:
        paragraphs = [
            p.strip() for p in section_text.split("\n\n") if p.strip()
        ]
        if not paragraphs:
            return []

        chunks: List[Dict[str, str]] = []
        current: List[str] = []
        current_len = 0

        for p in paragraphs:
            p_len = len(p)
            if current and current_len + p_len + 2 > self.chunk_size:
                chunks.append({
                    "text": "\n\n".join(current),
                    "section": section_heading,
                })
                overlap = self._get_overlap(current)
                if overlap:
                    current = [overlap, p]
                    current_len = len(overlap) + p_len + 2
                else:
                    current = [p]
                    current_len = p_len
            else:
                current.append(p)
                current_len += p_len + 2

        if current:
            chunks.append({
                "text": "\n\n".join(current),
                "section": section_heading,
            })

        # Break any individual chunk still larger than chunk_size
        final_chunks: List[Dict[str, str]] = []
        step = max(1, self.chunk_size - self.chunk_overlap)

        for c in chunks:
            c_text = c["text"]
            if len(c_text) <= self.chunk_size:
                final_chunks.append(c)
                continue

            start = 0
            while start < len(c_text):
                piece = c_text[start : start + self.chunk_size].strip()
                if piece:
                    final_chunks.append({
                        "text": piece,
                        "section": section_heading,
                    })
                start += step

        return final_chunks

    def _get_overlap(self, paragraphs: List[str]) -> str:
        if not paragraphs:
            return ""
        overlap_parts: List[str] = []
        total = 0
        for p in reversed(paragraphs):
            if total + len(p) > self.chunk_overlap:
                break
            overlap_parts.insert(0, p)
            total += len(p)
        return "\n\n".join(overlap_parts)

    def _build_document_chunks(
        self,
        chunks_text: List[Dict[str, str]],
        document_title: str,
        tenant_id: str,
        category: str,
        source_file: str,
        document_id: str = "",
    ) -> List[DocumentChunk]:
        total = len(chunks_text)
        clean_stem = (
            Path(source_file).stem.lower().replace(" ", "_")
            if source_file
            else (document_id or "chunk")
        )
        doc_id = document_id or clean_stem

        chunks: List[DocumentChunk] = []
        for index, item in enumerate(chunks_text, start=1):
            text_content = item["text"].strip()
            section = item["section"]

            # Deterministic chunk ID
            chunk_id = f"{tenant_id}:{category}:{clean_stem}:{index}"

            metadata = {
                "tenant_id": tenant_id,
                "domain": tenant_id,
                "category": category,
                "source_file": source_file,
                "source": source_file,
                "document_id": doc_id,
                "title": document_title,
                "section": section,
                "char_length": len(text_content),
                "chunk_index": index,
                "total_chunks": total,
            }

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=text_content,
                    tenant_id=tenant_id,
                    document_id=doc_id,
                    source=source_file,
                    category=category,
                    section=section,
                    chunk_index=index,
                    total_chunks=total,
                    metadata=metadata,
                )
            )

        return chunks


# Alias for backward compatibility
DocumentChunker = MarkdownChunker
