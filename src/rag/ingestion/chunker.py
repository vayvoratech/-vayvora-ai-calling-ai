from __future__ import annotations

import hashlib
import re

from src.rag.config import rag_config
from src.rag.schemas import DocumentChunk


class MarkdownChunker:
    """
    Structure-aware Markdown chunker.

    Keeps headings attached to their content so that chunks
    retain the meaning and context of the original document.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = (
            chunk_size or rag_config.chunk_size
        )
        self.chunk_overlap = (
            chunk_overlap or rag_config.chunk_overlap
        )

    def _split_sections(
        self,
        text: str,
    ) -> list[tuple[str, str]]:

        sections: list[tuple[str, str]] = []

        current_heading = "Introduction"
        current_lines: list[str] = []

        for line in text.splitlines():

            heading_match = re.match(
                r"^(#{1,6})\s+(.+)$",
                line.strip(),
            )

            if heading_match:

                if current_lines:
                    content = "\n".join(
                        current_lines
                    ).strip()

                    if content:
                        sections.append(
                            (
                                current_heading,
                                content,
                            )
                        )

                current_heading = (
                    heading_match.group(2).strip()
                )

                current_lines = []

            else:
                current_lines.append(line)

        if current_lines:
            content = "\n".join(
                current_lines
            ).strip()

            if content:
                sections.append(
                    (
                        current_heading,
                        content,
                    )
                )

        return sections

    def _split_text(
        self,
        text: str,
    ) -> list[str]:

        text = text.strip()

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []

        start = 0
        text_length = len(text)

        while start < text_length:

            end = min(
                start + self.chunk_size,
                text_length,
            )

            if end >= text_length:
                chunk = text[start:].strip()

                if chunk:
                    chunks.append(chunk)

                break

            # Prefer paragraph boundaries.
            split_at = text.rfind(
                "\n\n",
                start,
                end,
            )

            # Then sentence boundaries.
            if split_at <= start:
                split_at = text.rfind(
                    ". ",
                    start,
                    end,
                )

                if split_at > start:
                    split_at += 1

            # Then word boundary.
            if split_at <= start:
                split_at = text.rfind(
                    " ",
                    start,
                    end,
                )

            # Absolute fallback.
            if split_at <= start:
                split_at = end

            chunk = text[
                start:split_at
            ].strip()

            if chunk:
                chunks.append(chunk)

            next_start = (
                split_at
                - self.chunk_overlap
            )

            if next_start <= start:
                next_start = split_at

            start = next_start

        return chunks

    def chunk_document(
        self,
        document: dict,
        tenant_id: str = "default",
    ) -> list[DocumentChunk]:

        chunks: list[DocumentChunk] = []

        sections = self._split_sections(
            document["text"]
        )

        chunk_number = 0

        for section, content in sections:

            section_chunks = self._split_text(
                content
            )

            for chunk_text in section_chunks:

                # Stable deterministic ID.
                raw_id = (
                    f"{tenant_id}:"
                    f"{document['source']}:"
                    f"{section}:"
                    f"{chunk_number}:"
                    f"{chunk_text}"
                )

                chunk_id = hashlib.sha256(
                    raw_id.encode("utf-8")
                ).hexdigest()

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document[
                            "document_id"
                        ],
                        tenant_id=tenant_id,
                        text=chunk_text,
                        source=document[
                            "source"
                        ],
                        category=document[
                            "category"
                        ],
                        section=section,
                        version="1",
                        metadata={
                            "chunk_number": chunk_number,
                            "document_type": (
                                "company"
                                if document[
                                    "category"
                                ] == "company"
                                else document[
                                    "category"
                                ]
                            ),
                        },
                    )
                )

                chunk_number += 1

        return chunks