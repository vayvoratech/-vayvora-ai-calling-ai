"""Markdown chunking utilities for the JB Infra knowledge base."""

from __future__ import annotations

import re
from typing import List


def chunk_markdown(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[str]:
    """Split Markdown into retrieval-friendly chunks.

    The chunker keeps paragraphs together where possible and uses
    character-based boundaries for the JB Infra demo MVP.
    """

    if not text or not text.strip():
        return []

    text = text.replace("\r\n", "\n").strip()

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    # Normalize excessive blank lines while preserving Markdown structure.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        candidate = (
            f"{current}\n\n{paragraph}"
            if current
            else paragraph
        )

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())

        # A single paragraph may itself be larger than the target size.
        if len(paragraph) <= chunk_size:
            current = paragraph
            continue

        paragraph_chunks = _split_long_text(
            paragraph,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if paragraph_chunks:
            chunks.extend(
                paragraph_chunks[:-1]
            )
            current = paragraph_chunks[-1]

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _split_long_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Split an oversized Markdown section with word boundaries."""

    words = text.split()

    if not words:
        return []

    chunks: List[str] = []
    current_words: List[str] = []
    current_length = 0

    for word in words:
        additional_length = (
            len(word)
            if not current_words
            else len(word) + 1
        )

        if (
            current_words
            and current_length + additional_length
            > chunk_size
        ):
            chunk = " ".join(current_words).strip()

            if chunk:
                chunks.append(chunk)

            overlap_words: List[str] = []
            overlap_length = 0

            for previous_word in reversed(
                current_words
            ):
                word_length = (
                    len(previous_word)
                    if not overlap_words
                    else len(previous_word) + 1
                )

                if (
                    overlap_length + word_length
                    > chunk_overlap
                ):
                    break

                overlap_words.insert(
                    0,
                    previous_word,
                )
                overlap_length += word_length

            current_words = overlap_words
            current_length = overlap_length

        current_words.append(word)

        current_length = (
            len(" ".join(current_words))
        )

    if current_words:
        chunks.append(
            " ".join(current_words).strip()
        )

    return chunks


__all__ = [
    "chunk_markdown",
]