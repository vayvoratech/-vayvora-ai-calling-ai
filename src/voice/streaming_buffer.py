"""Streaming phrase buffer for incremental LLM token-to-speech synthesis.

Accumulates arbitrary incoming text fragments and emits clean, prosodically
natural phrase chunks for low-latency text-to-speech (TTS) streaming.

Key Responsibilities:
1. Fragment Accumulation: Handles tokens arriving in arbitrary slices, including words
   split across token boundaries (e.g. 'Hy' -> 'der' -> 'abad').
2. Sentence-Ending Punctuation: Flushes at '.', '!', '?', and newline ('\\n') boundaries.
3. Clause Punctuation: Flushes at ',', ';', ':' only when the buffer contains at least
   `min_clause_words` (default 3 words).
4. Run-on Word Ceiling: Flushes when buffer reaches `max_words_ceiling` (default 8 words)
   without punctuation, safely completing the 8th word before splitting.
5. End-of-Stream Flush: Emits any remaining valid buffered text via `flush()`.
6. Decimal & Time Safety: Preserves decimal numbers (e.g. '1.5', '3.14') and time
   expressions (e.g. '10:30') without splitting across chunk boundaries.
7. Abbreviation Safety: Protects titles (e.g. 'Mr.', 'Dr.') and common units/terms
   (e.g. 'sq.', 'yd.', 'cr.', 'e.g.') from false sentence breaks.
8. Clean Boundaries: Never emits empty strings, whitespace-only chunks, or words
   chopped midway through tokenization.
"""

from __future__ import annotations

import re
from typing import List, Optional


# Titles and honorifics that must never trigger sentence breaks
TITLES = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
})

# Common abbreviations and real-estate/numerical units
COMMON_ABBREVIATIONS = frozenset({
    "sq", "yd", "yds", "ft", "in",
    "e.g", "i.e", "etc", "vs", "approx", "est", "dept", "apt", "no", "nos",
    "cr", "lakh", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
})

# Punctuation that attaches to preceding sentences/clauses
CLOSING_PUNCTUATION = frozenset({'"', "'", "”", "’", ")", "]", "}"})


class StreamingPhraseBuffer:
    """Zero-dependency incremental text buffer for low-latency TTS streaming."""

    def __init__(
        self,
        min_clause_words: int = 3,
        max_words_ceiling: int = 8,
    ) -> None:
        """Initialize the phrase buffer with configurable boundary thresholds.

        Args:
            min_clause_words: Minimum words required before clause punctuation
                (',', ';', ':') triggers a flush. Defaults to 3.
            max_words_ceiling: Maximum words allowed without punctuation before
                forcing a phrase flush at a word boundary. Defaults to 8.
        """
        if min_clause_words < 1:
            raise ValueError("min_clause_words must be at least 1")
        if max_words_ceiling < min_clause_words:
            raise ValueError("max_words_ceiling must be greater than or equal to min_clause_words")

        self.min_clause_words = min_clause_words
        self.max_words_ceiling = max_words_ceiling
        self._buffer: str = ""

    @property
    def is_empty(self) -> bool:
        """Return True if the internal buffer contains no text or only whitespace."""
        return not bool(self._buffer.strip())

    @property
    def buffer_text(self) -> str:
        """Return the current un-flushed raw buffer content."""
        return self._buffer

    def reset(self) -> None:
        """Clear all buffered text."""
        self._buffer = ""

    def feed(self, fragment: str) -> List[str]:
        """Feed an incremental text fragment and return complete phrases ready to speak.

        Args:
            fragment: Arbitrary incoming string slice from an LLM token stream.

        Returns:
            List of clean, speakable phrase strings. May be empty if the buffer
            has not yet reached a valid sentence, clause, or word-ceiling boundary.
        """
        if not fragment:
            return []

        self._buffer += fragment
        return self._extract_chunks()

    def flush(self) -> List[str]:
        """Flush all remaining buffered text at the end of the generation stream.

        Returns:
            List of any remaining valid phrase chunks. Empty chunks or whitespace-only
            strings are never returned.
        """
        chunks = self._extract_chunks()
        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining:
            chunks.append(remaining)
        return chunks

    def _extract_chunks(self) -> List[str]:
        """Iteratively extract all completed phrase chunks from the buffer."""
        chunks: List[str] = []
        while True:
            split_idx = self._find_next_split(self._buffer)
            if split_idx is None:
                break
            chunk = self._buffer[:split_idx].strip()
            self._buffer = self._buffer[split_idx:].lstrip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _find_next_split(self, text: str) -> Optional[int]:
        """Find the earliest safe index to split `text` into a speakable phrase.

        Returns:
            Index (int) where the chunk ends in `text`, or None if no safe
            boundary has been reached yet.
        """
        if not text:
            return None

        word_matches = list(re.finditer(r"\S+", text))

        for idx, ch in enumerate(text):
            # 1. Sentence-ending punctuation & newline boundaries
            if ch in (".", "!", "?", "\n"):
                if ch == "\n":
                    return idx + 1

                if ch == ".":
                    # Check decimal numbers (e.g. '1.5', '3.14')
                    if idx > 0 and text[idx - 1].isdigit():
                        if idx + 1 < len(text) and text[idx + 1].isdigit():
                            continue
                        if idx + 1 == len(text):
                            # Digit followed by dot at very end: wait for next token
                            return None

                    # Check ellipsis (e.g. '...')
                    if idx + 1 < len(text) and text[idx + 1] == ".":
                        continue
                    if idx > 0 and text[idx - 1] == "." and idx + 1 < len(text) and text[idx + 1] == ".":
                        continue

                    # Check single letter initials (e.g. 'J.', 'B.' in 'J.B.', 'A.' in 'A. Smith')
                    if re.search(r"(?:^|[\s.])[A-Za-z]\Z", text[:idx]):
                        continue

                    # Check titles and common abbreviations
                    prev_token_match = re.search(r"([A-Za-z.]+)\Z", text[:idx])
                    if prev_token_match:
                        prev_token = prev_token_match.group(1).lower().rstrip(".")
                        if prev_token in TITLES:
                            continue
                        if prev_token in COMMON_ABBREVIATIONS:
                            after = text[idx + 1 :]
                            # Only treat abbreviation dot as sentence end if followed by capitalized word
                            if not re.match(r"\s+[A-Z]", after):
                                continue

                # Valid sentence ending found
                end_pos = idx + 1
                while end_pos < len(text) and text[end_pos] in CLOSING_PUNCTUATION:
                    end_pos += 1

                # If word ceiling was reached earlier in this sentence, prioritize ceiling
                if len(word_matches) >= self.max_words_ceiling:
                    target_word = word_matches[self.max_words_ceiling - 1]
                    if target_word.end() < end_pos and (
                        len(word_matches) > self.max_words_ceiling
                        or (target_word.end() < len(text) and text[target_word.end()].isspace())
                    ):
                        return target_word.end()

                return end_pos

            # 2. Clause punctuation (',', ';', ':')
            elif ch in (",", ";", ":"):
                # Protect commas in numbers (e.g. '10,000') and colons in time (e.g. '10:30')
                if ch in (",", ":") and idx > 0 and text[idx - 1].isdigit():
                    if idx + 1 < len(text) and text[idx + 1].isdigit():
                        continue
                    if idx + 1 == len(text):
                        # Digit followed by colon/comma at very end of buffer: wait for next token
                        return None

                words_before = len(re.findall(r"\S+", text[: idx + 1]))
                if words_before >= self.min_clause_words:
                    end_pos = idx + 1
                    while end_pos < len(text) and text[end_pos] in CLOSING_PUNCTUATION:
                        end_pos += 1

                    # If word ceiling was reached earlier, prioritize ceiling
                    if len(word_matches) >= self.max_words_ceiling:
                        target_word = word_matches[self.max_words_ceiling - 1]
                        if target_word.end() < end_pos and (
                            len(word_matches) > self.max_words_ceiling
                            or (target_word.end() < len(text) and text[target_word.end()].isspace())
                        ):
                            return target_word.end()

                    return end_pos

        # 3. Word count ceiling without punctuation
        if len(word_matches) >= self.max_words_ceiling:
            target_word = word_matches[self.max_words_ceiling - 1]
            if len(word_matches) > self.max_words_ceiling or (
                target_word.end() < len(text) and text[target_word.end()].isspace()
            ):
                return target_word.end()

        return None
