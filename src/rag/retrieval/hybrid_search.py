"""Okapi BM25 keyword retriever for lexical search."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any, Dict, List, Optional

from src.rag.schemas import RetrievalResult


@dataclass
class KeywordDocument:
    """Document representation for lexical indexing."""

    chunk_id: str
    text: str
    source: str = ""
    category: str = "general"
    section: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BM25Retriever:
    """Lightweight pure-Python Okapi BM25 lexical retriever."""

    def __init__(self, documents: List[KeywordDocument], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b

        self.tokenized_documents = [
            self._tokenize(doc.text) for doc in documents
        ]
        self.document_lengths = [
            len(tokens) for tokens in self.tokenized_documents
        ]
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
            if self.document_lengths
            else 0.0
        )

        self.document_frequency = Counter()
        for tokens in self.tokenized_documents:
            for token in set(tokens):
                self.document_frequency[token] += 1

        self.total_documents = len(documents)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.documents:
            return []

        scores: List[tuple[float, KeywordDocument]] = []

        for index, document in enumerate(self.documents):
            tokens = self.tokenized_documents[index]
            term_frequency = Counter(tokens)
            score = 0.0
            doc_len = len(tokens)

            for term in query_tokens:
                freq = term_frequency.get(term, 0)
                if freq == 0:
                    continue

                doc_freq = self.document_frequency.get(term, 0)
                idf = math.log(
                    1 + (self.total_documents - doc_freq + 0.5) / (doc_freq + 0.5)
                )

                denom = freq + self.k1 * (
                    1 - self.b + self.b * (doc_len / (self.average_document_length or 1.0))
                )
                score += idf * (freq * (self.k1 + 1) / (denom or 1.0))

            if score > 0:
                scores.append((score, document))

        scores.sort(key=lambda x: x[0], reverse=True)

        results: List[RetrievalResult] = []
        for score, doc in scores[:top_k]:
            results.append(
                RetrievalResult(
                    chunk_id=doc.chunk_id,
                    text=doc.text,
                    score=float(score),
                    source=doc.source,
                    category=doc.category,
                    section=doc.section,
                    metadata={
                        **doc.metadata,
                        "retrieval_method": "bm25",
                        "bm25_score": float(score),
                    },
                )
            )

        return results
