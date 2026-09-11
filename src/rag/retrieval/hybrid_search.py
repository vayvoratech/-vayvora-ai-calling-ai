from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from src.rag.schemas import RetrievalResult


@dataclass
class KeywordDocument:
    chunk_id: str
    text: str
    source: str
    category: str
    section: str | None
    metadata: dict


class BM25Retriever:
    """
    Lightweight BM25 keyword retriever.

    Designed for the current Vayvora knowledge base.

    For a small knowledge base this avoids requiring
    RediSearch while still providing lexical retrieval.
    """

    def __init__(self, documents: list[KeywordDocument]):
        self.documents = documents

        self.k1 = 1.5
        self.b = 0.75

        self.tokenized_documents = [
            self._tokenize(document.text)
            for document in documents
        ]

        self.document_lengths = [
            len(tokens)
            for tokens in self.tokenized_documents
        ]

        self.average_document_length = (
            sum(self.document_lengths)
            / len(self.document_lengths)
            if self.document_lengths
            else 0
        )

        self.document_frequency = Counter()

        for tokens in self.tokenized_documents:
            for token in set(tokens):
                self.document_frequency[token] += 1

        self.total_documents = len(documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:

        query_tokens = self._tokenize(query)

        if not query_tokens or not self.documents:
            return []

        scores = []

        for index, document in enumerate(self.documents):

            tokens = self.tokenized_documents[index]
            term_frequency = Counter(tokens)

            score = 0.0
            document_length = len(tokens)

            for term in query_tokens:

                frequency = term_frequency.get(term, 0)

                if frequency == 0:
                    continue

                document_frequency = self.document_frequency.get(
                    term,
                    0,
                )

                idf = math.log(
                    1
                    + (
                        self.total_documents
                        - document_frequency
                        + 0.5
                    )
                    / (
                        document_frequency
                        + 0.5
                    )
                )

                denominator = (
                    frequency
                    + self.k1
                    * (
                        1
                        - self.b
                        + self.b
                        * (
                            document_length
                            / self.average_document_length
                        )
                    )
                )

                score += (
                    idf
                    * (
                        frequency
                        * (self.k1 + 1)
                        / denominator
                    )
                )

            if score > 0:
                scores.append(
                    (
                        score,
                        document,
                    )
                )

        scores.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        results = []

        for score, document in scores[:top_k]:
            results.append(
                RetrievalResult(
                    chunk_id=document.chunk_id,
                    text=document.text,
                    score=float(score),
                    source=document.source,
                    category=document.category,
                    section=document.section,
                    metadata={
                        **document.metadata,
                        "retrieval_method": "bm25",
                    },
                )
            )

        return results