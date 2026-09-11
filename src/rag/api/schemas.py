from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RAGSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    tenant_id: str = Field(default="default", min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class RAGResultItem(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str
    category: str
    section: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGSearchResponse(BaseModel):
    query: str
    results: list[RAGResultItem] = Field(default_factory=list)
    context: str = ""
    has_relevant_context: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)