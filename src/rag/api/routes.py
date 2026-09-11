from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.rag.api.schemas import (
    RAGResultItem,
    RAGSearchRequest,
    RAGSearchResponse,
)
from src.rag.services.retriever import get_rag_retriever


router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


@router.post(
    "/search",
    response_model=RAGSearchResponse,
)
async def search_rag(request: RAGSearchRequest) -> RAGSearchResponse:
    """
    Search the Vayvora knowledge base using the production RAG pipeline.
    """

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    try:
        retriever = get_rag_retriever()

        rag_response = await retriever.search(
            query=query,
            tenant_id=request.tenant_id,
            top_k=request.top_k,
        )

        results = [
            RAGResultItem(
                chunk_id=result.chunk_id,
                text=result.text,
                score=result.score,
                source=result.source,
                category=result.category,
                section=result.section,
                metadata=result.metadata,
            )
            for result in rag_response.results
        ]

        return RAGSearchResponse(
            query=rag_response.query,
            results=results,
            context=rag_response.context,
            has_relevant_context=rag_response.has_relevant_context,
            metadata=rag_response.metadata,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG search failed: {exc}",
        ) from exc