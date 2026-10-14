"""RAG services for the JB Infra demo MVP."""

from src.rag.services.fast_path import (
    FAST_PATH_RULES,
    get_deterministic_fast_path,
)

from src.rag.services.recommendation import (
    PROJECTS,
    ProjectRecommendationService,
    get_project_recommendations,
)

from src.rag.services.retriever import (
    RAGRetriever,
    get_rag_retriever,
)

__all__ = [
    "FAST_PATH_RULES",
    "get_deterministic_fast_path",
    "PROJECTS",
    "ProjectRecommendationService",
    "get_project_recommendations",
    "RAGRetriever",
    "get_rag_retriever",
]