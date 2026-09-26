"""RAG services: hybrid retriever, deterministic fast-path, and recommendations."""

from src.rag.services.fast_path import (
    FAST_PATH_RULES,
    get_deterministic_fast_path,
)
from src.rag.services.recommendation import (
    CourseRecommendationService,
    get_course_recommendations,
)
from src.rag.services.retriever import RAGRetriever, get_rag_retriever

__all__ = [
    "FAST_PATH_RULES",
    "get_deterministic_fast_path",
    "CourseRecommendationService",
    "get_course_recommendations",
    "RAGRetriever",
    "get_rag_retriever",
]
