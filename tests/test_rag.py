import asyncio
import sys

from src.core.types import DomainType, RAGQuery
from src.rag.redis_client import RedisVectorStore
from src.rag.retriever import GroundedKnowledgeProvider

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


async def main():
    vector_store = RedisVectorStore()

    provider = GroundedKnowledgeProvider(
        vector_store=vector_store,
        in_memory=False,
    )

    queries = [
        "What courses are available in EduSaaS?",
        "List all EduSaaS courses",
        "Does EduSaaS offer Artificial Intelligence and Machine Learning?",
        "What technologies are taught in the AI/ML course?",
    ]

    for text in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {text}")
        print("=" * 80)

        query = RAGQuery(
            domain=DomainType.EDUSAAS,
            query_text=text,
            top_k=5,
            relevance_threshold=0.35,
        )

        result = await provider.retrieve_grounded_context(query)

        print(f"Knowledge available : {result.knowledge_available}")
        print(f"Service unavailable : {result.service_unavailable}")
        print(f"Chunks retrieved    : {len(result.chunks)}")

        for i, chunk in enumerate(result.chunks, 1):
            print(f"\n--- RESULT {i} ---")
            print(f"Doc ID : {chunk.doc_id}")
            print(f"Title  : {chunk.title}")
            print(f"Score  : {chunk.score}")
            print(f"Content:\n{chunk.content}")

        print("\n--- GROUNDED CONTEXT ---")
        print(result.formatted_context)

    await vector_store.close()


if __name__ == "__main__":
    asyncio.run(main())