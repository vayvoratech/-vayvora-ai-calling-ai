import asyncio

from src.rag.retriever import GroundedKnowledgeProvider
from src.core.types import DomainType, RAGQuery


async def main():
    provider = GroundedKnowledgeProvider()

    query = RAGQuery(
        domain=DomainType.VAYVORA,
        query_text="What AI solutions does Vayvora provide?",
        top_k=5,
        relevance_threshold=0.0,
    )

    results = await provider.search(query)

    print(f"RESULTS: {len(results)}")

    for result in results:
        print(f"\n--- {result.title} | score={result.score} ---")
        print(result.content[:500])


if __name__ == "__main__":
    asyncio.run(main())