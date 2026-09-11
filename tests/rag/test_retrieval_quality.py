from __future__ import annotations

import asyncio

from src.rag.services.retriever import get_rag_retriever


TEST_CASES = [
    {
        "query": "What AI solutions does Vayvora provide?",
        "expected_relevant": True,
    },
    {
        "query": "What services does Vayvora offer?",
        "expected_relevant": True,
    },
    {
        "query": "How can I contact Vayvora?",
        "expected_relevant": True,
    },
    {
        "query": "What is Vayvora's pricing?",
        "expected_relevant": True,
    },
    {
        "query": "What is Vayvora office in London?",
        "expected_relevant": False,
    },
    {
        "query": "What is Vayvora office in Mumbai?",
        "expected_relevant": False,
    },
    {
        "query": "What is Vayvora's revenue?",
        "expected_relevant": False,
    },
]


async def main() -> None:
    retriever = get_rag_retriever()

    print("=" * 80)
    print("VAYVORA RAG - RETRIEVAL QUALITY EVALUATION")
    print("=" * 80)

    for index, case in enumerate(TEST_CASES, start=1):
        query = case["query"]
        expected = case["expected_relevant"]

        response = await retriever.search(
            query=query,
            tenant_id="default",
        )

        print(f"\n[{index}] {query}")
        print(f"Expected relevant: {expected}")
        print(f"Retrieved relevant: {response.has_relevant_context}")

        if response.results:
            print("Top scores:")

            for result in response.results[:5]:
                print(
                    f"  {result.score:.4f} | "
                    f"{result.source} | "
                    f"{result.section}"
                )
        else:
            print("  No results")

        print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())