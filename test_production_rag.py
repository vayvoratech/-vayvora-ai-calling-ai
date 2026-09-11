import asyncio
from src.rag.services.retriever import RAGRetriever

async def main():
    retriever = RAGRetriever()

    questions = [
        "What AI solutions does Vayvora provide?",
        "What services does Vayvora offer?",
        "Tell me about Vayvora pricing.",
        "How can I contact Vayvora?",
        "What is Vayvora?",
    ]

    for question in questions:
        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        response = await retriever.search(
            query=question,
            tenant_id="default"
        )

        print(f"Relevant context: {response.has_relevant_context}")
        print(f"Results: {len(response.results)}")
        print(f"Metadata: {response.metadata}")

        for index, result in enumerate(response.results, start=1):
            print(
                f"\n{index}. "
                f"{result.score:.4f} | "
                f"{result.source} | "
                f"{result.section}"
            )
            print(result.text[:500])


if __name__ == "__main__":
    asyncio.run(main())
