import asyncio

from src.rag.services.retriever import RAGRetriever


async def main():
    retriever = RAGRetriever()

    response = await retriever.search(
        query="What AI solutions does Vayvora provide?",
        tenant_id="default",
    )

    print("Has relevant context:", response.has_relevant_context)
    print("Results:", len(response.results))
    print("Metadata:", response.metadata)

    print("\n--- CONTEXT ---\n")
    print(response.context)


if __name__ == "__main__":
    asyncio.run(main())
