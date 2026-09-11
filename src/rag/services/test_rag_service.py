import asyncio

from src.rag.services.retriever import RAGRetriever


async def main():
    print("Initializing RAG retriever...")

    retriever = RAGRetriever()

    print("Searching knowledge base...")

    response = await retriever.search(
        query="What AI solutions does Vayvora provide?",
        tenant_id="default",
    )

    print("\n==============================")
    print("RAG SEARCH RESULT")
    print("==============================")

    print(f"Has relevant context : {response.has_relevant_context}")
    print(f"Results              : {len(response.results)}")
    print(f"Metadata             : {response.metadata}")

    print("\n==============================")
    print("RETRIEVED DOCUMENTS")
    print("==============================")

    for index, result in enumerate(response.results, start=1):
        print(f"\n--- Result {index} ---")
        print(f"Score    : {result.score:.4f}")
        print(f"Source   : {result.source}")
        print(f"Section : {result.section}")
        print(f"Category: {result.category}")
        print(f"\n{result.text}")

    print("\n==============================")
    print("FINAL RAG CONTEXT")
    print("==============================")

    print(response.context)


if __name__ == "__main__":
    asyncio.run(main())