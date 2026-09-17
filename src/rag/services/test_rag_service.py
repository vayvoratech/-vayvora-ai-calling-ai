import asyncio
from src.rag.services.retriever import get_rag_retriever


async def main():
    retriever = get_rag_retriever()
    query = "What generative AI and cloud infrastructure services does Vayvora offer?"
    response = await retriever.search(query=query)
    print(f"Query: {response.query}")
    print(f"Found context: {response.has_relevant_context}")
    print(f"Results count: {len(response.results)}")
    print(f"Context snippet: {response.context[:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
