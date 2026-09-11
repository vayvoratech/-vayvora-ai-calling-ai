import asyncio

from src.rag.services.retriever import RAGRetriever


TEST_CASES = [
    {
        "query": "What AI solutions does Vayvora provide?",
        "expected_relevant": True,
        "expected_sources": ["services/ai_solutions.md"],
    },
    {
        "query": "What services does Vayvora offer?",
        "expected_relevant": True,
        "expected_sources": [
            "services/ai_solutions.md",
            "services/web_development.md",
            "services/mobile.md",
            "services/saas.md",
            "services/devops.md",
        ],
    },
    {
        "query": "How can I contact Vayvora?",
        "expected_relevant": True,
        "expected_sources": ["company/contact.md"],
    },
    {
        "query": "What is Vayvora's pricing?",
        "expected_relevant": True,
        "expected_sources": [
            "pricing/ai.md",
            "pricing/maintenance.md",
        ],
    },
    {
        "query": "Tell me about Vayvora.",
        "expected_relevant": True,
        "expected_sources": ["company/about.md"],
    },
    {
        "query": "What payment methods does Vayvora accept?",
        "expected_relevant": True,
        "expected_sources": ["sales/payment.md"],
    },
    {
        "query": "What is Vayvora's sales process?",
        "expected_relevant": True,
        "expected_sources": ["sales/process.md"],
    },
    {
        "query": "What support does Vayvora provide?",
        "expected_relevant": True,
        "expected_sources": ["support/overview.md"],
    },
    {
        "query": "What careers are available at Vayvora?",
        "expected_relevant": True,
        "expected_sources": ["careers/overview.md"],
    },
    {
        "query": "What is Vayvora's office in London?",
        "expected_relevant": False,
        "expected_sources": [],
    },
    {
        "query": "What is Vayvora's office in Mumbai?",
        "expected_relevant": False,
        "expected_sources": [],
    },
    {
        "query": "What is Vayvora's annual revenue?",
        "expected_relevant": False,
        "expected_sources": [],
    },
    {
        "query": "Who is the CEO of Vayvora?",
        "expected_relevant": False,
        "expected_sources": [],
    },
    {
        "query": "How many employees does Vayvora have?",
        "expected_relevant": False,
        "expected_sources": [],
    },
]


async def evaluate():
    retriever = RAGRetriever()

    total = len(TEST_CASES)
    correct_relevance = 0
    source_hits = 0

    print("=" * 80)
    print("VAYVORA RAG EVALUATION")
    print("=" * 80)

    for index, test_case in enumerate(TEST_CASES, start=1):
        query = test_case["query"]
        expected_relevant = test_case["expected_relevant"]
        expected_sources = test_case["expected_sources"]

        response = await retriever.search(
            query=query,
            tenant_id="default",
        )

        actual_relevant = response.has_relevant_context

        relevance_correct = actual_relevant == expected_relevant

        if relevance_correct:
            correct_relevance += 1

        returned_sources = {
            result.source
            for result in response.results
        }

        source_correct = True

        if expected_sources:
            source_correct = any(
                source in returned_sources
                for source in expected_sources
            )

            if source_correct:
                source_hits += 1
        else:
            source_hits += 1

        status = "PASS" if relevance_correct and source_correct else "FAIL"

        print()
        print(f"[{index}/{total}] {status}")
        print(f"Query: {query}")
        print(f"Expected relevant: {expected_relevant}")
        print(f"Actual relevant:   {actual_relevant}")

        if response.results:
            print("Top results:")

            for result in response.results[:3]:
                print(
                    f"  score={result.score:.4f} | "
                    f"source={result.source} | "
                    f"section={result.section}"
                )
        else:
            print("Top results: NONE")

        print(f"Expected sources: {expected_sources}")
        print(f"Returned sources: {sorted(returned_sources)}")

    relevance_accuracy = correct_relevance / total

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"Total test cases:       {total}")
    print(f"Correct relevance:      {correct_relevance}")
    print(f"Relevance accuracy:     {relevance_accuracy:.2%}")
    print(f"Source correctness:     {source_hits}/{total}")
    print(f"Source accuracy:        {source_hits / total:.2%}")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(evaluate())