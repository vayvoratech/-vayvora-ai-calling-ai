import asyncio

from src.agents.nodes.router_node import RouterNode


async def main():
    router = RouterNode()

    test_questions = [
        "hello",
        "What AI solutions does Vayvora provide?",
        "What services does Vayvora offer?",
        "What is your pricing?",
        "Tell me about Vayvora",
        "check my appointment",
        "book an appointment",
        "How are you?",
    ]

    for question in test_questions:
        state = await router.run({
            "user_input": question
        })

        print(
            f"{question:<45} -> "
            f"{state.get('route')}"
        )


if __name__ == "__main__":
    asyncio.run(main())
