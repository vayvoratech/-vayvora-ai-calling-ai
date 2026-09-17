import asyncio
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.config import DEFAULT_AGENT_CONFIG
from src.agents.runtime import AgentRuntime
from src.rag.services.retriever import RAGRetriever
from dotenv import load_dotenv

load_dotenv()

async def main():
    # 1. Initialize Gemini Flash Lite
    # Omit 'temperature' (uses fixed defaults) and set tool_config mode to 'NONE' to suppress AFC warnings
    llm = ChatGoogleGenerativeAI(
        model=DEFAULT_AGENT_CONFIG.model_name,
        max_output_tokens=DEFAULT_AGENT_CONFIG.max_tokens,
        tool_config={"function_calling_config": {"mode": "NONE"}},
    )

    # 2. Wire RAG Retriever and LLM into Agent Runtime
    runtime = AgentRuntime(
        llm=llm,
        memory_store=None,
        retriever=RAGRetriever(),
    )

    # 3. Boot MCP client connection and compile the graph
    print("Starting MCP Client and Compiling Graph...")
    await runtime.start()
    print("Agent is ready.")

    try:
        # -------------------------------------------------------------
        # Scenario A: MCP Action (Calendar booking)
        # -------------------------------------------------------------
        print("\n--- Testing MCP Tool Flow ---")
        mcp_state = {
            "session_id": "call_123",
            "user_input": "Schedule a team sync meeting tomorrow at 3pm",
            "tenant_id": "default",
        }
        result_mcp = await runtime.run(mcp_state)
        print("Final Output:", result_mcp.get("response"))
        print("Detailed Error:", result_mcp.get("error"))

        # -------------------------------------------------------------
        # Scenario B: Knowledge Base RAG Search
        # -------------------------------------------------------------
        print("\n--- Testing RAG Flow ---")
        rag_state = {
            "session_id": "call_123",
            "user_input": "where the vayvora company is located?",
            "tenant_id": "default",
        }
        result_rag = await runtime.run(rag_state)
        print("Final Output:", result_rag.get("response"))
        print("Detailed Error:", result_rag.get("error"))

    finally:
        # 4. Clean up MCP background sub-process
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())