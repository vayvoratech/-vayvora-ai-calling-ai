from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.config import DEFAULT_AGENT_CONFIG
from src.agents.runtime import AgentRuntime
from src.rag.services.retriever import RAGRetriever

runtime: Optional[AgentRuntime] = None

def get_agent_runtime() -> AgentRuntime:
    global runtime
    if runtime is None:
        llm = ChatGoogleGenerativeAI(
            model=DEFAULT_AGENT_CONFIG.model_name, # "gemini-3.5-flash-lite"
            max_output_tokens=150,
            streaming=True
            # temperature omitted: gemini-3.5-flash-lite uses fixed sampling defaults
        )
        runtime = AgentRuntime(
            llm=llm,
            memory_store=None,
            retriever=RAGRetriever(),
        )
    return runtime