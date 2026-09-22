# src/dependencies.py
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
            model=DEFAULT_AGENT_CONFIG.model_name,
            max_output_tokens=DEFAULT_AGENT_CONFIG.max_tokens,
            temperature=0.1
        )
        runtime = AgentRuntime(
            llm=llm,
            memory_store=None,
            retriever=RAGRetriever(),
        )
    return runtime