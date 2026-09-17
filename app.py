import os
import sys
import time
import asyncio
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# 1. Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "src" else Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Suppress noisy warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.config import DEFAULT_AGENT_CONFIG
from src.agents.runtime import AgentRuntime
from src.rag.services.retriever import RAGRetriever

load_dotenv()

# Global runtime holder
runtime: Optional[AgentRuntime] = None

def get_agent_runtime() -> AgentRuntime:
    global runtime
    if runtime is None:
        llm = ChatGoogleGenerativeAI(
            model=DEFAULT_AGENT_CONFIG.model_name,
            max_output_tokens=DEFAULT_AGENT_CONFIG.max_tokens,
        )
        runtime = AgentRuntime(
            llm=llm,
            memory_store=None,
            retriever=RAGRetriever(),
        )
    return runtime

app = FastAPI(title="Vayvora AI Voice Agent API")

class ChatRequest(BaseModel):
    user_input: str
    messages: List[Dict[str, Any]] = []

@app.on_event("startup")
async def startup_event():
    agent = get_agent_runtime()
    await agent.start()

@app.on_event("shutdown")
async def shutdown_event():
    global runtime
    if runtime:
        await runtime.close()

@app.get("/config")
async def get_config():
    return {
        "model_name": DEFAULT_AGENT_CONFIG.model_name,
        "router_confidence_threshold": DEFAULT_AGENT_CONFIG.router_confidence_threshold,
        "router_temperature": DEFAULT_AGENT_CONFIG.router_temperature,
        "kb_resonance_weight": DEFAULT_AGENT_CONFIG.kb_resonance_weight,
        "tool_similarity_threshold": DEFAULT_AGENT_CONFIG.tool_similarity_threshold,
    }

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    agent = get_agent_runtime()
    
    state = {
        "session_id": "html_session",
        "user_input": payload.user_input,
        "tenant_id": "default",
        "messages": payload.messages,
    }

    t0 = time.perf_counter()
    result = await agent.run(state)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "response": result.get("response", "No response generated."),
        "latency": elapsed_ms,
        "route": result.get("route", "unknown"),
        "confidence": result.get("route_confidence", 0.0),
        "margin": result.get("route_margin", 0.0),
        "rationale": result.get("route_rationale", ""),
        "scores": result.get("similarity_scores", {}),
        "kb_resonance": result.get("kb_resonance", 0.0),
        "tool_name": result.get("tool_name"),
        "tool_input": result.get("tool_input"),
        "tool_result": result.get("tool_result"),
        "tool_confidence": result.get("tool_confidence"),
        "rag_context": result.get("rag_context", ""),
        "error": result.get("error"),
    }

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_file = Path(__file__).parent / "index.html"
    return html_file.read_text(encoding="utf-8")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)