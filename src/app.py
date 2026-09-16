import os
import sys
from pathlib import Path

# 1. Resolve project root so imports like 'from src...' work consistently
PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "src" else Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Suppress noisy terminal/HF warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import asyncio
import logging
import time
import warnings
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents.config import DEFAULT_AGENT_CONFIG
from src.agents.runtime import AgentRuntime
from src.rag.services.retriever import RAGRetriever

load_dotenv()

logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Vayvora AI Testbed", layout="wide")
st.title("🎙️ Vayvora AI Voice Agent Tester")


@st.cache_resource
def get_runtime():
    """Initializes runtime once and keeps it alive across re-renders."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    llm = ChatGoogleGenerativeAI(
        model=DEFAULT_AGENT_CONFIG.model_name,
        max_output_tokens=DEFAULT_AGENT_CONFIG.max_tokens,
    )
    runtime = AgentRuntime(
        llm=llm,
        memory_store=None,
        retriever=RAGRetriever(),
    )
    loop.run_until_complete(runtime.start())
    return runtime, loop


runtime, event_loop = get_runtime()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "latency" in msg:
            st.caption(
                f"⏱️ **Latency:** {msg['latency']:.1f} ms | "
                f"🧭 **Route:** `{msg['route']}` | "
                f"⚠️ **Error:** `{msg['error']}`"
            )
            # Display retrieved RAG context if present in historical message
            if msg.get("rag_context"):
                with st.expander("📚 Retrieved Knowledge Context", expanded=False):
                    st.markdown(msg["rag_context"])

# Input prompt
if user_prompt := st.chat_input("Enter a query (e.g., greetings, policy questions, calendar bookings)..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    state = {
        "session_id": "streamlit_session",
        "user_input": user_prompt,
        "tenant_id": "default",
    }

    t0 = time.perf_counter()
    result = event_loop.run_until_complete(runtime.run(state))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    response_text = result.get("response", "No response generated.")
    route_used = result.get("route", "unknown")
    error_raised = result.get("error", None)
    rag_context = result.get("rag_context") or ""

    # Display assistant response
    with st.chat_message("assistant"):
        st.write(response_text)
        st.caption(
            f"⏱️ **Latency:** {elapsed_ms:.1f} ms | "
            f"🧭 **Route:** `{route_used}` | "
            f"⚠️ **Error:** `{error_raised}`"
        )
        # Surface retrieved document context on RAG turns
        if rag_context:
            with st.expander("📚 Retrieved Knowledge Context", expanded=False):
                st.markdown(rag_context)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "latency": elapsed_ms,
            "route": route_used,
            "error": error_raised,
            "rag_context": rag_context,
        }
    )