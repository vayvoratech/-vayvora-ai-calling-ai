# Unified Conversational AI Voice Agent

A unified real-time conversational voice agent supporting dual business domains (**EduSaaS** and **Vayvora**) across inbound inquiries and outbound outreach campaigns.

---

## Python Version Requirement

> [!IMPORTANT]
> **Supported Python Versions:** **Python 3.11** or **Python 3.12** (64-bit).  
> **Python 3.14 is NOT supported** for this project due to dependency compatibility constraints with upstream C-extensions and audio packages planned in later phases.

---

## Project Structure (Phase 1 Baseline)

```
voice/
├── .env.example              # Environment variables template with safe placeholders
├── .gitignore                # Git ignore rules for Python, virtualenv, secrets, and cache
├── ARCHITECTURE.md           # Master System Architecture & Phase Plan (Step 0)
├── pyproject.toml            # Project metadata and Phase 1 dependencies
├── README.md                 # Project documentation & runtime guidance
├── src/
│   ├── __init__.py           # Package marker
│   ├── config.py             # Typed configuration management using pydantic-settings
│   ├── logging.py            # Centralized standardized logging configuration
│   └── core/
│       ├── __init__.py       # Core package exports
│       ├── interfaces.py     # Base abstract provider interfaces (LLM, STT, TTS, VAD, RAG, Tool, Memory)
│       └── types.py          # Foundational Pydantic v2 data models and enums
└── tests/
    ├── __init__.py           # Test package marker
    ├── test_config.py        # Configuration loading and env override tests
    ├── test_interfaces.py    # Abstract provider interface tests
    ├── test_logging.py       # Logging initialization and formatting tests
    └── test_types.py         # Data contract creation, serialization, and validation tests
```

---

## Getting Started (Phase 1)

### 1. Setup Virtual Environment (Python 3.11 / 3.12)

On Windows using the Python Launcher:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Phase 1 Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

### 4. Run Tests
```powershell
pytest
```

---

## Roadmap

- **Phase 1 (Current):** Project foundation, core data contracts, configuration, provider interfaces, and test suite.
- **Phase 2:** Domain Engine & State Management (EduSaaS & Vayvora intent schemas, session memory).
- **Phase 3:** Gemini 3.5 Flash LLM integration & prompt synthesizer.
- **Phase 4:** Redis Stack Grounded RAG implementation.
- **Phase 5:** MCP tool layer & execution verification protocols.
- **Phase 6:** Streamlit testing workbench.
- **Phase 7:** Audio engines (Silero VAD, Faster-Whisper STT, Kokoro/Piper TTS).
- **Phase 8:** Pipecat voice pipeline orchestration and barge-in.
- **Phase 9:** End-to-end inbound & outbound integration.
