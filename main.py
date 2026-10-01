"""Unified Vayvora AI Voice Engine and Workbench Application Entrypoint.

Starts the unified FastAPI application providing:
- Web Testing Workbench UI (/ and /static)
- Telephony & Voice Media Streaming (/media-stream and /voice)
- REST Chat & Session APIs (/api/v1/chat, /api/v1/voice, /api/v1/health)
"""

import os
from pathlib import Path
import sys
from dotenv import load_dotenv
import uvicorn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

# Suppress native warning floods
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

from src.ui.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting Vayvora AI Voice Engine on http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=False)
