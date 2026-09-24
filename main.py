import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

# Suppress native warning floods
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

from api.dependencies import get_agent_runtime
from api.routes import chat, health, voice

app = FastAPI(
    title="Vayvora AI Voice Engine",
    description="Real-Time STT, AgentRuntime, and TTS Microservice",
    version="1.0.0",
)

# CORS configuration for Java Backend & Frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Endpoints
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(voice.router)

# Mount frontend directory and root index.html handler
FRONTEND_DIR = PROJECT_ROOT / "src" / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    # 1. Root route to serve index.html directly at http://localhost:8000/
    @app.get("/", response_class=FileResponse)
    async def serve_root():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse("<h3>index.html not found in src/frontend</h3>", status_code=404)

    # 2. Direct route for pcm-player.js if referenced at the root level
    @app.get("/pcm-player.js")
    async def serve_pcm_player():
        player_script = FRONTEND_DIR / "pcm-player.js"
        if player_script.exists():
            return FileResponse(player_script, media_type="application/javascript")
        return HTMLResponse("Not found", status_code=404)


@app.on_event("startup")
async def on_startup():
    print("Starting AgentRuntime...")
    agent = get_agent_runtime()
    await agent.start()
    print("AgentRuntime active and ready for inference.")


@app.on_event("shutdown")
async def on_shutdown():
    agent = get_agent_runtime()
    if agent:
        await agent.close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)