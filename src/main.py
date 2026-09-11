from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.voice.voice import router as voice_router

# Locate the folder containing this file (C:\AI VOICE CALLING\src)
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

src = FastAPI()


# ============================================================
# FRONTEND
# ============================================================

src.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static"
)


@src.get("/")
async def home():
    return FileResponse(
        FRONTEND_DIR / "index.html"
    )


# ============================================================
# PCM AUDIO WORKLET
# ============================================================

@src.get("/pcm-player.js")
async def pcm_player():
    return FileResponse(
        FRONTEND_DIR / "pcm-player.js",
        media_type="application/javascript"
    )


# ============================================================
# VOICE ROUTES
# ============================================================

src.include_router(
    voice_router
)