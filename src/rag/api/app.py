from fastapi import FastAPI

from src.rag.api.routes import router


app = FastAPI(
    title="Vayvora RAG API",
    description="Production RAG retrieval API for Vayvora AI Calling",
    version="1.0.0",
)

app.include_router(router)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "rag",
    }