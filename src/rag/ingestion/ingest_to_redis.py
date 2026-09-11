from __future__ import annotations

from src.rag.embeddings.provider import get_embedding_provider
from src.rag.ingestion.pipeline import IngestionPipeline
from src.rag.retrieval.vector_store import RAGVectorStore


def main() -> None:
    tenant_id = "default"

    print("=" * 70)
    print("VAYVORA RAG - REDIS INGESTION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load and chunk knowledge base
    # ---------------------------------------------------------

    pipeline = IngestionPipeline()

    chunks = pipeline.run(
        tenant_id=tenant_id
    )

    print(f"Tenant: {tenant_id}")
    print(f"Chunks prepared: {len(chunks)}")

    if not chunks:
        raise RuntimeError(
            "No knowledge-base chunks were generated."
        )

    # ---------------------------------------------------------
    # 2. Create embeddings
    # ---------------------------------------------------------

    print("\nGenerating embeddings...")

    embedding_provider = (
        get_embedding_provider()
    )

    texts = [
        chunk.text
        for chunk in chunks
    ]

    vectors = embedding_provider.embed_documents(
        texts
    )

    print(
        f"Embeddings generated: {len(vectors)}"
    )

    # ---------------------------------------------------------
    # 3. Connect to Redis vector store
    # ---------------------------------------------------------

    vector_store = RAGVectorStore()

    vector_store.create_index()

    # ---------------------------------------------------------
    # 4. Clear existing RAG index
    #
    # IMPORTANT:
    # This rebuilds the default knowledge base
    # from scratch using the new chunks.
    # ---------------------------------------------------------

    print("\nClearing existing RAG index...")

    deleted = vector_store.delete_all()

    print(
        f"Deleted vector index: "
        f"{deleted['vector_set_deleted']}"
    )

    print(
        f"Deleted metadata keys: "
        f"{deleted['metadata_deleted']}"
    )

    # ---------------------------------------------------------
    # 5. Insert new chunks
    # ---------------------------------------------------------

    print("\nInserting chunks into Redis...")

    inserted = vector_store.add_chunks(
        chunks=chunks,
        vectors=vectors,
    )

    print(
        f"Chunks inserted: {inserted}"
    )

    # ---------------------------------------------------------
    # 6. Verify Redis index
    # ---------------------------------------------------------

    print("\nVerifying Redis index...")

    from src.memory.redis_client import redis_client
    from src.rag.config import rag_config

    info = redis_client.execute_command(
        "VINFO",
        rag_config.redis_index_name,
    )

    print(f"Redis vector index: {rag_config.redis_index_name}")
    print(f"Redis index info: {info}")

    print("\n" + "=" * 70)
    print("RAG INGESTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()