from __future__ import annotations

from src.rag.embeddings.provider import get_embedding_provider
from src.rag.ingestion.incremental import (
    IncrementalIngestionService,
)
from src.rag.retrieval.vector_store import RAGVectorStore


def main() -> None:
    tenant_id = "default"

    print("=" * 70)
    print("VAYVORA RAG - INCREMENTAL REDIS INGESTION")
    print("=" * 70)

    service = IncrementalIngestionService(
        tenant_id=tenant_id
    )

    vector_store = RAGVectorStore()
    embedding_provider = get_embedding_provider()

    # ---------------------------------------------------------
    # 1. Detect document changes
    # ---------------------------------------------------------

    print("\nChecking knowledge base...")

    plan = service.prepare()

    changed_documents = plan[
        "changed_documents"
    ]

    unchanged_documents = plan[
        "unchanged_documents"
    ]

    deleted_document_ids = plan[
        "deleted_document_ids"
    ]

    print(
        f"Changed documents: "
        f"{len(changed_documents)}"
    )

    print(
        f"Unchanged documents: "
        f"{len(unchanged_documents)}"
    )

    print(
        f"Deleted documents: "
        f"{len(deleted_document_ids)}"
    )

    # ---------------------------------------------------------
    # 2. Delete removed documents
    # ---------------------------------------------------------

    for document_id in deleted_document_ids:

        print(
            f"\nDeleting removed document: "
            f"{document_id}"
        )

        deleted = vector_store.delete_document(
            document_id=document_id,
            tenant_id=tenant_id,
        )

        service.manifest.delete(
            document_id
        )

        print(
            f"Deleted chunks: {deleted}"
        )

    # ---------------------------------------------------------
    # 3. Process changed/new documents
    # ---------------------------------------------------------

    for item in changed_documents:

        document = item[
            "document"
        ]

        content_hash = item[
            "content_hash"
        ]

        chunks = item[
            "chunks"
        ]

        document_id = document[
            "document_id"
        ]

        print(
            f"\nProcessing: "
            f"{document['source']}"
        )

        # Remove previous version first.
        old_deleted = (
            vector_store.delete_document(
                document_id=document_id,
                tenant_id=tenant_id,
            )
        )

        if old_deleted:
            print(
                f"Removed old chunks: "
                f"{old_deleted}"
            )

        if not chunks:
            print(
                "No chunks generated. "
                "Skipping document."
            )
            continue

        # -----------------------------------------------------
        # Generate embeddings
        # -----------------------------------------------------

        texts = [
            chunk.text
            for chunk in chunks
        ]

        vectors = (
            embedding_provider.embed_documents(
                texts
            )
        )

        # -----------------------------------------------------
        # Insert new version
        # -----------------------------------------------------

        inserted = (
            vector_store.add_chunks(
                chunks=chunks,
                vectors=vectors,
            )
        )

        print(
            f"New chunks: {len(chunks)}"
        )

        print(
            f"Inserted chunks: {inserted}"
        )

        # -----------------------------------------------------
        # Update manifest ONLY after successful insertion
        # -----------------------------------------------------

        service.update_manifest(
            document=document,
            content_hash=content_hash,
            chunk_count=len(chunks),
            version="1",
        )

        print(
            "Manifest updated."
        )

    # ---------------------------------------------------------
    # 4. Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("INCREMENTAL INGESTION COMPLETED")
    print("=" * 70)

    print(
        f"Changed: {len(changed_documents)}"
    )

    print(
        f"Unchanged: {len(unchanged_documents)}"
    )

    print(
        f"Deleted: {len(deleted_document_ids)}"
    )


if __name__ == "__main__":
    main()