"""Markdown document loader with strict tenant/domain isolation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag.config import rag_config


class MarkdownDocumentLoader:
    """Load Markdown/text documents only from the requested tenant directory.

    For Infra Builder ingestion, documents must live under:
        <knowledge_base_dir>/infra_builder/

    The loader deliberately does not fall back to scanning the shared knowledge
    root. This prevents root-level or other-domain documents from being
    accidentally indexed as Infra Builder knowledge.
    """

    ALLOWED_SUFFIXES = {".md", ".txt"}
    SHARED_TENANTS = {"default", "all"}

    def __init__(self, base_dir: Optional[str | Path] = None) -> None:
        self.base_dir = Path(base_dir or rag_config.knowledge_base_dir).expanduser()

    def load_documents(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load documents for one tenant.

        Args:
            tenant_id: Tenant/domain folder name, e.g. ``infra_builder``.
                ``None``, ``default`` and ``all`` scan only the explicitly
                supplied base directory; they do not infer a tenant for files.

        Returns:
            Document dictionaries containing source, category, tenant_id and text.

        Raises:
            FileNotFoundError: If the requested tenant directory does not exist.
            NotADirectoryError: If the configured base path is not a directory.
            ValueError: If tenant_id is an unsafe path component.
        """
        if not self.base_dir.exists():
            raise FileNotFoundError(
                f"Knowledge base root does not exist: {self.base_dir}"
            )
        if not self.base_dir.is_dir():
            raise NotADirectoryError(
                f"Knowledge base root is not a directory: {self.base_dir}"
            )

        if tenant_id and tenant_id not in self.SHARED_TENANTS:
            tenant_component = Path(tenant_id)
            if (
                tenant_component.is_absolute()
                or len(tenant_component.parts) != 1
                or tenant_id in {".", ".."}
                or "/" in tenant_id
                or "\\" in tenant_id
            ):
                raise ValueError(f"Invalid tenant_id path component: {tenant_id!r}")

            search_dir = self.base_dir / tenant_id
            if not search_dir.exists():
                raise FileNotFoundError(
                    f"Knowledge directory for tenant '{tenant_id}' does not exist: "
                    f"{search_dir}. Create this directory and place that tenant's "
                    "approved .md/.txt files inside it."
                )
            if not search_dir.is_dir():
                raise NotADirectoryError(
                    f"Knowledge path for tenant '{tenant_id}' is not a directory: "
                    f"{search_dir}"
                )
            tenant_filter = tenant_id
        else:
            # Shared/root scans are explicit only. Do not label root files as
            # Infra Builder (or any other tenant) implicitly.
            search_dir = self.base_dir
            tenant_filter = None

        documents: List[Dict[str, Any]] = []
        for path in sorted(search_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.ALLOWED_SUFFIXES:
                continue

            # Ignore files within hidden directories and hidden files.
            try:
                relative_to_search = path.relative_to(search_dir)
            except ValueError:
                continue
            if path.name.startswith(".") or any(
                part.startswith(".") for part in relative_to_search.parts
            ):
                continue

            try:
                text = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                # Skip unreadable documents; continue loading the rest.
                continue
            if not text:
                continue

            relative_to_base = path.relative_to(self.base_dir).as_posix()
            parts = relative_to_base.split("/")

            if tenant_filter:
                # Because search_dir is the tenant directory, every loaded file
                # is explicitly scoped to that tenant.
                doc_tenant = tenant_filter
                relative_parts = relative_to_search.parts
                category = (
                    relative_parts[0]
                    if len(relative_parts) > 1
                    else "general"
                )
            else:
                # Root/shared mode must not infer a business tenant.
                doc_tenant = "default"
                category = (
                    path.parent.name
                    if path.parent != self.base_dir
                    else "general"
                )

            doc_id = path.stem.lower().replace(" ", "_")
            documents.append(
                {
                    "document_id": doc_id,
                    "source": relative_to_base,
                    "category": category,
                    "tenant_id": doc_tenant,
                    "text": text,
                    "file_path": str(path),
                }
            )

        return documents
