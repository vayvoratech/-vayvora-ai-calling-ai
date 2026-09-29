"""Centralized prompt loader and template rendering utility.

Loads externalized prompt templates from the project's prompts directory with
safe placeholder replacement, UTF-8 encoding, and in-memory caching.
"""

from pathlib import Path
from typing import Any, Dict, Union

# Resolves to project root / "prompts"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"

_PROMPT_CACHE: Dict[str, str] = {}


def get_prompts_dir() -> Path:
    """Return the absolute path to the prompts directory.
    
    Falls back gracefully if the package structure is altered.
    """
    if PROMPTS_DIR.is_dir():
        return PROMPTS_DIR
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        candidate = parent / "prompts"
        if candidate.is_dir():
            return candidate
    return PROMPTS_DIR


def clear_prompt_cache() -> None:
    """Clear the in-memory cache of loaded prompts."""
    _PROMPT_CACHE.clear()


def load_prompt(relative_path: Union[str, Path], cache: bool = True) -> str:
    """Load a prompt file from the prompts directory.

    Args:
        relative_path: Path relative to the prompts directory (e.g. 'common/rules.txt').
        cache: Whether to cache the file contents in memory.

    Returns:
        The content of the prompt file as a string.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    rel_path_str = str(relative_path).replace("\\", "/")
    if cache and rel_path_str in _PROMPT_CACHE:
        return _PROMPT_CACHE[rel_path_str]

    prompts_dir = get_prompts_dir()
    file_path = prompts_dir / relative_path

    if not file_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    if cache:
        _PROMPT_CACHE[rel_path_str] = content

    return content


def render_prompt(
    relative_path: Union[str, Path],
    cache: bool = True,
    **kwargs: Any,
) -> str:
    """Load and render a prompt template with safe placeholder replacement.

    Replaces {key} placeholders with str(value). Avoids str.format() syntax
    errors when prompts contain literal JSON curly braces.

    Args:
        relative_path: Path relative to prompts directory.
        cache: Whether to cache the raw template.
        **kwargs: Key-value pairs to substitute into {key} placeholders.

    Returns:
        Rendered prompt string.
    """
    content = load_prompt(relative_path, cache=cache)
    for k, v in kwargs.items():
        content = content.replace(f"{{{k}}}", str(v))
    return content
