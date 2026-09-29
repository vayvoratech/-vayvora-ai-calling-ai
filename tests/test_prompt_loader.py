"""Unit tests for the centralized prompt loader and rendering utilities."""

from pathlib import Path
import pytest

from src.prompts import (
    clear_prompt_cache,
    get_prompts_dir,
    load_prompt,
    render_prompt,
)
from src.prompts.loader import _PROMPT_CACHE


class TestPromptLoader:
    """Test suite for prompt loading, caching, error handling, and template rendering."""

    def setup_method(self) -> None:
        clear_prompt_cache()

    def teardown_method(self) -> None:
        clear_prompt_cache()

    def test_get_prompts_dir_exists(self) -> None:
        prompts_dir = get_prompts_dir()
        assert isinstance(prompts_dir, Path)
        assert prompts_dir.is_dir()
        assert (prompts_dir / "common" / "rules.txt").is_file()

    @pytest.mark.parametrize(
        "rel_path, expected_substring",
        [
            ("common/rules.txt", "CORE BEHAVIORAL RULES"),
            ("agent/json_schema.txt", "OUTPUT FORMAT"),
            ("agent/domain_section.txt", "ACTIVE BUSINESS DOMAIN"),
            ("agent/inbound.txt", "CALL DIRECTION: INBOUND CALL"),
            ("agent/outbound.txt", "CALL DIRECTION: OUTBOUND OUTREACH CALL"),
            ("agent/session_context.txt", "KNOWN SESSION CONTEXT"),
            ("agent/decision_instructions.txt", "DECISION INSTRUCTIONS"),
            ("conversation/outbound_opening.txt", "OUTBOUND phone call"),
            ("conversation/outbound_opening_system.txt", "professional voice agent"),
            ("rag/grounded_answer.txt", "verified reference data"),
            ("rag/system.txt", "verified reference data"),
            ("tools/action_confirmation.txt", "EXECUTED and VERIFIED"),
            ("tools/system.txt", "confirming a verified action"),
        ],
    )
    def test_load_all_registered_prompt_files(
        self, rel_path: str, expected_substring: str
    ) -> None:
        content = load_prompt(rel_path)
        assert isinstance(content, str)
        assert len(content) > 0
        assert expected_substring in content

    def test_load_prompt_missing_file_raises_filenotfound(self) -> None:
        with pytest.raises(FileNotFoundError) as exc_info:
            load_prompt("non_existent_folder/missing_prompt.txt")
        assert "Prompt file not found" in str(exc_info.value)

    def test_prompt_caching(self) -> None:
        rel_path = "rag/system.txt"
        assert rel_path not in _PROMPT_CACHE

        # First load populates cache
        content1 = load_prompt(rel_path, cache=True)
        assert rel_path in _PROMPT_CACHE

        # Second load reads from cache
        content2 = load_prompt(rel_path, cache=True)
        assert content1 == content2

        # Cache bypass
        content_no_cache = load_prompt(rel_path, cache=False)
        assert content_no_cache == content1

        # Clearing cache removes entries
        clear_prompt_cache()
        assert rel_path not in _PROMPT_CACHE

    def test_render_prompt_replaces_placeholders_safely(self) -> None:
        rendered = render_prompt(
            "conversation/outbound_opening_system.txt",
            domain_name="EduSaaS Academy",
        )
        assert "EduSaaS Academy" in rendered
        assert "{domain_name}" not in rendered

    def test_render_prompt_tolerates_literal_json_braces(self) -> None:
        rendered = render_prompt(
            "agent/domain_section.txt",
            name="TestingCorp",
            domain="vayvora",
            description="Leading AI solutions provider.",
            persona_guidelines="Be courteous and prompt.",
            supported_intents="company_overview, career_inquiry",
            supported_slots="caller_name (str: name of caller)",
        )
        assert "TestingCorp" in rendered
        assert "Leading AI solutions provider." in rendered
        assert "{name}" not in rendered
        assert "{domain}" not in rendered
