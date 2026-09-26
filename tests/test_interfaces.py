"""Unit tests for abstract provider interfaces."""

from typing import Any, AsyncIterator, Dict, List, Optional
import pytest

from src.core.interfaces import (
    KnowledgeProvider,
    LLMProvider,
    MemoryProvider,
    STTProvider,
    ToolProvider,
    TTSProvider,
    VADProvider,
)
from src.core.types import (
    CallMetadata,
    DialogueTurn,
    RAGChunk,
    RAGQuery,
    ToolCallRequest,
    ToolExecutionResult,
)


class TestProviderInterfaces:
    """Verify abstract contracts cannot be instantiated without implementation."""

    def test_abstract_classes_raise_type_error(self):
        with pytest.raises(TypeError):
            LLMProvider()

        with pytest.raises(TypeError):
            STTProvider()

        with pytest.raises(TypeError):
            TTSProvider()

        with pytest.raises(TypeError):
            VADProvider()

        with pytest.raises(TypeError):
            KnowledgeProvider()

        with pytest.raises(TypeError):
            ToolProvider()

        with pytest.raises(TypeError):
            MemoryProvider()

    def test_concrete_llm_implementation(self):
        class MockLLM(LLMProvider):
            async def generate_response(
                self,
                prompt: str,
                system_instruction: Optional[str] = None,
                tools: Optional[List[Dict[str, Any]]] = None,
            ) -> str:
                return "Mocked response"

            async def stream_response(
                self,
                prompt: str,
                system_instruction: Optional[str] = None,
                tools: Optional[List[Dict[str, Any]]] = None,
            ) -> AsyncIterator[str]:
                yield "Mocked stream"

        mock = MockLLM()
        assert isinstance(mock, LLMProvider)

    def test_concrete_vad_implementation(self):
        class MockVAD(VADProvider):
            def process_frame(self, audio_frame: bytes) -> bool:
                return True

            def reset(self) -> None:
                pass

        vad = MockVAD()
        assert isinstance(vad, VADProvider)
        assert vad.process_frame(b"\x00\x01") is True

    def test_concrete_tool_provider(self):
        class MockToolProvider(ToolProvider):
            async def execute_tool(
                self, request: ToolCallRequest
            ) -> ToolExecutionResult:
                return ToolExecutionResult(
                    tool_name=request.tool_name,
                    success=True,
                )

            def list_tools(self) -> List[Dict[str, Any]]:
                return [{"name": "mock_tool"}]

        tool_provider = MockToolProvider()
        assert isinstance(tool_provider, ToolProvider)
        assert len(tool_provider.list_tools()) == 1
