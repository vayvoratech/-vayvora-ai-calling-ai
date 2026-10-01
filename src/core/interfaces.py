"""Abstract provider interfaces and protocols.

Defines foundational contracts for interchangeable AI, audio, knowledge,
tool, and state components without implementing external services.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional
from src.core.types import (
    CallMetadata,
    DialogueTurn,
    RAGChunk,
    RAGQuery,
    ToolCallRequest,
    ToolExecutionResult,
)


class LLMProvider(ABC):
    """Abstract interface for large language model generation and streaming."""

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a complete text response for the given prompt."""
        pass

    @abstractmethod
    async def stream_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[str]:
        """Stream response tokens asynchronously as they become available."""
        pass


class STTProvider(ABC):
    """Abstract interface for speech-to-text transcription."""

    @abstractmethod
    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Transcribe a complete audio buffer into text."""
        pass

    @abstractmethod
    async def stream_transcribe(
        self, audio_stream: AsyncIterator[bytes], sample_rate: int = 16000
    ) -> AsyncIterator[str]:
        """Transcribe streaming audio frames incrementally into text tokens."""
        pass


class TTSProvider(ABC):
    """Abstract interface for text-to-speech synthesis."""

    @abstractmethod
    async def synthesize(self, text: str, voice_id: Optional[str] = None) -> bytes:
        """Synthesize text into a complete audio frame buffer."""
        pass

    @abstractmethod
    async def stream_synthesize(
        self, text_stream: AsyncIterator[str], voice_id: Optional[str] = None
    ) -> AsyncIterator[bytes]:
        """Synthesize text chunks incrementally into streaming audio chunks."""
        pass

    def cancel_current_synthesis(self) -> None:
        """Cancel ongoing audio synthesis for low-latency barge-in support."""
        pass


class VADProvider(ABC):
    """Abstract interface for voice activity detection and barge-in tracking."""

    @abstractmethod
    def process_frame(self, audio_frame: bytes) -> bool:
        """Process a single PCM frame and return True if human speech is detected."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal speech detector state between turns or after barge-in."""
        pass


class KnowledgeProvider(ABC):
    """Abstract interface for Grounded RAG search and document indexing."""

    @abstractmethod
    async def search(self, query: RAGQuery) -> List[RAGChunk]:
        """Execute a vector search within the specified domain and return relevant chunks."""
        pass

    @abstractmethod
    async def index_document(self, chunk: RAGChunk) -> bool:
        """Index a knowledge chunk into the domain vector store."""
        pass


class ToolProvider(ABC):
    """Abstract interface for discovering and executing external actions via MCP."""

    @abstractmethod
    async def execute_tool(self, request: ToolCallRequest) -> ToolExecutionResult:
        """Dispatch a tool execution request and return verified outcome."""
        pass

    @abstractmethod
    def list_tools(self) -> List[Dict[str, Any]]:
        """List tool definitions and JSON schemas exposed to the LLM."""
        pass


class MemoryProvider(ABC):
    """Abstract interface for session metadata and dialogue turn persistence."""

    @abstractmethod
    async def save_turn(self, call_id: str, turn: DialogueTurn) -> None:
        """Record a conversation turn in session history."""
        pass

    @abstractmethod
    async def get_history(self, call_id: str) -> List[DialogueTurn]:
        """Retrieve the ordered dialogue turns for a given call ID."""
        pass

    @abstractmethod
    async def get_metadata(self, call_id: str) -> Optional[CallMetadata]:
        """Fetch metadata for an active or completed call session."""
        pass

    @abstractmethod
    async def update_metadata(self, metadata: CallMetadata) -> None:
        """Update existing call session metadata."""
        pass
