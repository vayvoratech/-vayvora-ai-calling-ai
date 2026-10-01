"""Speech-to-Text (STT) service using Groq Whisper API based on Kiran voice implementation."""

from __future__ import annotations

import io
import os
import time
from typing import Any, AsyncIterator, Optional, Union
import wave

from src.core.interfaces import STTProvider
from src.logging import get_logger

logger = get_logger("voice.stt")


class GroqWhisperSTT(STTProvider):
    """Groq Whisper STT Service using cloud whisper-large-v3-turbo."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-large-v3-turbo",
        sample_rate: int = 16000,
        language: str = "en",
        prompt: str = "Vayvora Technology, AI, engineering services, consultation, appointment",
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
        self.sample_rate = sample_rate
        self.language = language
        self.prompt = prompt

        self.client = None
        self._fallback_provider = None

        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                logger.info(
                    "Initialized Groq Whisper STT (model: %s, sample_rate: %dHz, language: %s)",
                    self.model,
                    self.sample_rate,
                    self.language,
                )
            except Exception as e:
                logger.warning("Failed to initialize Groq client: %s", e)
        else:
            logger.info("GROQ_API_KEY not set; using mock STT fallback.")

    def _get_fallback_provider(self):
        """Lazy-load deterministic mock STT fallback provider when API key is unavailable."""
        if self._fallback_provider is None:
            try:
                from src.audio.stt import MockSTTProvider
                self._fallback_provider = MockSTTProvider(language=self.language)
            except Exception as e:
                logger.warning("Fallback STT provider init failed: %s", e)
        return self._fallback_provider

    def _filter_whisper_hallucination(self, text: str, pcm_audio: bytes) -> str:
        """Filter out common Whisper hallucinations generated on background noise or silence."""
        if not text:
            return ""

        hallucination_phrases = {
            "thank you.", "thank you", "thank you very much.", "thanks.", "thanks",
            "thanks for watching!", "thanks for watching.", "thank you for watching.",
            "you", "yeah.", "yeah", "bye.", "bye", "goodbye.", "goodbye",
            "[silence]", "[music]", "[applause]", "[laughter]",
            "mbc 뉴스", "시청해 주셔서 감사합니다.",
            ".", "..", "...", ",", "?", "!",
        }

        normalized = text.strip().lower()
        if normalized in hallucination_phrases:
            # Check duration and energy
            duration_ms = (len(pcm_audio) / 2 / self.sample_rate) * 1000.0
            if len(pcm_audio) >= 2:
                import numpy as np
                samples = np.frombuffer(pcm_audio, dtype=np.int16)
                rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
            else:
                rms = 0.0

            # If audio duration is short (< 1000ms) or energy is low (< 350), reject hallucination
            if duration_ms < 1000.0 or rms < 350.0:
                logger.info(
                    "Whisper hallucination rejected: '%s' (duration=%.1fms, rms=%.1f)",
                    text, duration_ms, rms
                )
                return ""

        return text

    def pcm16_to_wav_bytes(self, pcm_data: bytes) -> io.BytesIO:
        """Encapsulate raw PCM16 bytes into an in-memory WAV container."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)        # Mono
            wav_file.setsampwidth(2)       # 16-bit (2 bytes per sample)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm_data)

        wav_buffer.seek(0)
        wav_buffer.name = "audio.wav"
        return wav_buffer

    async def transcribe(
        self,
        audio_data: Union[bytes, Any],
        sample_rate: int = 16000,
    ) -> str:
        """Asynchronous transcription interface conforming to STTProvider."""
        if audio_data is None:
            return ""

        # Extract bytes from AudioChunk or SpeechSegment if passed
        if hasattr(audio_data, "data"):
            pcm_bytes = audio_data.data
        elif hasattr(audio_data, "audio_data"):
            pcm_bytes = audio_data.audio_data
        elif isinstance(audio_data, (bytes, bytearray)):
            pcm_bytes = bytes(audio_data)
        else:
            return ""

        return self.transcribe_pcm16(pcm_bytes)

    async def stream_transcribe(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
    ) -> AsyncIterator[str]:
        """Incremental transcription for streaming audio frames."""
        accumulated = bytearray()
        async for chunk in audio_stream:
            accumulated.extend(chunk)
            if len(accumulated) >= sample_rate * 2:  # 1 second of audio
                text = self.transcribe_pcm16(bytes(accumulated))
                if text:
                    yield text
                accumulated.clear()

        if accumulated:
            text = self.transcribe_pcm16(bytes(accumulated))
            if text:
                yield text

    def transcribe_pcm16(self, pcm_audio: bytes) -> str:
        """Transcribe PCM16 audio bytes to text via Groq Whisper API."""
        if not pcm_audio or len(pcm_audio) < 100:
            return ""

        # 1. Groq Whisper API Call
        if self.client is not None:
            try:
                t0 = time.perf_counter()
                wav_file = self.pcm16_to_wav_bytes(pcm_audio)
                transcription = self.client.audio.transcriptions.create(
                    file=wav_file,
                    model=self.model,
                    language=self.language,
                    prompt=self.prompt,  # Guides domain terminology (Vayvora, EduSaaS, etc.)
                    temperature=0.0,
                    response_format="text",
                )
                text = str(transcription).strip()
                elapsed = time.perf_counter() - t0

                # Whisper Hallucination Guard on low-energy / silence artifacts
                filtered_text = self._filter_whisper_hallucination(text, pcm_audio)
                if not filtered_text:
                    logger.info("Filtered Whisper hallucination on noise/silence (%d bytes): '%s'", len(pcm_audio), text)
                    return ""

                logger.info(
                    "Groq Whisper transcribed (%d bytes in %.3fs): '%s'",
                    len(pcm_audio),
                    elapsed,
                    filtered_text,
                )
                return filtered_text
            except Exception as e:
                logger.warning(
                    "Groq STT Error: %s %s. Returning structured empty transcript.",
                    type(e).__name__,
                    e,
                )
                return ""

        # 2. Mock STT Fallback (when GROQ_API_KEY is not set)
        fallback = self._get_fallback_provider()
        if fallback is not None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        res = pool.submit(
                            asyncio.run,
                            fallback.transcribe(pcm_audio, sample_rate=self.sample_rate),
                        ).result()
                else:
                    res = asyncio.run(fallback.transcribe(pcm_audio, sample_rate=self.sample_rate))
                return getattr(res, "text", str(res)).strip()
            except Exception as e:
                logger.warning("Mock STT fallback failed: %s", e)

        return ""


# Canonical Alias for backward compatibility with Kiran and Vayvora codebase
STTService = GroqWhisperSTT
