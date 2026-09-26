"""Dependency injection and bootstrap factory for the Streamlit testing workbench.

Detects runtime environment and constructs live or mock service components accordingly,
clearly tagging runtime modes so mock actions are never mistaken for real external mutations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from src.core.llm import GeminiLLMProvider, MockLLMProvider
from src.config import Settings, get_settings
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine
from src.core.interfaces import KnowledgeProvider, LLMProvider, ToolProvider

from src.core.types import ConversationStage, DomainType
from src.logging import get_logger
from src.rag.embeddings import MockEmbeddingProvider, get_embedding_provider
from src.rag.ingestion import KnowledgeIngestionPipeline
from src.rag.redis_client import RedisVectorStore
from src.rag.retriever import GroundedKnowledgeProvider
from src.state.manager import ConversationStateManager
from src.tools import (
    EmailProvider,
    HttpMCPToolProvider,
    MockEmailProvider,
    MockToolProvider,
    SMTPEmailProvider,
)
from src.ui.service import RuntimeMode, ServiceComponents, WorkbenchService, run_sync

logger = get_logger("ui.bootstrap")


class InteractiveMockLLMProvider(MockLLMProvider):
    """Context-aware mock LLM provider enabling realistic text interactions during workbench testing."""

    def __init__(
        self,
        canned_responses: Optional[List[str]] = None,
        canned_decisions: Optional[List[ConversationalDecision]] = None,
    ) -> None:
        super().__init__(canned_responses=canned_responses, canned_decisions=canned_decisions)

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if "initiating an OUTBOUND phone call" in prompt:
            if "EduSaaS" in prompt:
                return "Hi, this is the EduSaaS team. I'm reaching out because you had shown interest in our courses. Is this a good time to speak for a couple of minutes?"
            else:
                return "Hi, this is the Vayvora team. I'm reaching out to understand whether your organization is currently exploring AI or automation solutions. Is this a good time for a quick conversation?"
        return await super().generate_response(prompt, system_instruction, tools)

    async def generate_decision(
        self,
        prompt: str,
        system_instruction: str,
        max_retries: int = 1,
    ) -> ConversationalDecision:
        self.invocations.append({
            "prompt": prompt,
            "system_instruction": system_instruction,
        })

        # 1. Respect explicit canned decisions first
        if self.canned_decisions:
            decision = self.canned_decisions.pop(0)
            return self.validator.validate_and_filter(decision)

        # 2. Extract caller's latest utterance from prompt
        user_text = ""
        if "=== LATEST CALLER MESSAGE" in prompt:
            chunk = prompt.split("=== LATEST CALLER MESSAGE")[1]
            if "===" in chunk:
                chunk = chunk.split("===")[1]
            if "Analyze the latest" in chunk:
                chunk = chunk.split("Analyze the latest")[0]
            user_text = chunk.strip().strip('"').strip().lower()
        elif "Latest Caller Message:" in prompt:
            chunk = prompt.split("Latest Caller Message:")[-1]
            if "Analyze the latest" in chunk:
                chunk = chunk.split("Analyze the latest")[0]
            user_text = chunk.strip().strip('"').strip().lower()
        elif "Caller said:" in prompt:
            chunk = prompt.split("Caller said:")[-1]
            user_text = chunk.strip().strip('"').strip().lower()
        else:
            prompt_lines = [l for l in prompt.strip().splitlines() if l.strip()]
            user_text = prompt_lines[-1].strip().lower() if prompt_lines else ""

        clean_user_text = user_text.rstrip(".!?")

        # 3. Contextual Heuristic Decision Generation
        # (A) Farewells / Explicit Terminations
        if any(w in clean_user_text for w in ["bye", "goodbye", "that's all", "thank you, that's all", "end call"]):
            decision = ConversationalDecision(
                detected_domain=DomainType.GENERAL,
                detected_intent="farewell",
                proposed_stage=ConversationStage.COMPLETED,
                user_facing_response="Thank you for speaking with us today! Have a wonderful day. Goodbye!",
            )
            return self.validator.validate_and_filter(decision)

        # (B) Declines / Non-terminating Negatives
        if clean_user_text in ["no", "no thanks", "no thank you", "not really", "nope", "nah"]:
            decision = ConversationalDecision(
                detected_domain=DomainType.GENERAL,
                detected_intent="decline",
                proposed_stage=ConversationStage.INFORMATION,
                user_facing_response="Understood! Is there anything else about EduSaaS courses or Vayvora solutions I can assist you with?",
            )
            return self.validator.validate_and_filter(decision)

        # (C) Scheduling / Meeting Consultations
        if any(w in clean_user_text for w in ["schedule", "meeting", "consultation", "demo", "book a slot", "calendar"]):
            decision = ConversationalDecision(
                detected_domain=DomainType.VAYVORA if "vayvora" in clean_user_text else DomainType.EDUSAAS,
                detected_intent="schedule_consultation",
                proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                action_proposed=True,
                proposed_action=ProposedAction(
                    tool_name="create_calendar_event",
                    arguments={"slot": "Tomorrow 10:00 AM", "confirmed": True},
                ),
                user_facing_response="I would be happy to schedule a consultation for you. I have confirmed Tomorrow at 10:00 AM.",
            )
            return self.validator.validate_and_filter(decision)

        # (D) Brochure / Email Information
        if any(w in clean_user_text for w in ["email", "brochure", "send me", "syllabus pdf", "mail me"]):
            decision = ConversationalDecision(
                detected_domain=DomainType.EDUSAAS if "course" in clean_user_text or "edusaas" in clean_user_text else DomainType.VAYVORA,
                detected_intent="request_brochure",
                proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                action_proposed=True,
                proposed_action=ProposedAction(
                    tool_name="send_email",
                    arguments={"subject": "Detailed Information & Curriculum"},
                ),
                user_facing_response="I will send over the detailed syllabus and documentation right away.",
            )
            return self.validator.validate_and_filter(decision)

        # (E) Careers / HR Followups
        if any(w in clean_user_text for w in ["career", "careers", "job", "hiring", "apply", "internship"]):
            decision = ConversationalDecision(
                detected_domain=DomainType.VAYVORA,
                detected_intent="career_inquiry",
                proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                action_proposed=True,
                proposed_action=ProposedAction(
                    tool_name="create_hr_followup",
                    arguments={"candidate_name": "Applicant", "position": "Software Engineer"},
                ),
                user_facing_response="We are always looking for talented engineers at Vayvora. I have queued a follow-up ticket with our HR team.",
            )
            return self.validator.validate_and_filter(decision)

        # (F) Vayvora Enterprise & Corporate Solutions
        if any(w in clean_user_text for w in ["vayvora", "solution", "solutions", "enterprise", "consulting", "software", "custom ai"]):
            decision = ConversationalDecision(
                detected_domain=DomainType.VAYVORA,
                detected_intent="solutions_inquiry",
                proposed_stage=ConversationStage.INFORMATION,
                knowledge_required=True,
                knowledge_query=clean_user_text,
                user_facing_response="Vayvora provides enterprise AI solutions, autonomous agent workflows, and intelligent software engineering.",
            )
            return self.validator.validate_and_filter(decision)

        # (G) EduSaaS Courses & Curriculum
        if any(w in clean_user_text for w in ["course", "courses", "syllabus", "learn", "enroll", "curriculum", "ai engineering"]):
            decision = ConversationalDecision(
                detected_domain=DomainType.EDUSAAS,
                detected_intent="course_information",
                proposed_stage=ConversationStage.INFORMATION,
                knowledge_required=True,
                knowledge_query=clean_user_text,
                user_facing_response="At EduSaaS, we offer comprehensive programs in AI Engineering, Data Science, and Machine Learning.",
            )
            return self.validator.validate_and_filter(decision)

        # (H) Affirmative confirmations / continuations ("yes", "sure", "okay", "yeah", "tell me more")
        if any(w in clean_user_text for w in ["yes", "yeah", "sure", "okay", "ok", "yep", "certainly", "tell me more", "sounds good", "interested"]):
            detected_dom = DomainType.EDUSAAS if "edusaas" in prompt.lower() else DomainType.VAYVORA
            decision = ConversationalDecision(
                detected_domain=detected_dom,
                detected_intent="confirm_interest",
                proposed_stage=ConversationStage.DISCOVERY,
                user_facing_response="Great! To better assist you, could you share more about your specific requirements or interests?",
            )
            return self.validator.validate_and_filter(decision)

        # (I) Pure Greetings & Initial Openings
        if any(w in clean_user_text for w in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            detected_dom = DomainType.EDUSAAS if "edusaas" in prompt.lower() else DomainType.VAYVORA
            decision = ConversationalDecision(
                detected_domain=detected_dom,
                detected_intent="greeting",
                proposed_stage=ConversationStage.GREETING,
                user_facing_response="Hi, how can I help you today?",
            )
            return self.validator.validate_and_filter(decision)

        # (J) Default Fallback (Conversational continuity, NEVER reset to greeting)
        detected_dom = DomainType.EDUSAAS if "edusaas" in prompt.lower() else DomainType.VAYVORA
        decision = ConversationalDecision(
            detected_domain=detected_dom,
            detected_intent="general_inquiry",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response=f"Understood. Could you share more details about your specific questions or requirements regarding {detected_dom.value}?",
        )
        return self.validator.validate_and_filter(decision)


def is_redis_available(settings: Settings) -> bool:
    """Check whether a live Redis Stack instance is accessible."""
    try:
        store = RedisVectorStore(settings=settings)
        reachable = run_sync(store.health_check())
        run_sync(store.close())
        return reachable
    except Exception:
        return False


def is_mcp_available(url: str) -> bool:
    """Check whether an external MCP server endpoint is reachable."""
    if not url or "localhost" not in url and "127.0.0.1" not in url:
        return False
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(url)
            return resp.status_code in (200, 404, 405)
    except Exception:
        return False


def bootstrap_workbench(
    settings: Optional[Settings] = None,
    force_mock: bool = False,
    live_llm: bool = False,
    live_rag: bool = False,
    live_tools: bool = False,
) -> WorkbenchService:
    """Instantiate and configure the WorkbenchService with detected or mock subsystems."""
    cfg = settings or get_settings()
    notes: List[str] = []

    # -------------------------------------------------------------------------
    # 1. LLM Provider
    # -------------------------------------------------------------------------
    llm_mode = "MOCK"

    gemini_key_str = ""
    if cfg.gemini_api_key:
        gemini_key_str = (
            cfg.gemini_api_key.get_secret_value()
            if hasattr(cfg.gemini_api_key, "get_secret_value")
            else str(cfg.gemini_api_key)
        )

    can_use_live_gemini = (
        not force_mock
        and cfg.llm_provider.lower() == "gemini"
        and (
            live_llm
            or bool(
                gemini_key_str
                and len(gemini_key_str.strip()) > 10
                and not gemini_key_str.strip().startswith("mock")
            )
        )
    )

    if can_use_live_gemini:
        try:
            llm_provider: LLMProvider = GeminiLLMProvider(
                settings=cfg
            )
            llm_mode = "LIVE"
            notes.append(
                f"LLM: Live Gemini provider active ({cfg.gemini_model})"
            )
        except Exception as exc:
            logger.error("Gemini initialization failed: %s", exc)
            if live_llm:
                raise RuntimeError(f"Gemini was explicitly requested as LIVE but failed to initialize: {exc}") from exc
            llm_provider = InteractiveMockLLMProvider()
            llm_mode = "MOCK"
            notes.append(
                f"LLM: Mock fallback (Gemini initialization failed: {exc})"
            )
    else:
        llm_provider = InteractiveMockLLMProvider()
        notes.append(
            "LLM: Interactive Mock provider active"
        )

    # -------------------------------------------------------------------------
    # 2. Grounded Knowledge Provider (RAG)
    # -------------------------------------------------------------------------
    rag_mode = "MOCK"
    can_use_live_redis = not force_mock and (live_rag or is_redis_available(cfg))

    if can_use_live_redis:
        try:
            store = RedisVectorStore(settings=cfg)
            knowledge_provider: Optional[KnowledgeProvider] = GroundedKnowledgeProvider(
                vector_store=store,
                embedding_provider=get_embedding_provider(settings=cfg, force_mock=False),
                in_memory=False,
            )
            rag_mode = "LIVE"
            notes.append("RAG: Live Redis Stack vector store connected")
        except Exception as exc:
            logger.warning("Falling back to in-memory RAG: %s", exc)
            knowledge_provider = GroundedKnowledgeProvider(
                embedding_provider=get_embedding_provider(settings=cfg, force_mock=force_mock),
                in_memory=True,
            )
            rag_mode = "IN_MEMORY"
            notes.append(f"RAG: In-memory fallback ({exc})")
    else:
        # High-fidelity in-memory provider seeded with local knowledge documents
        knowledge_provider = GroundedKnowledgeProvider(
            embedding_provider=get_embedding_provider(settings=cfg, force_mock=force_mock),
            in_memory=True,
        )
        rag_mode = "MOCK" if force_mock else "IN_MEMORY"
        try:
            pipeline = KnowledgeIngestionPipeline(knowledge_provider=knowledge_provider)
            run_sync(pipeline.ingest_domain(DomainType.EDUSAAS))
            run_sync(pipeline.ingest_domain(DomainType.VAYVORA))
            notes.append("RAG: In-memory vector store seeded with EduSaaS & Vayvora knowledge")
        except Exception as ingest_err:
            logger.warning("Could not pre-seed in-memory knowledge: %s", ingest_err)
            notes.append(f"RAG: In-memory store unseeded ({ingest_err})")

    # -------------------------------------------------------------------------
    # 3. Tool Provider (MCP) & Email Provider (SMTP)
    # -------------------------------------------------------------------------
    email_provider: Optional[EmailProvider] = None
    if cfg.smtp_host and cfg.smtp_host != "localhost" and cfg.smtp_username:
        email_provider = SMTPEmailProvider(settings=cfg)
        notes.append(f"Email: SMTPEmailProvider active ({cfg.smtp_host}:{cfg.smtp_port})")
    elif not force_mock and cfg.smtp_host and cfg.smtp_host != "localhost":
        email_provider = SMTPEmailProvider(settings=cfg)
        notes.append(f"Email: SMTPEmailProvider active ({cfg.smtp_host}:{cfg.smtp_port})")
    else:
        email_provider = MockEmailProvider(force_success=True)
        notes.append("Email: MockEmailProvider active")

    tool_mode = "MOCK"
    can_use_live_mcp = not force_mock and (live_tools or is_mcp_available(cfg.mcp_server_url))

    if can_use_live_mcp:
        try:
            tool_provider: Optional[ToolProvider] = HttpMCPToolProvider(settings=cfg, email_provider=email_provider)
            tool_mode = "LIVE"
            notes.append(f"Tools: Live HTTP MCP provider connected to {cfg.mcp_server_url}")
        except Exception as exc:
            logger.warning("Falling back to MockToolProvider: %s", exc)
            tool_provider = MockToolProvider(settings=cfg, email_provider=email_provider)
            notes.append(f"Tools: Mock fallback ({exc})")
    else:
        tool_provider = MockToolProvider(settings=cfg, email_provider=email_provider)
        notes.append("Tools: Mock Tool Provider active (idempotent, verified mock actions)")

    # -------------------------------------------------------------------------
    # 4. Audio Subsystems (VAD, STT, TTS)
    # -------------------------------------------------------------------------
    # VAD
    vad_mode = "MOCK"
    vad_provider = None
    if not force_mock and getattr(cfg, "vad_provider", "").lower() == "silero":
        try:
            from src.audio.vad import SileroVADProvider

            candidate_vad = SileroVADProvider(
                sample_rate=cfg.vad_sample_rate,
                model_path=cfg.vad_model_path,
                threshold=cfg.vad_threshold,
                min_speech_duration=cfg.vad_min_speech_duration,
                min_silence_duration=cfg.vad_min_silence_duration,
                settings=cfg,
            )
            if candidate_vad._session is not None:
                vad_provider = candidate_vad
                vad_mode = "LIVE"
                notes.append("VAD: Live Silero ONNX provider active")
        except Exception as exc:
            logger.warning("Silero VAD init failed, falling back to mock: %s", exc)

    if vad_provider is None:
        from src.audio.vad import MockVADProvider

        vad_provider = MockVADProvider(
            sample_rate=getattr(cfg, "vad_sample_rate", 16000),
            threshold=getattr(cfg, "vad_threshold", 0.5),
            min_speech_duration=getattr(cfg, "vad_min_speech_duration", 0.25),
            min_silence_duration=getattr(cfg, "vad_min_silence_duration", 0.5),
        )
        notes.append("VAD: Mock VAD provider active")

    # STT
    stt_mode = "MOCK"
    stt_provider = None
    if not force_mock and getattr(cfg, "stt_provider", "").lower() in ["faster-whisper", "whisper"]:
        try:
            from src.audio.stt import FasterWhisperSTTProvider

            candidate_stt = FasterWhisperSTTProvider(
                model_name=cfg.stt_model,
                language=cfg.stt_language,
                device=cfg.stt_device,
                compute_type=cfg.stt_compute_type,
                beam_size=cfg.stt_beam_size,
                settings=cfg,
            )
            if candidate_stt._model is not None:
                stt_provider = candidate_stt
                stt_mode = "LIVE"
                notes.append(f"STT: Live faster-whisper provider active ({cfg.stt_model})")
        except Exception as exc:
            logger.warning("Faster-whisper STT init failed, falling back to mock: %s", exc)

    if stt_provider is None:
        from src.audio.stt import MockSTTProvider

        stt_provider = MockSTTProvider(
            language=getattr(cfg, "stt_language", "en"),
        )
        notes.append("STT: Mock STT provider active")

    # TTS
    tts_mode = "MOCK"
    tts_provider = None
    if not force_mock and getattr(cfg, "tts_provider", "").lower() == "kokoro":
        try:
            from src.audio.tts import KokoroTTSProvider

            candidate_tts = KokoroTTSProvider(
                voice=cfg.tts_voice,
                language=cfg.tts_language,
                sample_rate=cfg.tts_sample_rate,
                output_format=cfg.tts_output_format,
                settings=cfg,
            )
            if candidate_tts._pipeline is not None:
                tts_provider = candidate_tts
                tts_mode = "LIVE"
                notes.append(f"TTS: Live Kokoro provider active ({cfg.tts_voice})")
        except Exception as exc:
            logger.warning("Kokoro TTS init failed, falling back to mock: %s", exc)

    if tts_provider is None:
        from src.audio.tts import MockTTSProvider

        tts_provider = MockTTSProvider(
            voice=getattr(cfg, "tts_voice", "af_heart"),
            language=getattr(cfg, "tts_language", "en-us"),
            sample_rate=getattr(cfg, "tts_sample_rate", 24000),
            output_format=getattr(cfg, "tts_output_format", "wav"),
        )
        notes.append("TTS: Mock TTS provider active (sine wave audio generator)")

    # -------------------------------------------------------------------------
    # 5. Overall Mode Determination
    # -------------------------------------------------------------------------
    if force_mock or (llm_mode == "MOCK" and rag_mode == "MOCK" and tool_mode == "MOCK"):
        overall_mode = RuntimeMode.MOCK
    elif llm_mode == "LIVE" and rag_mode == "LIVE" and tool_mode == "LIVE":
        overall_mode = RuntimeMode.LIVE
    else:
        overall_mode = RuntimeMode.HYBRID

    # -------------------------------------------------------------------------
    # 6. Core Engine & State Manager Assembly
    # -------------------------------------------------------------------------
    state_manager = ConversationStateManager()
    engine = ConversationEngine(
        llm_provider=llm_provider,
        knowledge_provider=knowledge_provider,
        tool_provider=tool_provider,
        settings=cfg,
    )

    components = ServiceComponents(
        engine=engine,
        state_manager=state_manager,
        llm_provider=llm_provider,
        knowledge_provider=knowledge_provider,
        tool_provider=tool_provider,
        vad_provider=vad_provider,
        stt_provider=stt_provider,
        tts_provider=tts_provider,
        llm_mode=llm_mode,
        rag_mode=rag_mode,
        tool_mode=tool_mode,
        vad_mode=vad_mode,
        stt_mode=stt_mode,
        tts_mode=tts_mode,
        overall_mode=overall_mode,
        initialization_notes=notes,
    )

    logger.info("Workbench bootstrapped in %s mode: %s", overall_mode.value, notes)
    return WorkbenchService(components=components)

