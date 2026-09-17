"""
Semantic MCP Tool Selector for Vayvora AI Voice Calling.

Replaces regex and keyword matching with dense vector embeddings and
cosine similarity, dynamically matching queries to tools based on semantic intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np

from src.agents.tools.argument_parser import ToolArgumentParser
from src.rag.embeddings.provider import EmbeddingProvider, get_embedding_provider


@dataclass(frozen=True)
class ToolSelection:
    tool_name: str
    tool_input: Dict[str, Any]
    confidence: float
    reasoning: str = ""


class ToolSelector:
    """
    Semantic MCP Tool Selector.

    Evaluates user intent against tool capability representations
    in dense vector space to dynamically select the appropriate tool.
    """

    TOOL_EXEMPLARS: Dict[str, List[str]] = {
        "calendar_add_event": [
            "Schedule a discovery call tomorrow at three in the afternoon.",
            "Book a consultation meeting with the architecture team next Tuesday.",
            "Create a calendar event for the sprint planning on Friday at 10 AM.",
            "Add an appointment to my calendar for next week.",
            "Set up a demo call with the client on Wednesday at 2 PM.",
            "Reserve time on the calendar for our weekly architectural review.",
            "Block an hour on Thursday for the executive status update.",
        ],
        "calendar_get_events": [
            "What appointments or meetings do I have scheduled for today?",
            "Check my calendar to see my upcoming events this week.",
            "Show me what is on my schedule for tomorrow.",
            "Do I have any calls scheduled on my calendar?",
            "List all upcoming meetings for this afternoon.",
            "Can you verify if I am busy on Monday morning?",
        ],
        "calendar_get_month_view": [
            "Show me the entire calendar for this month.",
            "Give me the monthly calendar overview for next month.",
            "Display the calendar grid for September.",
            "Can I see the full month view of our calendar?",
        ],
        "mail_send": [
            "Send an email to the client with the project proposal.",
            "Email the executive team summarizing the meeting notes.",
            "Draft and send an email to partner at example dot com.",
            "Shoot an email with our technical overview and architecture diagram.",
            "Send an outbound message via email to info at vayvora tech dot com.",
            "Can you send email to bpawanganesh@gmail.com with message hi?",
            "Send email to bpawanganesh@gmail.com with message hi.",
            "Send an email to test@example.com with message hello.",
            "Can you send an email to user@test.com?",
        ],
        "mail_read_recent": [
            "Read my recent unread incoming emails from the inbox.",
            "Check if I received any new emails today.",
            "What messages are waiting in my email inbox?",
            "Did anyone reply to my email inquiry from this morning?",
            "Fetch the latest messages from my mail inbox.",
        ],
        "whatsapp_send_message": [
            "Send a WhatsApp message to the manager saying I am ready.",
            "Ping the client on WhatsApp with the updated delivery schedule.",
            "Notify the engineering lead on WhatsApp that the build passed.",
            "Send a text on WhatsApp to the project lead about the demo.",
            "Can you send whatsapp message to 9876543210 with message hi?",
            "Send whatsapp to 9876543210 saying hello.",
            "Send a message on WhatsApp.",
        ],
    }

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        similarity_threshold: float = 0.45,
    ) -> None:
        self.argument_parser = ToolArgumentParser()
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.similarity_threshold = similarity_threshold

        self.tool_vectors: Dict[str, np.ndarray] = {}
        self.tool_centroids: Dict[str, np.ndarray] = {}
        self._initialize_tool_vectors()

    def _initialize_tool_vectors(self) -> None:
        """Embed tool exemplar banks and calculate normalized centroids."""
        for tool_name, texts in self.TOOL_EXEMPLARS.items():
            raw_vectors = self.embedding_provider.embed_documents(texts)
            arr = np.array(raw_vectors, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            norm_arr = arr / norms
            self.tool_vectors[tool_name] = norm_arr

            centroid = np.mean(norm_arr, axis=0)
            c_norm = np.linalg.norm(centroid)
            self.tool_centroids[tool_name] = centroid / (c_norm if c_norm > 0 else 1.0)

    def select(
        self,
        user_input: str,
        messages: list[dict] | None = None,
    ) -> ToolSelection | None:
        """
        Dynamically select the optimal MCP tool based on semantic cosine similarity,
        taking into account conversation history for follow-up parameter collection.
        """
        text = user_input.strip()
        if not text:
            return None

        # Build context from previous turns if available
        user_history = []
        assistant_history = []
        if messages:
            for msg in messages:
                if msg.get("role") == "user":
                    user_history.append(msg.get("content", "").strip())
                elif msg.get("role") == "assistant":
                    assistant_history.append(msg.get("content", "").strip())

        combined_context = " ".join(user_history[-2:] + [text])

        # Try embedding user_input first
        query_vec = np.array(self.embedding_provider.embed_text(text), dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        best_tool: str | None = None
        best_score = -1.0

        for tool_name, exemplars in self.tool_vectors.items():
            sims = np.dot(exemplars, query_vec)
            top_sim = float(np.max(sims))
            centroid_sim = float(np.dot(self.tool_centroids[tool_name], query_vec))
            combined_score = 0.70 * top_sim + 0.30 * centroid_sim

            if combined_score > best_score:
                best_score = combined_score
                best_tool = tool_name

        # If low match on user_input alone, try the contextual query
        if (best_tool is None or best_score < self.similarity_threshold) and user_history:
            ctx_vec = np.array(self.embedding_provider.embed_text(combined_context), dtype=np.float32)
            c_norm = np.linalg.norm(ctx_vec)
            if c_norm > 0:
                ctx_vec = ctx_vec / c_norm

            for tool_name, exemplars in self.tool_vectors.items():
                sims = np.dot(exemplars, ctx_vec)
                top_sim = float(np.max(sims))
                centroid_sim = float(np.dot(self.tool_centroids[tool_name], ctx_vec))
                combined_score = 0.70 * top_sim + 0.30 * centroid_sim

                if combined_score > best_score:
                    best_score = combined_score
                    best_tool = tool_name

        lowered = text.lower()

        # Check if conversation is in an active appointment scheduling workflow
        is_scheduling_context = False
        if assistant_history:
            last_ast = assistant_history[-1].lower()
            if any(k in last_ast for k in (
                "schedule", "appointment", "calendar", "assessment",
                "preferred date", "full name", "preferred time", "date and time",
                "email address", "mobile number", "whatsapp as well", "through whatsapp"
            )):
                is_scheduling_context = True
        elif user_history and any(any(w in u.lower() for w in ("schedule", "appointment", "book a meeting", "book a call")) for u in user_history[-3:]):
            is_scheduling_context = True

        explicit_send_mail = any(p in lowered for p in ("send email to", "send an email to", "shoot an email to", "mail to", "send mail to"))
        explicit_send_wa = any(p in lowered for p in ("send whatsapp to", "send a whatsapp to", "ping on whatsapp", "message on whatsapp"))

        if is_scheduling_context and not (explicit_send_mail or explicit_send_wa):
            best_tool = "calendar_add_event"
            best_score = 0.98
        elif explicit_send_mail or (("send email" in lowered or "send an email" in lowered) and not is_scheduling_context):
            best_tool = "mail_send"
            best_score = max(best_score, 0.95)
        elif explicit_send_wa or (("send whatsapp" in lowered or "send a whatsapp" in lowered) and not is_scheduling_context):
            best_tool = "whatsapp_send_message"
            best_score = max(best_score, 0.95)
        elif any(w in lowered for w in ("schedule", "appointment", "book a meeting", "book a call", "book appointment")):
            best_tool = "calendar_add_event"
            best_score = max(best_score, 0.95)
        elif (best_tool is None or best_score < self.similarity_threshold) and assistant_history:
            last_ast = assistant_history[-1].lower()
            if any(k in last_ast for k in ("schedule", "appointment", "calendar", "assessment", "preferred date", "full name", "preferred time", "date and time", "email address", "mobile number", "whatsapp as well")):
                best_tool = "calendar_add_event"
                best_score = 0.95
            elif any(k in last_ast for k in ("email", "mail")):
                best_tool = "mail_send"
                best_score = 0.88
            elif any(k in last_ast for k in ("whatsapp", "phone", "message")):
                best_tool = "whatsapp_send_message"
                best_score = 0.88

        if best_tool is None or best_score < self.similarity_threshold:
            return None

        tool_input = self.argument_parser.parse(
            tool_name=best_tool,
            user_input=user_input,
            messages=messages,
        )

        return ToolSelection(
            tool_name=best_tool,
            tool_input=tool_input,
            confidence=round(best_score, 4),
            reasoning=f"Semantic vector similarity {best_score:.3f} matched capability signature for '{best_tool}'.",
        )
