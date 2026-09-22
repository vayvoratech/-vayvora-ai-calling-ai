"""
Semantic Vector Router for Vayvora AI Voice Calling.

Replaces fragile keyword matching and regex with continuous dense vector space
intent classification using SentenceTransformer embeddings, knowledge-base
semantic resonance, and temperature-calibrated softmax confidence.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Tuple
import numpy as np

from src.agents.config import DEFAULT_AGENT_CONFIG, AgentConfig
from src.agents.state import AgentState
from src.rag.embeddings.provider import EmbeddingProvider, get_embedding_provider


class RouterNode:
    ROUTE_EXEMPLARS: Dict[str, List[str]] = {
        "rag": [
            "What generative AI and machine learning engineering services does Vayvora offer?",
            "Can you explain Vayvora's cloud infrastructure, AWS, and SaaS architecture capabilities?",
            "What is Vayvora Technology's pricing structure, hourly rates, and engagement models?",
            "Tell me about your custom web application and cross-platform mobile app development.",
            "What are Vayvora's corporate refund, privacy, and service level agreement policies?",
            "Where is Vayvora Technology headquartered, and what is your official contact info and phone number?",
            "What programming languages, frameworks, vector databases, and tech stacks do you specialize in?",
            "Can you tell me about the company background, vision, and portfolio at Vayvora?",
            "What are your uptime guarantees and client security commitments?",
            "Do you provide dedicated engineering squads for long term software projects?",
            "How does your engineering team handle zero-downtime database migrations?",
            "Can you explain your enterprise SLA and escalation procedures for critical incidents?",
            "Can you provide me the company website link or official URL?",
            "Can you give me the company mobile number or official contact telephone?",
            "Can you provide some other information about the company so I can know about it clearly?",
            "What does your company do and what services do you provide to clients?",
            "Can you tell me about the AI engineering course and pricing?",
            "What is the fee and structure for the AI engineering course?",
            "How much does the engineering course cost?",
            "What training programs and courses does Vayvora provide?",
            "I want to apply for a role in your company.",
            "How can I apply for a job or career at Vayvora Technology?",
            "Are there any job openings, vacancies, or internships at Vayvora?",
            "I would like to apply for an engineering role at your company.",
            "How do I submit my resume or CV for a job at Vayvora?",
        ],
        "mcp": [
            "Can you schedule a discovery call for tomorrow at three in the afternoon?",
            "Book a meeting with the enterprise solutions team next Tuesday morning.",
            "Check my calendar to see if I have any upcoming appointments scheduled today.",
            "Show me what is on my calendar for this week.",
            "Send an email to the client summarizing our architecture consultation.",
            "Read my recent unread incoming emails from the inbox.",
            "Send a WhatsApp message to the project lead with the status update.",
            "Reschedule my consultation appointment from two P M to four P M.",
            "Cancel the calendar meeting that was booked for Friday afternoon.",
            "Notify the engineering team on WhatsApp that the deployment is complete.",
            "Email partner at example dot com with the technical proposal attached.",
            "Check my email inbox to see if the client responded.",
        ],
        "direct": [
            "Hello, good morning! Can you hear me clearly?",
            "Hi there, how are you doing today?",
            "Hey Vayvora, good afternoon.",
            "Thank you very much, you have been super helpful.",
            "Thanks a lot, I appreciate your assistance.",
            "Goodbye! Have a wonderful rest of your day.",
            "Bye for now, talk to you later.",
            "Good evening, is anyone available?",
            "Hello there, thank you.",
        ],
        "llm": [
            "Can you explain that in simpler terms for me?",
            "What do you think is the best general architectural philosophy for modern startups?",
            "I am not quite sure what I need yet, can we talk through some conceptual ideas?",
            "Could you clarify what you mean by that previous statement?",
            "Can you summarize our conversation so far in one sentence?",
            "Why do you recommend microservices over monolithic architectures in general?",
            "How does asynchronous event-driven design differ from synchronous REST APIs?",
            "What is machine learning in general?",
            "What is artificial intelligence and ML?",
            "Can you explain what cloud computing is?",
            "What is deep learning?",
            "Who founded Apple or Microsoft?",
            "What is the capital of France?",
            "Tell me a joke or a story.",
        ],
    }

    DIRECT_RESPONSES: Dict[str, str] = {
        "hello": "Hello! I am Vayvora AI, the voice assistant for Vayvora Technology. How can I assist you with our engineering services or calendar appointments today?",
        "hi": "Hi there! Welcome to Vayvora Technology. How may I help you today?",
        "good morning": "Good morning! Welcome to Vayvora Technology. How can I assist you today?",
        "good afternoon": "Good afternoon! How can I assist you with Vayvora's services or schedule a call for you?",
        "good evening": "Good evening! How can I help you today?",
        "thanks": "You are very welcome! Please let me know if there is anything else I can assist you with.",
        "thank you": "You are most welcome! It is my pleasure to help.",
        "bye": "Goodbye! Thank you for calling Vayvora Technology. Have a wonderful day.",
        "goodbye": "Goodbye! Feel free to reach out anytime if you need further assistance.",
    }

    KB_TOPIC_ANCHORS: Dict[str, List[str]] = {
        "services_ai": [
            "Generative AI, machine learning engineering, AI Engineering course and training, custom LLM fine-tuning, voice AI calling, LangGraph multi-agent systems, vector databases",
            "Retrieval-Augmented Generation, semantic search, embeddings, Model Context Protocol integration, enterprise workflow automation",
        ],
        "services_cloud": [
            "Cloud infrastructure engineering, Amazon Web Services AWS, Google Cloud Platform GCP, Microsoft Azure, Kubernetes container orchestration",
            "Terraform infrastructure as code, serverless microservices, distributed Redis caching, zero-downtime CI CD pipelines, multi-tenant SaaS backends",
        ],
        "services_app": [
            "Full stack web development, React, Next.js, TypeScript, Node.js, Python, Go APIs, responsive web applications",
            "Cross platform mobile app development, iOS, Android, Flutter, React Native, real-time WebRTC audio video communication",
        ],
        "company_profile": [
            "Vayvora Technology company background, executive vision, mission, headquarters in San Francisco California, engineering centers in Hyderabad and Bengaluru India",
            "Official contact channels, email info at vayvora tech dot com, official website vayvora tech dot com, consultation inquiries",
        ],
        "policies_pricing": [
            "Enterprise Service Level Agreement SLA, ninety-nine point nine five percent uptime commitment, round-the-clock priority incident response",
            "Corporate refund policy, master service agreements, milestone safeguards, transparent pricing models, dedicated squads, hourly rates, AI Engineering course fee five thousand rupees",
        ],
        "company_careers": [
            "Vayvora Technology careers, job openings, vacancies, hiring process, applying for a role, software engineer jobs, AI engineer roles",
            "Submit resume to careers at vayvora tech dot com or info at vayvora tech dot com, engineering internships, technical recruitment, work at Vayvora",
        ],
        "company_assessments": [
            "Vayvora Technology technical assessment, practical coding examination, hiring interview stages, screening test",
            "Assessment duration sixty to ninety minutes, data structures algorithms GenAI evaluations, assessment scheduling and link",
        ],
    }

    def __init__(
        self,
        config: AgentConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.config = config or DEFAULT_AGENT_CONFIG
        self.embedding_provider = embedding_provider or get_embedding_provider()

        self.exemplar_vectors: Dict[str, np.ndarray] = {}
        self.centroids: Dict[str, np.ndarray] = {}
        self.kb_topic_centroids: Dict[str, np.ndarray] = {}
        self._initialize_vector_spaces()

    def _initialize_vector_spaces(self) -> None:
        for route, texts in self.ROUTE_EXEMPLARS.items():
            raw_vectors = self.embedding_provider.embed_documents(texts)
            arr = np.array(raw_vectors, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            norm_arr = arr / norms
            self.exemplar_vectors[route] = norm_arr

            centroid = np.mean(norm_arr, axis=0)
            c_norm = np.linalg.norm(centroid)
            self.centroids[route] = centroid / (c_norm if c_norm > 0 else 1.0)

        for topic_key, texts in self.KB_TOPIC_ANCHORS.items():
            raw_vectors = self.embedding_provider.embed_documents(texts)
            arr = np.array(raw_vectors, dtype=np.float32)
            centroid = np.mean(arr, axis=0)
            c_norm = np.linalg.norm(centroid)
            self.kb_topic_centroids[topic_key] = centroid / (c_norm if c_norm > 0 else 1.0)

    def _compute_kb_semantic_resonance(self, query_vec: np.ndarray) -> Tuple[float, str]:
        best_score = 0.0
        best_topic = "none"

        for topic_key, centroid in self.kb_topic_centroids.items():
            sim = float(np.dot(centroid, query_vec))
            if sim > best_score:
                best_score = sim
                best_topic = topic_key

        return best_score, best_topic

    def _compute_route_score(self, query_vec: np.ndarray, route: str, kb_resonance: float) -> float:
        exemplars = self.exemplar_vectors[route]
        similarities = np.dot(exemplars, query_vec)
        top3_avg = float(np.mean(np.sort(similarities)[-3:]))
        centroid_sim = float(np.dot(self.centroids[route], query_vec))

        base_score = float(
            self.config.router_top_k_weight * top3_avg
            + self.config.router_centroid_weight * centroid_sim
        )

        if route == "rag":
            w = self.config.kb_resonance_weight
            base_score = float((1.0 - w) * base_score + w * kb_resonance)

        return base_score

    @staticmethod
    def is_company_or_service_query(text: str) -> bool:
        t = text.lower().strip()
        
        # Catch phonetic misrecognitions from local STT
        vayvora_phonetics = (
            "vayvora", "vyvora", "vayvara", "wayvora", "vaivora",
            "where technologies", "what technology", "where technology",
            "why vora", "vi vora", "where technologies provide"
        )
        if any(p in t for p in vayvora_phonetics):
            return True

        company_refs = (
            "your company", "your firm", "your team", "your developers",
            "your engineers", "your agency", "your organization",
            "the company", "this company", "about company",
            "about the company", "company's", "companys",
            "your", "you offer", "you provide", "you build", "you have",
            "you do", "can you build", "can you develop",
        )
        if any(ref in t for ref in company_refs):
            return True

        company_domain_terms = (
            "course", "courses", "training", "bootcamp", "curriculum",
            "enroll", "enrolling", "enrollment", "admission", "internship", "internships", "intern", "interns",
            "price", "pricing", "cost", "fee", "fees", "rate", "rates",
            "charge", "charges", "how much", "quote", "quotation", "retainer",
            "refund", "sla", "service level agreement", "policy", "policies",
            "hire", "hiring", "squad", "dedicated squad", "engagement model",
            "office", "location", "headquarter", "headquarters", "address",
            "website", "mobile number", "phone number", "contact number",
            "contact info", "contact telephone", "email address",
            "services do you", "services you", "what services", "our services",
            "portfolio", "case studies", "case study",
            "ai engineering course", "engineering course",
            "apply", "applying", "application", "job", "jobs", "career", "careers",
            "role", "roles", "position", "positions", "vacancy", "vacancies",
            "opening", "openings", "recruit", "recruitment", "resume", "cv",
            "work for", "work at", "work with", "join your team", "join vayvora",
            "assessment", "assessments", "interview", "interviews", "exam", "exams", "examination",
            "coding test", "technical screening", "hiring process", "technologies provide",
            "services provide", "what are the services", "what services are provided"
        )
        if any(term in t for term in company_domain_terms):
            return True

        return False

    @classmethod
    def is_general_query(cls, text: str) -> bool:
        t = text.lower().strip().rstrip("?!.,;:")
        if not t:
            return False

        if cls.is_company_or_service_query(t):
            return False

        words = re.findall(r"\b\w+\b", t)
        word_set = set(words)

        if "your" in word_set or "you" in word_set:
            return False

        action_words = {
            "schedule", "book", "appointment", "calendar", "meeting",
            "email", "mail", "whatsapp", "message", "reschedule", "cancel", "send",
            "apply", "job", "career", "hiring", "role", "internship", "resume", "cv",
            "assessment", "interview", "exam", "test",
        }
        if word_set & action_words:
            return False

        courtesies = {
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "thanks", "thank you", "bye", "goodbye",
        }
        if t in courtesies or (word_set & {"hello", "hi", "hey", "thanks", "bye", "goodbye"} and len(words) <= 3):
            return False

        general_starters = (
            "what is", "what are", "what's", "whats", "define", "explain",
            "tell me what is", "tell me about what is", "meaning of", "definition of",
            "who is", "who was", "where is", "why is", "how does", "how do",
            "can you explain", "can you define", "tell me about",
        )
        for s in general_starters:
            if t.startswith(s) or f" {s} " in f" {t} ":
                return True

        standalone_concepts = {
            "machine learning", "ai and ml", "ai & ml", "artificial intelligence",
            "generative ai", "deep learning", "neural network", "neural networks",
            "cloud computing", "saas", "devops", "microservices", "monolith",
            "data science", "nlp", "natural language processing", "computer vision",
            "blockchain", "cybersecurity", "docker", "kubernetes", "python",
            "javascript", "react", "golang", "rust", "sql", "nosql",
        }
        if t in standalone_concepts:
            return True

        return False

    def classify_intent(
        self,
        query: str,
    ) -> Tuple[str, float, Dict[str, float], float, str, float]:
        if self.is_general_query(query):
            return (
                "llm",
                0.95,
                {"llm": 0.95, "rag": 0.02, "mcp": 0.02, "direct": 0.01},
                0.93,
                "General conceptual query routed to LLM with RAG disabled.",
                0.0,
            )

        query_vec = np.array(self.embedding_provider.embed_text(query), dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        kb_resonance, best_topic = self._compute_kb_semantic_resonance(query_vec)

        raw_scores = {
            route: self._compute_route_score(query_vec, route, kb_resonance)
            for route in self.ROUTE_EXEMPLARS.keys()
        }

        temp = self.config.router_temperature
        exp_scores = {r: math.exp(s / temp) for r, s in raw_scores.items()}
        sum_exp = sum(exp_scores.values())
        norm_scores = {r: round(v / sum_exp, 4) for r, v in exp_scores.items()}

        sorted_routes = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
        top_route, top_score = sorted_routes[0]
        second_route, second_score = sorted_routes[1]
        margin = round(top_score - second_score, 4)
        max_raw = max(raw_scores.values())

        greeting_words = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening", "thanks", "thank you", "bye", "goodbye"}
        has_greeting_token = any(gw in query.lower() for gw in greeting_words)
        if top_route == "direct" and not has_greeting_token:
            top_route = second_route
            top_score = second_score
            margin = 0.05

        if max_raw < 0.18:
            selected_route = "llm"
            rationale = "Low raw semantic affinity. Falling back to LLM."
        elif margin < self.config.router_min_margin and top_score < 0.55:
            selected_route = "llm"
            rationale = f"Semantic ambiguity between '{top_route}' and '{second_route}'. Defaulting to LLM."
        elif top_score < self.config.router_confidence_threshold:
            selected_route = "llm"
            rationale = "Top probability below threshold. Defaulting to LLM."
        else:
            selected_route = top_route
            rationale = f"Semantic match: '{top_route}' ({top_score:.3f}). Resonance: {kb_resonance:.3f}."

        return selected_route, top_score, norm_scores, margin, rationale, kb_resonance

    async def run(self, state: AgentState) -> AgentState:
        start_time = time.perf_counter()
        user_input = state.get("user_input", "").strip()
        messages = list(state.get("messages", []))

        if not user_input:
            return {
                **state,
                "route": "direct",
                "route_confidence": 1.0,
                "similarity_scores": {"direct": 1.0, "rag": 0.0, "mcp": 0.0, "llm": 0.0},
                "route_margin": 1.0,
                "route_rationale": "Empty user input detected.",
                "kb_resonance": 0.0,
                "response": "I'm sorry, I didn't hear anything.",
                "is_complete": True,
                "routing_latency_ms": 0.1,
            }

        normalized_text = user_input.lower().strip().rstrip("?!.,;:")

        user_history = [m.get("content", "").strip() for m in messages if m.get("role") == "user"]
        assistant_history = [m.get("content", "").strip() for m in messages if m.get("role") == "assistant"]

        has_time = bool(re.search(r"\b(\d{1,2}(:\d{2})?\s*(am|pm|a\.m\.|p\.m\.)|\d{1,2}\s*o'?clock|morning|afternoon|evening|\d{1,2}:\d{2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", normalized_text))
        has_date = bool(re.search(r"\b(today|tomorrow|tommorow|tomorow|tommorrow|tmrw|next\s+day|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}-\d{2}-\d{2}|january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b", normalized_text))

        _last_asst = assistant_history[-1].lower() if assistant_history else ""
        _name_was_prompted = any(term in _last_asst for term in ("full name", "your name", "caller name", "who is calling"))

        _name_tokens = normalized_text.split()
        _non_name_tokens = {
            "ok", "okay", "yes", "no", "not", "nope", "never", "prefer", "the", "a", "an",
            "whatsapp", "email", "mobile", "number", "phone", "address", "vayvora", "here",
            "ready", "interested", "calling", "consultation", "meeting", "call", "schedule",
            "appointment", "tomorrow", "tommorow", "today", "morning", "afternoon", "evening", "please",
            "thanks", "thank", "hello", "hi", "hey", "good", "want", "like", "trying", "looking",
            "proceed", "confirm", "booking", "book", "date", "time", "details"
        }
        _is_pure_name = (
            all(re.match(r"^[A-Za-z]+$", w) for w in _name_tokens)
            and not any(w in _non_name_tokens for w in _name_tokens)
            and (
                (2 <= len(_name_tokens) <= 4)
                or (len(_name_tokens) == 1 and _name_was_prompted and len(_name_tokens[0]) >= 2)
            )
        )
        has_name = (
            bool(re.search(r"\b(my name is|name is|i am|this is)\b", normalized_text))
            or _is_pure_name
        )
        has_email = "@" in normalized_text or "dot com" in normalized_text
        has_phone = bool(re.search(r"\b\d{10}\b|(?:\+91[\s-]?)?[6-9]\d{9}\b", normalized_text))
        has_whatsapp = "whatsapp" in normalized_text

        is_affirmative = bool(re.search(r"\b(yes|sure|yeah|yep|please|confirm|go ahead|okay|ok|fine|proceed)\b", normalized_text))
        is_negative = bool(re.search(r"\b(no|nope|never|don't|dont|no thanks|skip|avoid|nah)\b", normalized_text))

        # Check multi-turn scheduling responses
        if assistant_history:
            last_assistant = assistant_history[-1].lower()
            if "whatsapp" in last_assistant and (is_affirmative or is_negative or has_whatsapp or has_email or has_phone):
                return {
                    **state,
                    "route": "mcp",
                    "route_confidence": 0.98,
                    "similarity_scores": {"mcp": 0.98, "llm": 0.01, "rag": 0.01, "direct": 0.0},
                    "route_margin": 0.97,
                    "route_rationale": "Caller answering WhatsApp confirmation preference.",
                    "kb_resonance": 0.0,
                    "llm_required": True,
                    "rag_required": False,
                    "tool_required": True,
                    "is_complete": False,
                    "routing_latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                }

            scheduling_keywords = (
                "full name", "preferred time", "preferred date", "what date",
                "which time", "schedule an appointment", "mobile number", "email address"
            )
            if any(k in last_assistant for k in scheduling_keywords) and (has_time or has_date or has_name or has_email or has_phone):
                return {
                    **state,
                    "route": "mcp",
                    "route_confidence": 0.98,
                    "similarity_scores": {"mcp": 0.98, "llm": 0.01, "rag": 0.01, "direct": 0.0},
                    "route_margin": 0.97,
                    "route_rationale": "Caller providing appointment parameters.",
                    "kb_resonance": 0.0,
                    "llm_required": True,
                    "rag_required": False,
                    "tool_required": True,
                    "is_complete": False,
                    "routing_latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                }

        # Check mid-call greetings
        if len(messages) > 0 and normalized_text in ("hello", "hi", "hey", "can you hear me", "are you there"):
            return {
                **state,
                "route": "direct",
                "route_confidence": 0.99,
                "similarity_scores": {"direct": 0.99, "rag": 0.0, "mcp": 0.0, "llm": 0.01},
                "route_margin": 0.98,
                "route_rationale": "Mid-call presence check-in.",
                "kb_resonance": 0.0,
                "response": "Yes, I am here. Please go ahead.",
                "is_complete": True,
                "routing_latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }

        # Check standalone initial greeting
        if len(messages) == 0 and normalized_text in self.DIRECT_RESPONSES:
            return {
                **state,
                "route": "direct",
                "route_confidence": 0.99,
                "similarity_scores": {"direct": 0.99, "rag": 0.0, "mcp": 0.0, "llm": 0.01},
                "route_margin": 0.98,
                "route_rationale": "Greeting at start of call.",
                "kb_resonance": 0.0,
                "response": self.DIRECT_RESPONSES[normalized_text],
                "is_complete": True,
                "routing_latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }

        # Fast direct check for company/services inquiries
        if self.is_company_or_service_query(user_input):
            return {
                **state,
                "route": "rag",
                "route_confidence": 0.95,
                "similarity_scores": {"rag": 0.95, "llm": 0.03, "mcp": 0.01, "direct": 0.01},
                "route_margin": 0.92,
                "route_rationale": "Inquiry matched Vayvora services, courses, or company profile.",
                "kb_resonance": 0.85,
                "llm_required": True,
                "rag_required": True,
                "tool_required": False,
                "is_complete": False,
                "routing_latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }

        # Vector classification without affirmative prefix poisoning
        selected_route, confidence, scores, margin, rationale, kb_resonance = self.classify_intent(user_input)

        if selected_route == "mcp":
            return {
                **state,
                "route": "mcp",
                "route_confidence": confidence,
                "similarity_scores": scores,
                "route_margin": margin,
                "route_rationale": rationale,
                "kb_resonance": kb_resonance,
                "llm_required": True,
                "rag_required": False,
                "tool_required": True,
                "is_complete": False,
                "routing_latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }

        if selected_route == "rag":
            return {
                **state,
                "route": "rag",
                "route_confidence": confidence,
                "similarity_scores": scores,
                "route_margin": margin,
                "route_rationale": rationale,
                "kb_resonance": kb_resonance,
                "llm_required": True,
                "rag_required": True,
                "tool_required": False,
                "is_complete": False,
                "routing_latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }

        return {
            **state,
            "route": "llm",
            "route_confidence": confidence,
            "similarity_scores": scores,
            "route_margin": margin,
            "route_rationale": rationale,
            "kb_resonance": kb_resonance,
            "llm_required": True,
            "rag_required": False,
            "tool_required": False,
            "is_complete": False,
            "routing_latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }