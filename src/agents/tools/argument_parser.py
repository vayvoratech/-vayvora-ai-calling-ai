import re
from datetime import datetime, timedelta
from typing import Any


class ToolArgumentParser:
    """
    Fast deterministic argument parser for simple MCP actions.
    Extracts dates, times, emails, caller names, and phone numbers across
    both current user input and multi-turn conversation history.
    """

    def parse(
        self,
        tool_name: str,
        user_input: str,
        messages: list[dict] | None = None,
    ) -> dict[str, Any]:
        text = user_input.strip()

        # Build merged context across conversation turns
        user_history = []
        if messages:
            for msg in messages:
                if msg.get("role") == "user":
                    user_history.append(msg.get("content", "").strip())
        all_user_texts = user_history + [text]

        if tool_name == "calendar_add_event":
            return self._parse_calendar_event(text, all_user_texts, messages=messages)

        if tool_name == "mail_send":
            return self._parse_email(text, all_user_texts)

        if tool_name == "whatsapp_send_message":
            return self._parse_whatsapp(text, all_user_texts)

        if tool_name == "calendar_get_month_view":
            return self._parse_month_view(text, all_user_texts)

        return {}

    def _parse_calendar_event(
        self,
        text: str,
        all_user_texts: list[str],
        messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        combined_text = " ".join(all_user_texts)

        # Check if the immediate preceding assistant message specifically asked for the caller's name
        name_was_asked = False
        if messages:
            for m in reversed(messages):
                if m.get("role") == "assistant":
                    c = m.get("content", "").lower()
                    if "full name" in c or "your name" in c or "caller name" in c or "who is calling" in c:
                        name_was_asked = True
                    break

        # Extract name by scanning user messages from latest to oldest
        name = ""
        for idx, u in enumerate(reversed(all_user_texts)):
            is_direct = (idx == 0 and name_was_asked)
            cand_name = self._extract_name(u, is_direct_name_response=is_direct)
            if cand_name:
                name = cand_name
                break

        base_title = self._extract_title(combined_text)

        # If caller name is available, format a clear event title
        if name and name.lower() not in base_title.lower():
            title = f"{base_title} - {name}"
        else:
            title = base_title

        # Date and Time: check newest utterances first
        date_str = ""
        for u in reversed(all_user_texts):
            cand_date = self._extract_date(u)
            if cand_date:
                date_str = cand_date
                break

        time_str = ""
        for u in reversed(all_user_texts):
            cand_time = self._extract_time(u)
            if cand_time:
                time_str = cand_time
                break

        email = self._extract_email(text) or self._extract_email(combined_text)
        mobile = self._extract_phone(text) or self._extract_phone(combined_text)
        
        last_assistant_text = ""
        if messages:
            for m in reversed(messages):
                if m.get("role") == "assistant":
                    last_assistant_text = m.get("content", "")
                    break

        whatsapp_opt_in = self._extract_whatsapp_pref(text, last_assistant=last_assistant_text)
        if whatsapp_opt_in is None:
            whatsapp_opt_in = self._extract_whatsapp_pref(combined_text, last_assistant=last_assistant_text)

        # Multi-stage missing fields tracking:
        # Stage 1: Collect Name, Date, and Time
        # Stage 2: Collect Email, Mobile Number, and WhatsApp confirmation preference
        # Stage 3: Complete
        missing_fields = []
        missing_stage = "complete"

        if not name or not date_str or not time_str:
            if not name and not date_str and not time_str:
                missing_fields = ["full name", "preferred date and time"]
                missing_stage = "name_and_datetime"
            elif not name and (not date_str or not time_str):
                missing_fields = ["full name", "preferred date and time"]
                missing_stage = "name_and_datetime"
            elif not name:
                missing_fields = ["full name"]
                missing_stage = "name"
            elif not date_str and not time_str:
                missing_fields = ["preferred date and time"]
                missing_stage = "datetime"
            elif not date_str:
                missing_fields = ["preferred date"]
                missing_stage = "date"
            else:
                missing_fields = ["preferred time"]
                missing_stage = "time"
        elif not email or not mobile or (whatsapp_opt_in is None):
            if not email and not mobile:
                missing_fields = ["email address", "mobile number", "whatsapp confirmation preference"]
                missing_stage = "contact_and_whatsapp"
            elif not email:
                missing_fields = ["email address"]
                missing_stage = "email"
            elif not mobile:
                missing_fields = ["mobile number"]
                missing_stage = "mobile"
            else:
                missing_fields = ["whatsapp confirmation preference"]
                missing_stage = "whatsapp"

        desc_parts = [f"Scheduled for {name or 'Client'}."]
        if email:
            desc_parts.append(f"Email: {email}.")
        if mobile:
            desc_parts.append(f"Mobile: {mobile}.")
        if whatsapp_opt_in is not None:
            desc_parts.append(f"WhatsApp Confirmation: {'Yes' if whatsapp_opt_in else 'No'}.")

        description = " ".join(desc_parts)

        return {
            "title": title,
            "date_str": date_str,
            "time_str": time_str,
            "description": description,
            "caller_name": name,
            "email": email,
            "mobile": mobile,
            "whatsapp_opt_in": bool(whatsapp_opt_in) if whatsapp_opt_in is not None else False,
            "missing_fields": missing_fields,
            "missing_stage": missing_stage,
            "is_complete": len(missing_fields) == 0,
        }

    def _parse_month_view(
        self,
        text: str,
        all_user_texts: list[str],
    ) -> dict[str, Any]:
        combined = " ".join(all_user_texts)
        now = datetime.now()
        year = now.year
        month = now.month

        # Check for month names
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        for name, num in months.items():
            if name in text.lower() or name in combined.lower():
                month = num
                break

        if "next month" in text.lower() or "next month" in combined.lower():
            if month == 12:
                month = 1
                year += 1
            else:
                month += 1

        return {"year": year, "month": month}

    def _parse_email(
        self,
        text: str,
        all_user_texts: list[str],
    ) -> dict[str, Any]:
        combined = " ".join(all_user_texts)
        to = self._extract_email(text) or self._extract_email(combined)

        # Extract subject if explicitly provided
        subject = ""
        sub_match = (
            re.search(r"(?:subject|regarding|topic)\s+(?:is|to|about)?\s*[\"']?([^\"'\n,]+)[\"']?", text, re.IGNORECASE)
            or re.search(r"(?:subject|regarding|topic)\s+(?:is|to|about)?\s*[\"']?([^\"'\n,]+)[\"']?", combined, re.IGNORECASE)
        )
        if sub_match:
            subject = sub_match.group(1).strip()

        # Extract explicit message content
        raw_content = ""
        body_match = re.search(
            r"(?:with\s+message|saying|message\s+is|content\s+is|say\s+that|tell\s+(?:them|him|her)\s+that)\s*[\"']?([^\"'\n]+)[\"']?",
            text,
            re.IGNORECASE
        )
        if body_match:
            raw_content = body_match.group(1).strip()
        else:
            # Check if user mentioned a specific topic/subject
            topic_match = re.search(r"(?:about|regarding|concerning)\s+([^\"'\n,]+)", text, re.IGNORECASE)
            if topic_match:
                topic = topic_match.group(1).strip()
                if not subject:
                    subject = f"Inquiry regarding {topic.title()}"
                raw_content = (
                    f"Thank you for contacting Vayvora Technology.\n\n"
                    f"We are following up regarding {topic}. "
                    f"Our team provides comprehensive enterprise AI systems, cloud architecture, "
                    f"and custom software engineering services."
                )

        # Build clean professional business email body (never echo the user's raw prompt query)
        name = self._extract_name(text) or self._extract_name(combined)
        recipient_greeting = f"Dear {name}," if name else "Hello,"

        if raw_content:
            body = (
                f"{recipient_greeting}\n\n"
                f"{raw_content}\n\n"
                f"If you have any questions or require additional details, please feel free to reply to this email.\n\n"
                f"Best regards,\n"
                f"Vayvora Technology Team\n"
                f"https://www.vayvoratech.com\n"
                f"info@vayvoratech.com"
            )
        else:
            body = ""

        if not subject:
            subject = "Message from Vayvora Technology"

        missing_fields = []
        if not to:
            missing_fields.append("recipient email address")
        if not body:
            missing_fields.append("email message or purpose")

        return {
            "to": to,
            "subject": subject,
            "body": body,
            "missing_fields": missing_fields,
            "is_complete": len(missing_fields) == 0,
        }

    def _parse_whatsapp(
        self,
        text: str,
        all_user_texts: list[str],
    ) -> dict[str, Any]:
        combined = " ".join(all_user_texts)
        phone = self._extract_phone(text) or self._extract_phone(combined)

        raw_content = ""
        body_match = re.search(
            r"(?:with\s+message|saying|message\s+is|content\s+is|say\s+that|tell\s+(?:them|him|her)\s+that)\s*[\"']?([^\"'\n]+)[\"']?",
            text,
            re.IGNORECASE
        )
        if body_match:
            raw_content = body_match.group(1).strip()
        else:
            topic_match = re.search(r"(?:about|regarding|concerning)\s+([^\"'\n,]+)", text, re.IGNORECASE)
            if topic_match:
                topic = topic_match.group(1).strip()
                raw_content = f"Hello from Vayvora Technology! We are reaching out regarding {topic}."

        missing_fields = []
        if not phone:
            missing_fields.append("phone number")
        if not raw_content:
            missing_fields.append("WhatsApp message content")

        message = (
            f"Hello from Vayvora Technology! {raw_content}" if raw_content
            else ""
        )

        return {
            "phone": phone,
            "message": message,
            "missing_fields": missing_fields,
            "is_complete": len(missing_fields) == 0,
        }

    @staticmethod
    def _extract_email(text: str) -> str:
        match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
        return match.group(0) if match else ""

    @staticmethod
    def _extract_phone(text: str) -> str:
        match = re.search(r"(?:\+91[\s-]?)?[6-9]\d{9}\b", text)
        if match:
            return re.sub(r"\D", "", match.group(0))
        match_gen = re.search(r"\b\d{10,12}\b", text)
        if match_gen:
            return match_gen.group(0)
        return ""

    @staticmethod
    def _extract_whatsapp_pref(text: str, last_assistant: str = "") -> bool | None:
        t = text.lower()
        ast = last_assistant.lower()

        # 1. Explicit mention of whatsapp in user text
        if "whatsapp" in t:
            negative_phrases = (
                "no whatsapp", "not on whatsapp", "don't send on whatsapp", "dont send on whatsapp",
                "not prefer the whatsapp", "not prefer whatsapp", "dont prefer whatsapp", "don't prefer whatsapp",
                "do not prefer whatsapp", "i am not prefer", "not prefer", "dont prefer", "don't prefer",
                "no to whatsapp", "not interested in whatsapp", "skip whatsapp", "avoid whatsapp",
                "not whatsapp", "no", "nope", "never", "dont", "don't"
            )
            if any(neg in t for neg in negative_phrases):
                return False

            if re.search(r"\b(no|not|dont|don't|never|skip|avoid|none)\b.*whatsapp", t) or re.search(r"whatsapp.*\b(no|not|dont|don't|never|skip|avoid|none)\b", t):
                return False

            positive_phrases = (
                "send on whatsapp", "send through whatsapp", "send via whatsapp",
                "send it on whatsapp", "yes for whatsapp", "yes to whatsapp",
                "whatsapp is fine", "whatsapp also", "both email and whatsapp",
                "both whatsapp and email", "send to both", "on whatsapp as well",
                "prefer whatsapp", "whatsapp is good", "whatsapp is okay",
                "yes please", "sure", "yes you can", "yeah", "yep", "okay", "ok", "yes"
            )
            if any(phrase in t for phrase in positive_phrases):
                return True
            if any(pos in t for pos in ("yes", "sure", "ok", "okay", "fine", "can send", "please", "send", "good")):
                return True
            return True

        # Check explicit email-only phrases (implying declining WhatsApp)
        if any(eo in t for eo in ("only email", "email only", "no, only email", "only to email", "only send to email", "send to email only")):
            return False

        # 2. If user did not mention WhatsApp, but the assistant specifically asked about WhatsApp in prior turn
        if "whatsapp" in ast:
            if re.search(r"\b(no|nope|never|not really|dont|don't|skip|avoid|no thanks|no thank you|only email)\b", t):
                return False
            if re.search(r"\b(yes|sure|yeah|yep|ok|okay|fine|please|please do|certainly|absolutely|proceed|good)\b", t):
                return True

        return None

    @staticmethod
    def _extract_name(text: str, is_direct_name_response: bool = False) -> str:
        """Extracts person name from utterances like 'Pawan', 'Pawan Ganesh', 'my name is Kiran', or 'Kanteti Kiran Kumar'."""
        common_non_name_tokens = {
            "ok", "okay", "yes", "no", "not", "nope", "never", "prefer", "the", "a", "an",
            "whatsapp", "email", "mobile", "number", "phone", "address", "vayvora", "here",
            "ready", "interested", "calling", "consultation", "meeting", "call", "schedule",
            "appointment", "tomorrow", "tommorow", "today", "morning", "afternoon", "evening", "please",
            "thanks", "thank", "hello", "hi", "hey", "hlo", "hlw", "helo", "heyy", "hiya", "yo", "sup",
            "hola", "namaste", "vanakkam", "greetings", "good", "want", "like", "trying", "looking",
            "can", "could", "would", "am", "is", "are", "sure", "yep", "yeah", "fine", "cool", "great",
            "proceed", "done", "now", "later", "soon", "confirm", "confirmed", "booking",
            "book", "date", "time", "details", "info", "invite", "link",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "january", "february", "march", "april", "may", "june", "july", "august",
            "september", "october", "november", "december", "pm", "am", "clock", "oclock", "o'clock",
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
            "role", "roles", "position", "positions", "job", "jobs", "career", "careers", "intern", "internship",
            "internships", "developer", "engineer", "engineering", "assessment", "assessments", "interview",
            "interviews", "exam", "exams", "test", "tests", "screening", "hire", "hiring", "apply", "applying",
            "candidate", "candidates", "student", "students", "resume", "cv", "portfolio", "github", "hr",
            "talent", "team", "company", "product", "products", "project", "projects", "enterprise", "client",
            "clients", "delegate", "delegates", "work", "tech", "technology", "software", "ai", "genai", "cloud",
            "aws", "gcp", "devops", "service", "services", "solve", "problem",
        }

        patterns = (
            r"(?:my\s+name\s+is|name\s+is|this\s+is)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})",
            r"(?:name:\s*)([A-Za-z]+(?:\s+[A-Za-z]+){0,3})",
            r"(?:for\s+)([A-Za-z]+(?:\s+[A-Za-z]+){0,2})\s+(?:tomorrow|tommorow|today|on\b|at\b)",
            r"(?:schedule|book|meeting|call|appointment)\s+(?:for\s+)?([A-Za-z]+(?:\s+[A-Za-z]+){0,3})",
        )
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                words = candidate.lower().split()
                if not any(w in common_non_name_tokens for w in words):
                    return candidate.title()

        # Check 'I am [Name]' but only if words are strictly proper name tokens
        i_am_match = re.search(r"\bi\s+am\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})", text, re.IGNORECASE)
        if i_am_match:
            candidate = i_am_match.group(1).strip()
            words = candidate.lower().split()
            if not any(w in common_non_name_tokens for w in words):
                return candidate.title()

        start_match = re.match(r"^\s*([A-Za-z]+(?:\s+[A-Za-z]+){0,3})\s+(?:and\s+my|and\s+preferred|\band\b)", text, re.IGNORECASE)
        if start_match:
            candidate = start_match.group(1).strip()
            words = candidate.lower().split()
            if not any(w in common_non_name_tokens for w in words):
                return candidate.title()

        cleaned = text.strip()
        words = cleaned.split()
        if all(re.match(r"^[A-Za-z]+$", w) for w in words):
            # Multi-word full names (e.g. 'Pawan Ganesh', 'Kiran Kanteti')
            if 2 <= len(words) <= 4:
                if not any(w.lower() in common_non_name_tokens for w in words):
                    return cleaned.title()
            # Single-word name only if caller was directly asked for their name
            elif len(words) == 1 and is_direct_name_response:
                w_low = words[0].lower()
                if w_low not in common_non_name_tokens and len(w_low) >= 2:
                    return cleaned.title()

        return ""

    @staticmethod
    def _extract_title(text: str) -> str:
        # Check specific event keywords first
        low = text.lower()
        if "with our head" in low or "with head" in low or "head of" in low:
            return "Meeting with Head of Technology"
        if "assessment" in low:
            return "Assessment"
        if "discovery call" in low:
            return "Discovery Call"
        if "consultation" in low:
            return "Consultation"
        if "architecture review" in low:
            return "Architecture Review"
        if "sprint planning" in low:
            return "Sprint Planning"
        if "interview" in low:
            return "Interview"
        if "demo" in low:
            return "Product Demo"

        patterns = (
            r"(?:schedule|book|create|register|set\s+up)\s+(?:a\s+|an\s+|my\s+)?([A-Za-z0-9\s-]+?)(?:\s+tomorrow|\s+tommorow|\s+today|\s+on\b|\s+at\b|\s+for\b|$)",
            r"(?:schedule|book|create|register)\s+(?:a\s+|an\s+)?(.+?)\s+at\s+\d",
            r"(?:meeting|call|session)\s+(?:for|about)\s+(.+?)(?:\s+at|\s+on|\s+tomorrow|\s+tommorow|\s+today|$)",
        )

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if candidate and len(candidate) < 40 and candidate.lower() not in ("a", "an", "the", "my"):
                    return candidate.title()

        return "Consultation Meeting"

    @staticmethod
    def _extract_date(text: str) -> str:
        today = datetime.now().date()
        low = text.lower()

        # Check day after tomorrow first (including common typos)
        if re.search(r"\b(?:day\s+after\s+(?:tomorrow|tommorow|tomorow|tommorrow|tmrw)|overmorrow)\b", low):
            return str(today + timedelta(days=2))

        # Check tomorrow variants (tomorrow, tommorow, tomorow, tommorrow, tmrw, 2moro, next day)
        if re.search(r"\b(?:tomorrow|tommorow|tomorow|tommorrow|tmrw|2moro|next\s+day)\b", low):
            return str(today + timedelta(days=1))

        # Check today variants
        if re.search(r"\b(?:today|tonight|2day|this\s+evening|this\s+afternoon)\b", low):
            return str(today)

        # Check weekday names like "monday", "next tuesday", "coming friday"
        weekdays = {
            "monday": 0, "mon": 0,
            "tuesday": 1, "tue": 1, "tues": 1,
            "wednesday": 2, "wed": 2,
            "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
            "friday": 4, "fri": 4,
            "saturday": 5, "sat": 5,
            "sunday": 6, "sun": 6,
        }
        for name, target_idx in weekdays.items():
            if re.search(rf"\b(?:next\s+|this\s+|coming\s+)?{name}\b", low):
                current_idx = today.weekday()
                days_ahead = (target_idx - current_idx + 7) % 7
                if days_ahead == 0:
                    days_ahead = 7
                return str(today + timedelta(days=days_ahead))

        # Check ISO format YYYY-MM-DD
        iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
        if iso_match:
            return f"{iso_match.group(1)}-{iso_match.group(2)}-{iso_match.group(3)}"

        # Check common DD-MM-YYYY or MM-DD-YYYY or DD/MM/YYYY
        dmy_match = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", text)
        if dmy_match:
            v1 = int(dmy_match.group(1))
            v2 = int(dmy_match.group(2))
            yr = int(dmy_match.group(3))
            if v1 > 12:
                day, month = v1, v2
            elif v2 > 12:
                day, month = v2, v1
            else:
                day, month = v1, v2
            return f"{yr:04d}-{month:02d}-{day:02d}"

        # Check named months e.g. "September 18, 2026" or "18 September 2026"
        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        for m_name, m_num in month_map.items():
            pattern1 = rf"\b{m_name}\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b"
            m1 = re.search(pattern1, text, re.IGNORECASE)
            if m1:
                day = int(m1.group(1))
                yr = int(m1.group(2)) if m1.group(2) else today.year
                return f"{yr:04d}-{m_num:02d}-{day:02d}"

            pattern2 = rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{m_name}(?:,?\s+(\d{{4}}))?\b"
            m2 = re.search(pattern2, text, re.IGNORECASE)
            if m2:
                day = int(m2.group(1))
                yr = int(m2.group(2)) if m2.group(2) else today.year
                return f"{yr:04d}-{m_num:02d}-{day:02d}"

        # Check standalone day of current month e.g. "on the 18th", "18th at 11 am"
        day_match = re.search(r"\b(?:on\s+(?:the\s+)?)?(\d{1,2})(?:st|nd|rd|th)\b", low)
        if day_match:
            d = int(day_match.group(1))
            if 1 <= d <= 31:
                try:
                    cand = today.replace(day=d)
                    if cand < today:
                        m = today.month + 1 if today.month < 12 else 1
                        y = today.year if today.month < 12 else today.year + 1
                        cand = today.replace(year=y, month=m, day=d)
                    return str(cand)
                except ValueError:
                    pass

        return ""

    @staticmethod
    def _extract_time(text: str) -> str:
        low = text.lower()
        word_hours = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
        }
        # Check word hours with qualifiers e.g. "eleven in the morning", "three pm"
        for word, num in word_hours.items():
            pattern_pm = rf"\b{word}\s*(?::\s*(\d{{2}}))?\s*(?:in the afternoon|in the evening|p\.?m\.?|pm|o'?clock in the afternoon|o'?clock in the evening)\b"
            m_pm = re.search(pattern_pm, low)
            if m_pm:
                mins = int(m_pm.group(1) or 0)
                hour = num + 12 if num != 12 else 12
                return f"{hour:02d}:{mins:02d}"

            pattern_am = rf"\b{word}\s*(?::\s*(\d{{2}}))?\s*(?:in the morning|a\.?m\.?|am|o'?clock in the morning)\b"
            m_am = re.search(pattern_am, low)
            if m_am:
                mins = int(m_am.group(1) or 0)
                hour = num if num != 12 else 0
                return f"{hour:02d}:{mins:02d}"

            pattern_oclock = rf"\b{word}\s+o'?clock\b"
            if re.search(pattern_oclock, low):
                hour = num
                if 1 <= hour <= 6:
                    hour += 12
                return f"{hour:02d}:00"

        # Check numeric with am/pm or a.m. / p.m.
        match = re.search(
            r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b",
            low,
        )
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
            merid = match.group(3).replace(".", "").lower()
            if merid == "pm" and hour != 12:
                hour += 12
            if merid == "am" and hour == 12:
                hour = 0
            return f"{hour:02d}:{minute:02d}"

        # Check numeric with "in the morning" / "in the afternoon" / "in the evening" / "o'clock"
        match_qual = re.search(
            r"\b(\d{1,2})(?::(\d{2}))?\s*(?:in the morning|in the afternoon|in the evening|at night|o'?clock)\b",
            low,
        )
        if match_qual:
            hour = int(match_qual.group(1))
            minute = int(match_qual.group(2) or 0)
            if "afternoon" in low or "evening" in low or "night" in low:
                if hour < 12:
                    hour += 12
            elif "morning" in low:
                if hour == 12:
                    hour = 0
            else:
                if 1 <= hour <= 6:
                    hour += 12
            return f"{hour:02d}:{minute:02d}"

        # Check "at <hour>" in scheduling context e.g. "at 11", "at 11:00"
        match_at = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\b", low)
        if match_at:
            hour = int(match_at.group(1))
            minute = int(match_at.group(2) or 0)
            if 1 <= hour <= 23:
                if 1 <= hour <= 6:
                    hour += 12
                return f"{hour:02d}:{minute:02d}"

        # Check standard 24h format HH:MM
        match_24 = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
        if match_24:
            return f"{int(match_24.group(1)):02d}:{int(match_24.group(2)):02d}"

        return ""
