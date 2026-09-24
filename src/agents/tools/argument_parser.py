import re
from datetime import datetime, timedelta
from typing import Any


class ToolArgumentParser:
    """
    Fast deterministic argument parser for voice-driven actions.
    Supports spoken word numbers ("double five", "triple zero", "nine"),
    spoken emails, and multi-turn parameter aggregation across conversation history.
    """

    DIGIT_WORDS = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "oh": "0", "o": "0"
    }

    def parse(
        self,
        tool_name: str,
        user_input: str,
        messages: list[dict] | None = None,
    ) -> dict[str, Any]:
        text = user_input.strip()

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

    @classmethod
    def normalize_spoken_digits(cls, text: str) -> str:
        """
        Converts spoken word numbers and multipliers into raw digit strings:
        e.g., 'nine double five double zero two seven two one nine' -> '9550027219'
        """
        low = text.lower()

        # Handle 'double X' and 'triple X'
        for word, digit in cls.DIGIT_WORDS.items():
            low = re.sub(rf"\bdouble\s+{word}\b", f"{digit}{digit}", low)
            low = re.sub(rf"\btriple\s+{word}\b", f"{digit}{digit}{digit}", low)

        # Map individual number words to digits
        tokens = re.split(r"(\W+)", low)
        normalized = []
        for tok in tokens:
            cleaned = tok.strip()
            if cleaned in cls.DIGIT_WORDS:
                normalized.append(cls.DIGIT_WORDS[cleaned])
            else:
                normalized.append(tok)

        return "".join(normalized)

    def _parse_calendar_event(
        self,
        text: str,
        all_user_texts: list[str],
        messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        combined_text = " ".join(all_user_texts)

        # Check if immediate preceding assistant message asked for caller's name
        name_was_asked = False
        if messages:
            for m in reversed(messages):
                if m.get("role") == "assistant":
                    c = m.get("content", "").lower()
                    if any(k in c for k in ("full name", "your name", "caller name", "who is calling")):
                        name_was_asked = True
                    break

        # Extract name
        name = ""
        for idx, u in enumerate(reversed(all_user_texts)):
            is_direct = (idx == 0 and name_was_asked)
            cand_name = self._extract_name(u, is_direct_name_response=is_direct)
            if cand_name:
                name = cand_name
                break

        base_title = self._extract_title(combined_text)
        title = f"{base_title} - {name}" if (name and name.lower() not in base_title.lower()) else base_title

        # Extract date
        date_str = ""
        for u in reversed(all_user_texts):
            cand_date = self._extract_date(u)
            if cand_date:
                date_str = cand_date
                break

        # Extract time
        time_str = ""
        for u in reversed(all_user_texts):
            cand_time = self._extract_time(u)
            if cand_time:
                time_str = cand_time
                break

        # Extract contact parameters
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

        # Staged parameter completion verification
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

        return {
            "title": title,
            "date_str": date_str,
            "time_str": time_str,
            "description": " ".join(desc_parts),
            "caller_name": name,
            "email": email,
            "mobile": mobile,
            "whatsapp_opt_in": bool(whatsapp_opt_in) if whatsapp_opt_in is not None else False,
            "missing_fields": missing_fields,
            "missing_stage": missing_stage,
            "is_complete": len(missing_fields) == 0,
        }

    @classmethod
    def _extract_email(cls, text: str) -> str:
        # 1. Standard literal email regex
        match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
        if match:
            return match.group(0)

        # 2. Spoken email normalization
        low = text.lower()
        if "email" in low or "at the rate" in low or "gmail" in low:
            cleaned = low.replace(" at the rate ", "@").replace(" at ", "@")
            cleaned = cleaned.replace(" dot ", ".").replace(" point ", ".")
            cleaned = re.sub(r"\b(one|two|three|four|five|six|seven|eight|nine|zero)\b", 
                             lambda m: cls.DIGIT_WORDS.get(m.group(0), m.group(0)), cleaned)
            cleaned = re.sub(r"\s+", "", cleaned)
            # Fix trailing phonetic slips (e.g. '.comf' -> '.com')
            cleaned = re.sub(r"\.com[a-z]*$", ".com", cleaned)

            cand = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", cleaned)
            if cand:
                return cand.group(0)

            # Fallback if the user stated an email username without the domain
            user_part = re.search(r"(?:email(?:address)?is)?([a-z0-9._-]+)", cleaned)
            if user_part:
                val = user_part.group(1).replace("email", "").replace("address", "").replace("is", "").strip()
                if val and len(val) >= 3 and not any(w in val for w in ("ok", "yes", "none")):
                    return f"{val}@gmail.com"

        return ""

    @classmethod
    def _extract_phone(cls, text: str) -> str:
        # Convert all spoken digit words and multipliers into numeric digits
        normalized = cls.normalize_spoken_digits(text)
        digits_only = re.sub(r"\D", "", normalized)

        # Indian phone numbers: match 10 digits (with optional +91 or 0 prefix)
        match = re.search(r"(?:(?:\+|0{0,2})91[\s-]?)?([6-9]\d{9})\b", digits_only)
        if match:
            return match.group(1)

        # Fallback to standard 10-digit sequence
        gen_match = re.search(r"\b(\d{10})\b", digits_only)
        if gen_match:
            return gen_match.group(1)

        return ""

    @classmethod
    def _extract_name(cls, text: str, is_direct_name_response: bool = False) -> str:
        common_non_name_tokens = {
            "ok", "okay", "yes", "no", "not", "nope", "never", "prefer", "the", "a", "an",
            "whatsapp", "email", "mobile", "number", "phone", "address", "vayvora", "here",
            "ready", "interested", "calling", "consultation", "meeting", "call", "schedule",
            "appointment", "tomorrow", "today", "please", "thanks", "thank", "hello", "hi",
            "hey", "good", "want", "like", "proceed", "confirm", "booking", "book", "date",
            "time", "details", "two", "one", "nine", "five", "zero", "seven", "fight",
            "normal", "you", "cant", "can't", "nothing", "stop"
        }

        # Remove trailing temporal descriptors before evaluating name tokens
        cleaned_text = re.sub(
            r"\b(tomorrow|tommorow|today|tonight|yesterday|at\s+\d+|in\s+the\s+afternoon|in\s+the\s+morning|pm|am)\b.*$",
            "",
            text,
            flags=re.IGNORECASE
        ).strip()

        # Check explicit markers: "My name is [Name]" or "I am [Name]"
        match = re.search(r"(?:my\s+name\s+is|name\s+is|i\s+am|this\s+is)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})", cleaned_text, re.IGNORECASE)
        if match:
            cand = match.group(1).strip()
            if not any(w.lower() in common_non_name_tokens for w in cand.split()):
                return cand.title()

        # Parse purely alphabetic tokens
        words = [w for w in cleaned_text.split() if re.match(r"^[A-Za-z]+$", w)]
        name_words = [w for w in words if w.lower() not in {"yeah", "yes", "ok", "okay", "it", "its", "is", "can", "cant", "you"}]

        # Guard: If number words appear in the phrase, it is part of a phone or time dictation, not a name
        if any(w.lower() in cls.DIGIT_WORDS for w in name_words):
            return ""

        if 2 <= len(name_words) <= 4:
            if not any(w.lower() in common_non_name_tokens for w in name_words):
                return " ".join(name_words).title()

        if len(name_words) == 1 and is_direct_name_response:
            if name_words[0].lower() not in common_non_name_tokens:
                return name_words[0].title()

        return ""

    @staticmethod
    def _extract_title(text: str) -> str:
        low = text.lower()
        if "with our head" in low or "with head" in low or "head of" in low:
            return "Meeting with Head of Technology"
        if "assessment" in low:
            return "Assessment"
        if "discovery" in low:
            return "Discovery Call"
        if "consultation" in low:
            return "Consultation"
        return "Consultation Meeting"

    @staticmethod
    def _extract_date(text: str) -> str:
        today = datetime.now().date()
        low = text.lower()

        if re.search(r"\b(?:day\s+after\s+(?:tomorrow|tommorow|tomorow|tommorrow|tmrw)|overmorrow)\b", low):
            return str(today + timedelta(days=2))

        if re.search(r"\b(?:tomorrow|tommorow|tomorow|tommorrow|tmrw|2moro|next\s+day)\b", low):
            return str(today + timedelta(days=1))

        if re.search(r"\b(?:today|tonight|2day|this\s+evening|this\s+afternoon)\b", low):
            return str(today)

        weekdays = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6
        }
        for name, target_idx in weekdays.items():
            if re.search(rf"\b(?:next\s+|this\s+|coming\s+)?{name}\b", low):
                current_idx = today.weekday()
                days_ahead = (target_idx - current_idx + 7) % 7
                return str(today + timedelta(days=days_ahead if days_ahead != 0 else 7))

        iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
        if iso_match:
            return iso_match.group(0)

        dmy_match = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", text)
        if dmy_match:
            d1, d2, yr = int(dmy_match.group(1)), int(dmy_match.group(2)), int(dmy_match.group(3))
            day, month = (d1, d2) if d1 > 12 else (d2, d1) if d2 > 12 else (d1, d2)
            return f"{yr:04d}-{month:02d}-{day:02d}"

        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12
        }
        for m_name, m_num in month_map.items():
            m = re.search(rf"\b{m_name}\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b", text, re.IGNORECASE)
            if m:
                d = int(m.group(1))
                y = int(m.group(2)) if m.group(2) else today.year
                return f"{y:04d}-{m_num:02d}-{d:02d}"

        return ""

    @staticmethod
    def _extract_time(text: str) -> str:
        low = text.lower()
        word_hours = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12
        }
        for word, num in word_hours.items():
            pattern_pm = rf"\b{word}\s*(?:in the afternoon|in the evening|pm|p\.m\.|o'?clock in the afternoon)\b"
            if re.search(pattern_pm, low):
                hour = num + 12 if num != 12 else 12
                return f"{hour:02d}:00"

            pattern_am = rf"\b{word}\s*(?:in the morning|am|a\.m\.|o'?clock in the morning)\b"
            if re.search(pattern_am, low):
                hour = num if num != 12 else 0
                return f"{hour:02d}:00"

            if re.search(rf"\b{word}\s+o'?clock\b", low):
                hour = num + 12 if 1 <= num <= 6 else num
                return f"{hour:02d}:00"

        match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b", low)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
            merid = match.group(3).replace(".", "").lower()
            if merid == "pm" and hour != 12:
                hour += 12
            elif merid == "am" and hour == 12:
                hour = 0
            return f"{hour:02d}:{minute:02d}"

        match_at = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\b", low)
        if match_at:
            hour = int(match_at.group(1))
            minute = int(match_at.group(2) or 0)
            if 1 <= hour <= 6:
                hour += 12
            return f"{hour:02d}:{minute:02d}"

        match_24 = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
        if match_24:
            return f"{int(match_24.group(1)):02d}:{int(match_24.group(2)):02d}"

        return ""

    @staticmethod
    def _extract_whatsapp_pref(text: str, last_assistant: str = "") -> bool | None:
        t = text.lower()
        if "whatsapp" in t:
            if any(neg in t for neg in ("no", "not", "dont", "don't", "skip", "avoid", "only email")):
                return False
            if any(pos in t for pos in ("yes", "sure", "ok", "okay", "send", "fine", "please", "prefer")):
                return True
            return True

        if any(eo in t for eo in ("only email", "email only", "no, only email", "only send to email")):
            return False

        if "whatsapp" in last_assistant.lower():
            if re.search(r"\b(no|nope|never|not really|dont|don't|skip|avoid|no thanks|only email)\b", t):
                return False
            if re.search(r"\b(yes|sure|yeah|yep|ok|okay|fine|please|please do|certainly|absolutely)\b", t):
                return True

        return None

    def _parse_month_view(self, text: str, all_user_texts: list[str]) -> dict[str, Any]:
        combined = " ".join(all_user_texts).lower()
        now = datetime.now()
        year, month = now.year, now.month

        months = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
        }
        for name, num in months.items():
            if name in combined:
                month = num
                break

        if "next month" in combined:
            month = 1 if month == 12 else month + 1
            year = year + 1 if month == 1 else year

        return {"year": year, "month": month}

    def _parse_email(self, text: str, all_user_texts: list[str]) -> dict[str, Any]:
        combined = " ".join(all_user_texts)
        to = self._extract_email(text) or self._extract_email(combined)

        subject = ""
        sub_match = re.search(r"(?:subject|regarding|topic)\s+(?:is|to|about)?\s*[\"']?([^\"'\n,]+)[\"']?", combined, re.IGNORECASE)
        if sub_match:
            subject = sub_match.group(1).strip()

        raw_content = ""
        body_match = re.search(r"(?:with\s+message|saying|message\s+is|content\s+is|say\s+that)\s*[\"']?([^\"'\n]+)[\"']?", text, re.IGNORECASE)
        if body_match:
            raw_content = body_match.group(1).strip()
        else:
            topic_match = re.search(r"(?:about|regarding|concerning)\s+([^\"'\n,]+)", text, re.IGNORECASE)
            if topic_match:
                topic = topic_match.group(1).strip()
                if not subject:
                    subject = f"Inquiry regarding {topic.title()}"
                raw_content = f"Thank you for contacting Vayvora Technology. We are following up regarding {topic}."

        name = self._extract_name(text) or self._extract_name(combined)
        recipient_greeting = f"Dear {name}," if name else "Hello,"

        body = (
            f"{recipient_greeting}\n\n{raw_content}\n\nBest regards,\nVayvora Technology Team\nhttps://www.vayvoratech.com\ninfo@vayvoratech.com"
            if raw_content else ""
        )

        missing_fields = []
        if not to:
            missing_fields.append("recipient email address")
        if not body:
            missing_fields.append("email message or purpose")

        return {
            "to": to,
            "subject": subject or "Message from Vayvora Technology",
            "body": body,
            "missing_fields": missing_fields,
            "is_complete": len(missing_fields) == 0,
        }

    def _parse_whatsapp(self, text: str, all_user_texts: list[str]) -> dict[str, Any]:
        combined = " ".join(all_user_texts)
        phone = self._extract_phone(text) or self._extract_phone(combined)

        raw_content = ""
        body_match = re.search(r"(?:with\s+message|saying|message\s+is|content\s+is|say\s+that)\s*[\"']?([^\"'\n]+)[\"']?", text, re.IGNORECASE)
        if body_match:
            raw_content = body_match.group(1).strip()
        else:
            topic_match = re.search(r"(?:about|regarding|concerning)\s+([^\"'\n,]+)", text, re.IGNORECASE)
            if topic_match:
                raw_content = f"Hello from Vayvora Technology! We are reaching out regarding {topic_match.group(1).strip()}."

        missing_fields = []
        if not phone:
            missing_fields.append("phone number")
        if not raw_content:
            missing_fields.append("WhatsApp message content")

        return {
            "phone": phone,
            "message": f"Hello from Vayvora Technology! {raw_content}" if raw_content else "",
            "missing_fields": missing_fields,
            "is_complete": len(missing_fields) == 0,
        }