import asyncio
import re
from typing import Any
from src.agents.state import AgentState
from src.agents.tools.tool_selector import ToolSelector


class ToolNode:
    """
    MCP / external tool execution node.

    Responsibilities:
    1. Select an MCP tool using semantic vector intent matching.
    2. Extract tool arguments deterministically.
    3. Execute the selected MCP tool.
    4. Store the structured tool outcome in AgentState.
    5. Pass the outcome to the speech LLM for natural spoken synthesis.
    """

    def __init__(
        self,
        tool_registry: Any = None,
        tool_selector: ToolSelector | None = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.tool_selector = tool_selector or ToolSelector()

    async def run(self, state: AgentState) -> AgentState:
        if self.tool_registry is None:
            return {
                **state,
                "tool_required": False,
                "error": "Tool registry is not configured.",
            }

        user_input = state.get("user_input", "").strip()
        messages = list(state.get("messages", []))
        tool_name = state.get("tool_name")
        tool_input = state.get("tool_input", {})
        tool_confidence = 0.0

        if not tool_name:
            selection = self.tool_selector.select(user_input, messages=messages)

            if selection is None:
                return {
                    **state,
                    "tool_required": False,
                    "error": "Unable to determine the required MCP tool from semantic intent.",
                }

            tool_name = selection.tool_name
            tool_input = selection.tool_input
            tool_confidence = selection.confidence

        # Accumulate conversation slots into persistent state memory
        slots = dict(state.get("slots") or {})
        for k in ("caller_name", "title", "date_str", "time_str", "email", "mobile", "whatsapp_opt_in"):
            val = tool_input.get(k)
            if val is not None and val != "":
                slots[k] = val

        # If required parameters are missing, request them politely rather than executing with dummy data
        missing_fields = tool_input.get("missing_fields", [])
        missing_stage = tool_input.get("missing_stage", "")
        if missing_fields:
            if tool_name == "calendar_add_event":
                name = tool_input.get("caller_name") or slots.get("caller_name") or ""
                if name.lower().strip() in ("hlo", "hlw", "helo", "hi", "hey", "ok", "okay", "client", "valued client"):
                    name = ""
                date_str = tool_input.get("date_str", "")
                time_str = tool_input.get("time_str", "")

                name_reference = f" {name}" if name else ""

                if missing_stage == "name_and_datetime":
                    spoken_prompt = (
                        "I would be glad to help schedule an appointment with our team. "
                        "Could you please share your full name along with your preferred date and time?"
                    )
                elif missing_stage == "name":
                    spoken_prompt = (
                        f"I have noted your appointment for {date_str} at {time_str}. "
                        "May I please have your full name to proceed with the booking?"
                    )
                elif missing_stage == "datetime":
                    spoken_prompt = (
                        f"Hello{name_reference}, what date and time would work best for your appointment?"
                    )
                elif missing_stage == "date":
                    spoken_prompt = (
                        f"I have noted your preferred time of {time_str}. Which date would you like to schedule the meeting for?"
                    )
                elif missing_stage == "time":
                    spoken_prompt = (
                        f"I have noted your preferred date of {date_str}. What time would you prefer for the meeting?"
                    )
                elif missing_stage == "contact_and_whatsapp":
                    spoken_prompt = (
                        f"Perfect{name_reference}, I have noted {date_str} at {time_str}. "
                        "Could you please provide your email address and mobile number so we can confirm the booking? "
                        "Also, would you like to receive the confirmation on WhatsApp as well?"
                    )
                elif missing_stage == "email":
                    spoken_prompt = (
                        "Thank you. Could you also please share your email address so we can send the meeting invitation?"
                    )
                elif missing_stage == "mobile":
                    spoken_prompt = (
                        "Thank you. Could you also please share your mobile number so we can finalize the booking?"
                    )
                elif missing_stage == "whatsapp":
                    spoken_prompt = (
                        "Thank you for sharing your details. Would you also like to receive your appointment confirmation on WhatsApp?"
                    )
                else:
                    missing_str = " and ".join(missing_fields)
                    spoken_prompt = (
                        f"To schedule your appointment, could you please provide your {missing_str}?"
                    )
            else:
                missing_str = " and ".join(missing_fields)
                intent_label = tool_input.get("title") or tool_name.replace("_", " ")
                spoken_prompt = (
                    f"To proceed with {intent_label}, could you please provide your {missing_str}?"
                )

            messages = [
                *state.get("messages", []),
                {"role": "assistant", "content": spoken_prompt},
            ]
            return {
                **state,
                "messages": messages,
                "response": spoken_prompt,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_result": spoken_prompt,
                "tool_confidence": tool_confidence,
                "tool_required": False,
                "slots": slots,
                "is_complete": True,
                "error": None,
            }

        tool = self.tool_registry.get(tool_name)

        if tool is None:
            return {
                **state,
                "tool_required": False,
                "error": f"Tool not found: {tool_name}",
            }

        # Filter out internal tracking keys before passing to tool function
        filtered_input = {
            k: v for k, v in tool_input.items()
            if k not in (
                "missing_fields", "missing_stage", "is_complete",
                "caller_name", "email", "mobile", "whatsapp_opt_in"
            )
        }

        try:
            result = await tool.ainvoke(filtered_input)

            # If appointment scheduling, also dispatch confirmation email and WhatsApp if opted in
            if tool_name == "calendar_add_event":
                email = tool_input.get("email")
                mobile = tool_input.get("mobile")
                whatsapp_opt_in = tool_input.get("whatsapp_opt_in", False)
                raw_name = (tool_input.get("caller_name") or slots.get("caller_name") or "").strip()
                if not raw_name or raw_name.lower() in ("hlo", "hlw", "helo", "hi", "hey", "ok", "okay", "client", "valued client", "unknown"):
                    caller_name = "Valued Client"
                else:
                    caller_name = raw_name

                date_str = tool_input.get("date_str", "")
                time_str = tool_input.get("time_str", "")
                title = tool_input.get("title", "Consultation Meeting")
                title = re.sub(r"\s*-\s*(hlo|hlw|helo|hi|hey|ok|okay)\b", "", title, flags=re.IGNORECASE).strip()
                if not title:
                    title = "Consultation Meeting"

                async def _send_mail() -> None:
                    if email:
                        mail_tool = self.tool_registry.get("mail_send")
                        if mail_tool:
                            try:
                                email_body = (
                                    f"Dear {caller_name},\n\n"
                                    f"Thank you for contacting Vayvora Technology. Your appointment has been successfully scheduled.\n\n"
                                    f"Appointment Details:\n"
                                    f"• Discussion Topic: {title}\n"
                                    f"• Date: {date_str}\n"
                                    f"• Time: {time_str}\n"
                                    f"• Registered Mobile: {mobile if mobile else 'Not provided'}\n"
                                    f"• Format: Consultation Call\n\n"
                                    f"If you need to reschedule or have any questions beforehand, please reply directly to this email or reach us at info@vayvoratech.com.\n\n"
                                    f"Best regards,\n"
                                    f"Vayvora Technology Team\n"
                                    f"https://www.vayvoratech.com\n"
                                    f"info@vayvoratech.com"
                                )
                                await mail_tool.ainvoke({
                                    "to": email,
                                    "subject": f"Appointment Confirmation - {title} | Vayvora Technology",
                                    "body": email_body,
                                })
                            except Exception:
                                pass

                async def _send_wa() -> None:
                    if whatsapp_opt_in and mobile:
                        wa_tool = self.tool_registry.get("whatsapp_send_message")
                        if wa_tool:
                            try:
                                wa_message = (
                                    f"Hello {caller_name}, your consultation meeting with Vayvora Technology "
                                    f"is confirmed for {date_str} at {time_str}. We look forward to speaking with you!"
                                )
                                await wa_tool.ainvoke({
                                    "phone": mobile,
                                    "message": wa_message,
                                })
                            except Exception:
                                pass

                # Dispatch email and WhatsApp in non-blocking background tasks
                # so the voice caller receives verbal confirmation immediately
                asyncio.create_task(_send_mail())
                if whatsapp_opt_in and mobile:
                    asyncio.create_task(_send_wa())

                if whatsapp_opt_in and mobile and email:
                    spoken_confirmation = (
                        f"Your consultation meeting for {title} has been successfully scheduled for {date_str} at {time_str}. "
                        f"I have sent the confirmation details to your email at {email} and to your WhatsApp."
                    )
                    result = (
                        f"Successfully scheduled '{title}' for {caller_name} on {date_str} at {time_str}. "
                        f"Sent appointment confirmation to both email ({email}) and WhatsApp ({mobile})."
                    )
                elif email:
                    spoken_confirmation = (
                        f"Your consultation meeting for {title} has been successfully scheduled for {date_str} at {time_str}. "
                        f"I have sent the confirmation details to your email at {email}."
                    )
                    result = (
                        f"Successfully scheduled '{title}' for {caller_name} on {date_str} at {time_str}. "
                        f"Sent appointment confirmation to email ({email})."
                    )
                else:
                    spoken_confirmation = (
                        f"Your consultation meeting for {title} has been successfully scheduled for {date_str} at {time_str}."
                    )
                    result = (
                        f"Successfully scheduled '{title}' for {caller_name} on {date_str} at {time_str}."
                    )

            elif tool_name == "mail_send":
                to_addr = tool_input.get("to", "your email")
                spoken_confirmation = f"I have sent the email to {to_addr}."
            elif tool_name == "whatsapp_send_message":
                phone_num = tool_input.get("phone", "your phone")
                spoken_confirmation = f"I have sent the WhatsApp message to {phone_num}."
            else:
                spoken_confirmation = str(result)

            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": str(result),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": spoken_confirmation,
                }
            )

            return {
                **state,
                "messages": messages,
                "response": spoken_confirmation,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_result": result,
                "tool_confidence": tool_confidence,
                "tool_required": False,
                "slots": slots,
                "is_complete": True,
                "error": None,
            }

        except Exception as exc:
            return {
                **state,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_required": False,
                "error": str(exc),
            }
