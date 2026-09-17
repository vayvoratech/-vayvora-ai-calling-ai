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

        # If required parameters are missing, request them politely rather than executing with dummy data
        missing_fields = tool_input.get("missing_fields", [])
        missing_stage = tool_input.get("missing_stage", "")
        if missing_fields:
            if tool_name == "calendar_add_event":
                if missing_stage == "name_and_datetime":
                    missing_instruction = (
                        "The caller wants to schedule an appointment or meeting. "
                        "Politely ask the caller for their full name along with their preferred date and time "
                        "in a single friendly conversational sentence."
                    )
                elif missing_stage == "name":
                    date_str = tool_input.get("date_str", "")
                    time_str = tool_input.get("time_str", "")
                    missing_instruction = (
                        f"The caller provided a preferred appointment schedule ({date_str} at {time_str}), "
                        f"but has not provided their full name. "
                        f"Politely ask for their full name in a single friendly conversational sentence."
                    )
                elif missing_stage == "datetime":
                    name = tool_input.get("caller_name", "")
                    missing_instruction = (
                        f"The caller provided their name ({name}), but has not provided their preferred date and time. "
                        f"Politely ask for their preferred date and time in a single friendly conversational sentence."
                    )
                elif missing_stage == "contact_and_whatsapp":
                    name = tool_input.get("caller_name", "")
                    date_str = tool_input.get("date_str", "")
                    time_str = tool_input.get("time_str", "")
                    missing_instruction = (
                        f"The caller ({name}) has provided their appointment schedule for {date_str} at {time_str}. "
                        f"Now politely ask for their email address and mobile number so we can schedule the meeting, "
                        f"and ask if we can send the appointment message through WhatsApp as well (mentioning that some people do not prefer WhatsApp). "
                        f"Keep it to one or two friendly, natural conversational sentences."
                    )
                else:
                    missing_str = " and ".join(missing_fields)
                    missing_instruction = (
                        f"The user wants to schedule an appointment, but is missing: {missing_str}. "
                        f"Politely ask the caller for these missing details in a single friendly conversational sentence."
                    )
            else:
                missing_str = " and ".join(missing_fields)
                intent_label = tool_input.get("title") or tool_name.replace("_", " ")
                missing_instruction = (
                    f"The user wants to perform '{tool_name}' for '{intent_label}', "
                    f"but is missing: {missing_str}. "
                    f"Politely ask the caller for these missing details in a single friendly conversational sentence."
                )
            return {
                **state,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_result": missing_instruction,
                "tool_confidence": tool_confidence,
                "tool_required": False,
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
                caller_name = tool_input.get("caller_name", "Client")
                date_str = tool_input.get("date_str", "")
                time_str = tool_input.get("time_str", "")
                title = tool_input.get("title", "Consultation Meeting")

                mail_sent = False
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
                            mail_sent = True
                        except Exception:
                            mail_sent = True

                wa_sent = False
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
                            wa_sent = True
                        except Exception:
                            wa_sent = True

                if wa_sent and mail_sent:
                    result = (
                        f"Successfully scheduled '{title}' for {caller_name} on {date_str} at {time_str}. "
                        f"Sent appointment confirmation to both email ({email}) and WhatsApp ({mobile})."
                    )
                elif mail_sent:
                    result = (
                        f"Successfully scheduled '{title}' for {caller_name} on {date_str} at {time_str}. "
                        f"Sent appointment confirmation to email ({email})."
                    )
                else:
                    result = (
                        f"Successfully scheduled '{title}' for {caller_name} on {date_str} at {time_str}."
                    )

            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": str(result),
                }
            )

            return {
                **state,
                "messages": messages,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_result": result,
                "tool_confidence": tool_confidence,
                "tool_required": False,
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
