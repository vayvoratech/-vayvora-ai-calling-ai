from typing import Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.agents.prompts.system import SYSTEM_PROMPT
from src.agents.state import AgentState


class LLMNode:
    """
    Single Speech LLM execution layer for Vayvora AI.
    Converts grounded facts and tool outcomes into voice-optimized speech.
    """

    def __init__(self, llm: Any):
        self.llm = llm

    @staticmethod
    def _extract_text(content: Any) -> str:
        """Safely extract string content whether Gemini returns str or list of parts."""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif hasattr(item, "text"):
                    parts.append(item.text)
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        return str(content).strip()

    async def run(self, state: AgentState) -> AgentState:
        user_input = state.get("user_input", "").strip()

        if not user_input:
            return {
                **state,
                "response": "I'm sorry, I didn't hear anything.",
                "is_complete": True,
            }

        # Check if the query is a general out-of-scope conceptual or trivia question
        from src.agents.nodes.router_node import RouterNode
        if RouterNode.is_general_query(user_input):
            out_of_scope_reply = (
                "I'm sorry, I can't process that query. I can only assist with "
                "questions regarding Vayvora Technology's engineering services, "
                "company information, or calendar appointments."
            )
            updated_messages = list(state.get("messages", [])) + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": out_of_scope_reply},
            ]
            return {
                **state,
                "messages": updated_messages,
                "response": out_of_scope_reply,
                "is_complete": True,
                "error": None,
            }

        # Build Single Combined System Instruction
        system_blocks = [SYSTEM_PROMPT]

        memory_context = state.get("memory_context", [])
        if memory_context:
            memory_text = "\n".join(str(item) for item in memory_context)
            system_blocks.append(
                f"\n\nRelevant conversation memory:\n{memory_text}"
            )

        rag_context = state.get("rag_context", "")
        if rag_context:
            system_blocks.append(
                f"\n\nGROUNDING INSTRUCTION:\n"
                f"Source of truth knowledge base:\n{rag_context}\n"
                f"Adhere strictly to these factual details. Never invent or hallucinate information."
            )

        # Inject structured conversation slots (known caller information)
        slots = state.get("slots", {})
        if slots:
            known_details = []
            if slots.get("caller_name"):
                known_details.append(f"- Caller Name: {slots['caller_name']}")
            if slots.get("date_str"):
                known_details.append(f"- Appointment Date: {slots['date_str']}")
            if slots.get("time_str"):
                known_details.append(f"- Appointment Time: {slots['time_str']}")
            if slots.get("email"):
                known_details.append(f"- Email: {slots['email']}")
            if slots.get("mobile"):
                known_details.append(f"- Mobile: {slots['mobile']}")
            if slots.get("whatsapp_opt_in") is not None:
                known_details.append(f"- WhatsApp Confirmation Opt-in: {'Yes' if slots['whatsapp_opt_in'] else 'No'}")
            if known_details:
                system_blocks.append(
                    f"\n\nCURRENT CONVERSATION SLOTS (PREVIOUSLY PROVIDED CALLER INFORMATION):\n"
                    + "\n".join(known_details)
                    + "\nDo NOT ask for any of these details again since the caller has already provided them."
                )

        tool_result = state.get("tool_result")
        tool_input = state.get("tool_input", {})
        missing_fields = tool_input.get("missing_fields", [])

        if missing_fields and tool_result is not None:
            system_blocks.append(
                f"\n\nCRITICAL DIRECTIVE - REQUIRED APPOINTMENT DETAILS MISSING:\n"
                f"{tool_result}\n\n"
                f"STRICT BEHAVIOR MANDATE:\n"
                f"1. The appointment is NOT booked or scheduled yet because required contact details are missing.\n"
                f"2. DO NOT say or imply that the appointment has been scheduled, booked, or confirmed.\n"
                f"3. Acknowledge what the caller provided (like their name and time) and DIRECTLY ASK the caller for the missing information specified in the directive.\n"
                f"4. Keep your response to one or two friendly, spoken conversational sentences."
            )
        elif tool_result is not None:
            system_blocks.append(
                f"\n\nResult from external action/tool:\n{tool_result}\n"
                f"Summarize this outcome in a friendly, conversational spoken sentence. "
                f"Never read technical parameter keys, JSON structures, or database IDs."
            )

        combined_system_prompt = "\n".join(system_blocks)

        formatted_messages = [SystemMessage(content=combined_system_prompt)]

        for msg in state.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "tool":
                continue
            elif role == "user":
                formatted_messages.append(HumanMessage(content=str(content)))
            elif role == "assistant":
                formatted_messages.append(AIMessage(content=str(content)))

        if not formatted_messages or formatted_messages[-1].content != user_input:
            formatted_messages.append(HumanMessage(content=user_input))

        try:
            response = await self.llm.ainvoke(formatted_messages)
            raw_content = getattr(response, "content", response)
            content = self._extract_text(raw_content)

            if not content:
                content = "I'm sorry, I wasn't able to generate a response."

            updated_messages = list(state.get("messages", [])) + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": content},
            ]

            return {
                **state,
                "messages": updated_messages,
                "response": content,
                "is_complete": True,
                "error": None,
            }

        except Exception as exc:
            return {
                **state,
                "response": "I'm sorry, I'm having trouble processing that right now.",
                "is_complete": True,
                "error": str(exc),
            }
