import os
from google import genai


class GeminiService:

    def __init__(
        self,
        model="gemini-3.5-flash-lite"
    ):
        self.model = model

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

        self.system_instruction = (
            "You are Vayvora AI, an ultrafast voice assistant. "
            "Reply directly in 1 short sentence. Never use markdown, lists, or asterisks. "
            "Prioritize instant speech output."
        )

        self.client = genai.Client(api_key=api_key)

    async def generate_stream_turn(self, history: list[dict]):
        """
        Fast stateless streaming with minimal prompt token overhead.
        """
        if len(history) <= 1:
            compiled_input = history[-1]["text"] if history else ""
        else:
            # Only retain last 2 exchanges to preserve sub-second TTFT
            context_window = history[-3:]
            lines = []
            for item in context_window[:-1]:
                speaker = "User" if item["role"] == "user" else "Assistant"
                lines.append(f"{speaker}: {item['text']}")
            lines.append(f"User: {context_window[-1]['text']}")
            lines.append("Assistant:")
            compiled_input = "\n".join(lines)

        stream = await self.client.aio.interactions.create(
            model=self.model,
            input=compiled_input,
            stream=True,
            system_instruction=self.system_instruction,
            generation_config={
                "thinking_level": "minimal",
                "max_output_tokens": 100
            }
        )

        async for event in stream:
            if getattr(event, "event_type", None) != "step.delta":
                continue

            delta = getattr(event, "delta", None)
            if delta is None or getattr(delta, "type", None) != "text":
                continue

            text = getattr(delta, "text", "")
            if text:
                yield text