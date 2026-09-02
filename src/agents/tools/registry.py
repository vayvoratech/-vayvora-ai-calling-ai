from typing import Any


class ToolRegistry:
    """
    Registry for tools available to the Vayvora AI Agent.

    Tools may be:
    - Local application tools
    - MCP adapter tools
    - Future external integrations

    The registry only manages tool discovery.
    Business logic remains inside the individual tool/MCP services.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, name: str, tool: Any) -> None:
        if not name:
            raise ValueError("Tool name cannot be empty.")

        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")

        if not hasattr(tool, "ainvoke"):
            raise TypeError(
                f"Tool '{name}' must provide an 'ainvoke' method."
            )

        self._tools[name] = tool

    def get(self, name: str) -> Any | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def clear(self) -> None:
        self._tools.clear()