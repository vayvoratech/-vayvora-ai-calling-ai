"""External tool execution and MCP protocol package."""

from src.tools.email_provider import (
    EmailProvider,
    EmailSendResult,
    MockEmailProvider,
    SMTPEmailProvider,
)
from src.tools.mcp_client import (
    BaseMCPClient,
    HttpMCPToolProvider,
    MockToolProvider,
)
from src.tools.registry import (
    REGISTERED_TOOLS,
    ToolDefinition,
    list_registered_tools,
)
from src.tools.schemas import (
    ToolResult,
    ValidatedToolRequest,
    VerificationStatus,
)
from src.tools.validation import (
    SUPPORTED_ACTIONS,
    ActionValidator,
)
from src.tools.verifier import ActionVerifier

__all__ = [
    "EmailProvider",
    "SMTPEmailProvider",
    "MockEmailProvider",
    "EmailSendResult",
    "BaseMCPClient",
    "MockToolProvider",
    "HttpMCPToolProvider",
    "ToolDefinition",
    "REGISTERED_TOOLS",
    "list_registered_tools",
    "ToolResult",
    "ValidatedToolRequest",
    "VerificationStatus",
    "SUPPORTED_ACTIONS",
    "ActionValidator",
    "ActionVerifier",
]
