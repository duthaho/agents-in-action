"""
Typed exceptions for the agent runtime.

Two categories matter, and they're handled very differently:

    ToolError        Raised inside a tool. Fed back to the LLM as an
                     observation ("ERROR: ..."). The LLM decides
                     whether to retry, pick a different tool, or give
                     up. The LLM *is* the retry policy for tools.

    LLMError         Transient LLM plumbing failures (timeouts, 5xx,
                     rate limits). Handled by the retry wrapper in
                     llm.py — the LLM never sees these. If retries
                     exhaust, we raise MaxRetriesExceeded.

    ApprovalDenied   User rejected a sensitive tool call. Fed back to
                     the LLM as an observation ("DENIED: ..."). Same
                     pattern as ToolError — the LLM replans without
                     the denied action.

Permanent LLM errors (BadRequestError, AuthenticationError) are NOT
wrapped — they propagate as-is. Retrying them is wasted money.
"""


class AgentError(Exception):
    """Base class for all runtime errors raised by this package."""


class ToolError(AgentError):
    def __init__(self, tool: str, message: str) -> None:
        self.tool = tool
        super().__init__(message)


class LLMError(AgentError):
    """Transient LLM failure (timeout, connection, 5xx, rate limit)."""


class MaxRetriesExceeded(AgentError):
    """Raised by the retry wrapper after exhausting attempts."""


class ApprovalDenied(AgentError):
    """Raised when the user denies a sensitive tool call."""
