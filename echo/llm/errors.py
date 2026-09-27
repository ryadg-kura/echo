"""LLM error types (kept dependency-free so tools can import them)."""


class LLMError(RuntimeError):
    """The LLM backend answered with an error."""


class LLMUnavailable(LLMError):
    """The LLM backend cannot be reached or the model is missing; use the offline fallback."""
