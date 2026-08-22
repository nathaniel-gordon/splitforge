"""Optional LLM layer: Claude API when credentials exist, deterministic fallback otherwise.

Canonical template — copied into agentic/RAG projects. Projects supply their own
task-specific offline fallbacks; this module only abstracts the live client.
"""
from __future__ import annotations

import os

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")


class LLMUnavailable(Exception):
    """Raised when no live LLM backend can be constructed."""


class ClaudeLLM:
    """Thin wrapper over the Anthropic SDK (messages API)."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 4096):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailable("anthropic SDK not installed (pip install anthropic)") from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, prompt: str, system: str = "") -> str:
        """Single-turn completion. Returns plain text ('' on safety refusal)."""
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        try:
            resp = self._client.messages.create(**kwargs)
        except self._anthropic.APIConnectionError as exc:
            raise LLMUnavailable(f"connection error: {exc}") from exc
        except self._anthropic.APIStatusError as exc:
            raise LLMUnavailable(f"API error {exc.status_code}: {exc.message}") from exc
        if getattr(resp, "stop_reason", None) == "refusal":
            return ""
        return "".join(b.text for b in resp.content if b.type == "text").strip()


def get_llm(model: str = DEFAULT_MODEL, max_tokens: int = 4096):
    """Return a ClaudeLLM if credentials exist, else None (caller uses offline fallback)."""
    try:
        return ClaudeLLM(model=model, max_tokens=max_tokens)
    except LLMUnavailable:
        return None
