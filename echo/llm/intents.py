"""Deterministic intent matching.

Used as a fast path for camera commands ("what do you see", "read this") and as an
offline fallback router when Ollama is unavailable.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

Intent = tuple[str, dict[str, Any]]

# Tools that are always routed directly, even when the LLM is up.
FAST_PATH_TOOLS = frozenset({"describe_scene", "read_text"})

_Rule = tuple[re.Pattern[str], str, Callable[[re.Match[str]], dict[str, Any]]]


def _rule(pattern: str, tool: str, args: Callable[[re.Match[str]], dict[str, Any]]) -> _Rule:
    return re.compile(pattern, re.IGNORECASE), tool, args


_RULES: list[_Rule] = [
    _rule(r"\bwhat (?:do|can) you see\b|\bdescribe (?:the scene|what you see)\b",
          "describe_scene", lambda m: {}),
    _rule(r"^(?:can you |please )?read (?:this|that|the text|it)\b", "read_text", lambda m: {}),
    _rule(r"\bwhat(?:'s| is) the time\b|\bwhat time\b", "system_info", lambda m: {"what": "time"}),
    _rule(r"\bwhat(?:'s| is) the date\b|\bwhat day\b|\btoday'?s date\b",
          "system_info", lambda m: {"what": "date"}),
    _rule(r"\bbattery\b", "system_info", lambda m: {"what": "battery"}),
    _rule(r"\bcpu\b|\bprocessor usage\b", "system_info", lambda m: {"what": "cpu"}),
    _rule(r"\bsystem (?:info|status)\b", "system_info", lambda m: {"what": "all"}),
    _rule(r"^(?:please )?(?:open|launch|start) (?P<app>[\w .+-]+?)[.!?]*$",
          "open_app", lambda m: {"app": m["app"].strip()}),
    _rule(r"^(?:search|google|look up)(?: the web)?(?: for)? (?P<q>.+?)[.?!]*$",
          "web_search", lambda m: {"query": m["q"]}),
    _rule(r"^(?:take|make|create|add) (?:a )?note(?: that| saying)?[:,]? (?P<c>.+)$",
          "notes", lambda m: {"action": "create", "title": _title(m["c"]), "content": m["c"]}),
    _rule(r"^(?:list|show) (?:my )?notes\b", "notes", lambda m: {"action": "list"}),
    _rule(r"^search (?:my )?notes for (?P<q>.+?)[.?!]*$",
          "notes", lambda m: {"action": "search", "query": m["q"]}),
    _rule(r"^summari[sz]e (?:the )?(?:pdf )?(?P<p>\S+\.pdf)\b",
          "summarize_pdf", lambda m: {"path": m["p"]}),
]


def _title(content: str) -> str:
    return " ".join(content.split()[:6])


def match_intent(text: str) -> Intent | None:
    cleaned = text.strip()
    # Notes search must win over generic web search.
    for pattern, tool, build in sorted(_RULES, key=lambda r: r[1] != "notes"):
        if match := pattern.search(cleaned):
            return tool, build(match)
    return None


_YES = {"yes", "yeah", "yep", "sure", "ok", "okay", "confirm", "confirmed", "do it", "go ahead",
        "please do", "affirmative", "y"}
_NO = {"no", "nope", "cancel", "stop", "don't", "do not", "never mind", "nevermind", "n"}


def parse_yes_no(text: str) -> bool | None:
    """Interpret a short confirmation answer; ``None`` when it is neither."""
    cleaned = re.sub(r"[^\w' ]", "", text.lower()).strip()
    if cleaned in _YES or any(cleaned.startswith(w + " ") for w in _YES if " " not in w):
        return True
    if cleaned in _NO or any(cleaned.startswith(w + " ") for w in _NO if " " not in w):
        return False
    return None
