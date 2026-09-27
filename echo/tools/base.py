"""Tool plugin API and registry.

A tool is a class with a ``name``, a ``description``, a JSON-schema ``parameters`` dict and an
async ``run`` method. Drop a file into ``echo/tools/`` and it is discovered automatically.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar

from echo.core.config import EchoConfig
from echo.llm.errors import LLMUnavailable

if TYPE_CHECKING:
    from echo.core.events import EventBus
    from echo.llm.ollama_client import ChatBackend
    from echo.vision.service import VisionService

log = logging.getLogger(__name__)

ConfirmFn = Callable[[str], Awaitable[bool]]


@dataclass
class ToolContext:
    """Everything a tool may need, injected by the orchestrator."""

    config: EchoConfig
    llm: ChatBackend | None = None
    vision: VisionService | None = None
    bus: EventBus | None = None

    async def ask_llm(
        self,
        prompt: str,
        system: str | None = None,
        images: list[str] | None = None,
        model: str | None = None,
    ) -> str:
        if self.llm is None:
            raise LLMUnavailable("no LLM backend configured")
        return await self.llm.generate(
            model or self.config.llm.model, prompt, system=system, images=images
        )


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}
    requires_confirmation: ClassVar[bool] = False

    @abstractmethod
    async def run(self, ctx: ToolContext, *args: Any, **kwargs: Any) -> str:
        """Execute the tool and return a short, speakable result."""

    def confirmation_prompt(self, **kwargs: Any) -> str:
        return f"Should I run {self.name}?"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    @classmethod
    def discover(
        cls,
        package: str = "echo.tools",
        enabled: Iterable[str] = (),
        disabled: Iterable[str] = (),
    ) -> ToolRegistry:
        """Import every module of ``package`` and register the ``Tool`` subclasses it defines."""
        allow, deny = set(enabled), set(disabled)
        registry = cls()
        pkg = importlib.import_module(package)
        for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda i: i.name):
            if info.name.startswith("_") or info.name == "base":
                continue
            try:
                module = importlib.import_module(f"{package}.{info.name}")
            except Exception:
                log.exception("failed to import tool module %s", info.name)
                continue
            for tool_cls in _tool_classes(module):
                if (allow and tool_cls.name not in allow) or tool_cls.name in deny:
                    continue
                registry.register(tool_cls())
        return registry

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    async def execute(
        self,
        name: str,
        args: dict[str, Any] | None,
        ctx: ToolContext,
        confirm: ConfirmFn | None = None,
    ) -> str:
        tool = self.get(name)
        if tool is None:
            return f"Error: there is no tool named '{name}'."
        clean = _clean_args(tool, args or {})
        missing = [p for p in tool.parameters.get("required", []) if p not in clean]
        if missing:
            return f"Error: {name} needs the argument(s): {', '.join(missing)}."
        needs_confirm = tool.requires_confirmation or name in ctx.config.tools.confirm
        if needs_confirm and confirm is not None:
            if not await confirm(tool.confirmation_prompt(**clean)):
                return "Cancelled: the user did not confirm."
        try:
            return await tool.run(ctx, **clean)
        except Exception as exc:
            log.exception("tool %s failed", name)
            return f"Error running {name}: {exc}"


def _tool_classes(module: ModuleType) -> list[type[Tool]]:
    return [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj)
        and issubclass(obj, Tool)
        and obj is not Tool
        and not inspect.isabstract(obj)
        and obj.__module__ == module.__name__
    ]


def _clean_args(tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown/empty arguments: small models often add extras or send nulls."""
    known = tool.parameters.get("properties", {})
    return {
        key: value
        for key, value in args.items()
        if key in known and value not in (None, "")
    }
