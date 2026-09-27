"""A tiny asyncio publish/subscribe event bus.

Topics used across Echo:

- ``user.text``        {text, source}         text from keyboard or speech
- ``assistant.reply``  {text, kind}           reply to show and speak
- ``assistant.thinking`` {}                   a turn started
- ``tool.call``        {name, args}           the agent invoked a tool
- ``listen.start`` / ``listen.stop`` / ``listen.toggle``
- ``gesture``          {name}                 open_palm | fist | thumbs_up
- ``mood``             {emotion, confidence}
- ``module.status``    {module, state, detail}
- ``tts.start`` / ``tts.end``
- ``system.shutdown``
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

WILDCARD = "*"


@dataclass(frozen=True)
class Event:
    topic: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


Handler = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """Handlers run as independent tasks, so a slow or failing handler never blocks others."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._pending: set[asyncio.Task[None]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def subscribe(self, topic: str, handler: Handler) -> Callable[[], None]:
        """Subscribe to a topic (``"*"`` for all). Returns an unsubscribe function."""
        self._subscribers[topic].append(handler)

        def unsubscribe() -> None:
            if handler in self._subscribers[topic]:
                self._subscribers[topic].remove(handler)

        return unsubscribe

    def publish(self, topic: str, **data: Any) -> Event:
        """Publish from the event-loop thread. Handlers are scheduled, not awaited."""
        event = Event(topic, data)
        loop = asyncio.get_running_loop()
        self._loop = loop
        for handler in [*self._subscribers.get(topic, ()), *self._subscribers.get(WILDCARD, ())]:
            task = loop.create_task(self._dispatch(handler, event))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
        return event

    def publish_threadsafe(self, topic: str, **data: Any) -> None:
        """Publish from a worker thread (camera, audio callbacks...)."""
        if self._loop is None:
            raise RuntimeError("publish_threadsafe used before the bus saw the event loop")
        self._loop.call_soon_threadsafe(lambda: self.publish(topic, **data))

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def wait_for(
        self,
        topic: str,
        timeout: float | None = None,
        predicate: Callable[[Event], bool] | None = None,
    ) -> Event | None:
        """Wait for the next matching event; returns ``None`` on timeout."""
        future: asyncio.Future[Event] = asyncio.get_running_loop().create_future()

        def handler(event: Event) -> None:
            if not future.done() and (predicate is None or predicate(event)):
                future.set_result(event)

        unsubscribe = self.subscribe(topic, handler)
        try:
            return await asyncio.wait_for(future, timeout)
        except TimeoutError:
            return None
        finally:
            unsubscribe()

    async def drain(self) -> None:
        """Wait until every scheduled handler (including ones they publish) has finished."""
        while self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    @staticmethod
    async def _dispatch(handler: Handler, event: Event) -> None:
        try:
            result = handler(event)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("event handler %r failed on %s", handler, event.topic)
