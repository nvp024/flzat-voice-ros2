from __future__ import annotations

import threading
from typing import Any, Hashable


def _goal_key(goal_handle: Any) -> Hashable:
    goal_id = getattr(goal_handle, "goal_id", None)
    uuid = getattr(goal_id, "uuid", None)
    if uuid is None:
        return id(goal_handle)
    return bytes(uuid)


class GoalCancellationRegistry:
    """Thread-safe mapping from a ROS goal to its cooperative stop token."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens: dict[Hashable, threading.Event] = {}

    def register(
        self,
        goal_handle: Any,
        token: threading.Event,
    ) -> None:
        key = _goal_key(goal_handle)
        with self._lock:
            if key in self._tokens:
                raise RuntimeError("A cancellation token already exists for this goal")
            self._tokens[key] = token

    def request_cancel(self, goal_handle: Any) -> bool:
        key = _goal_key(goal_handle)
        with self._lock:
            token = self._tokens.get(key)
            if token is None:
                return False
            token.set()
            return True

    def unregister(
        self,
        goal_handle: Any,
        token: threading.Event,
    ) -> None:
        key = _goal_key(goal_handle)
        with self._lock:
            if self._tokens.get(key) is token:
                self._tokens.pop(key, None)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._tokens)


class CancellationStoppingCriteria:
    """Transformers-compatible criterion backed by a threading event."""

    def __init__(self, token: threading.Event) -> None:
        self._token = token

    def __call__(self, _input_ids, _scores, **_kwargs) -> bool:
        return self._token.is_set()
