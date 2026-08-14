from __future__ import annotations

import threading


class VlmJobGate:
    """Thread-safe one-request reservation used by the action server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reserved = False

    def try_reserve(self) -> bool:
        with self._lock:
            if self._reserved:
                return False
            self._reserved = True
            return True

    def release(self) -> None:
        with self._lock:
            self._reserved = False

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._reserved
