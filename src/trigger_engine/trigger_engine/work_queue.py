from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class SubmitResult(Generic[T]):
    accepted: bool
    replaced: Optional[T] = None


class LatestPriorityWorkQueue(Generic[T]):
    """One active item plus one newest pending item, with optional priority."""

    def __init__(self) -> None:
        self._active = False
        self._pending: Optional[tuple[int, T]] = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    def submit(self, item: T, priority: int = 0) -> SubmitResult[T]:
        if self._pending is not None and self._pending[0] > priority:
            return SubmitResult(accepted=False)
        replaced = self._pending[1] if self._pending is not None else None
        self._pending = (priority, item)
        return SubmitResult(accepted=True, replaced=replaced)

    def begin_next(self) -> Optional[T]:
        if self._active or self._pending is None:
            return None
        _, item = self._pending
        self._pending = None
        self._active = True
        return item

    def complete(self) -> None:
        if not self._active:
            raise RuntimeError("Cannot complete work when no item is active")
        self._active = False
