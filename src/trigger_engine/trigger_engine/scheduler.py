from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Optional, TypeVar


T = TypeVar("T")


def is_usable_transcript(transcript: str) -> bool:
    """Return whether final STT text contains meaningful local content."""

    text = transcript.strip()
    return bool(text) and any(character.isalnum() for character in text)


class VlmTaskState(str, Enum):
    PENDING = "PENDING"
    DISPATCHING = "DISPATCHING"
    ACTIVE = "ACTIVE"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    DONE = "DONE"


@dataclass
class VlmTask(Generic[T]):
    vlm_task_id: int
    event_type: str
    payload: T
    deadline_ns: int
    state: VlmTaskState = VlmTaskState.PENDING
    goal_handle: Any = None
    cancel_requested: bool = False


@dataclass(frozen=True)
class HeldResponse:
    vlm_task_id: int
    voice_id: int
    speech: str
    expires_at_ns: int


@dataclass(frozen=True)
class TaskSubmitResult(Generic[T]):
    accepted: bool
    replaced: Optional[VlmTask[T]] = None


@dataclass(frozen=True)
class BeginTaskResult(Generic[T]):
    task: Optional[VlmTask[T]] = None
    expired: Optional[VlmTask[T]] = None
    blocked_by_stt: bool = False


@dataclass(frozen=True)
class HoldResult:
    held: HeldResponse
    replaced: Optional[HeldResponse] = None


@dataclass(frozen=True)
class VoiceResolution:
    recognized: bool
    usable: bool
    still_blocked: bool
    released: Optional[HeldResponse] = None
    discarded: Optional[HeldResponse] = None


@dataclass(frozen=True)
class CancelTransition(Generic[T]):
    task: Optional[VlmTask[T]] = None
    newly_requested: bool = False


@dataclass(frozen=True)
class CancellationDispatch:
    vlm_task_id: int
    goal_handle: Any


class VlmScheduler(Generic[T]):
    """Bounded VLM scheduler state; caller provides external locking."""

    def __init__(
        self,
        held_response_ttl_ns: int,
        active_timeout_ns: int = 25_000_000_000,
    ) -> None:
        if held_response_ttl_ns <= 0:
            raise ValueError("held_response_ttl_ns must be positive")
        if active_timeout_ns <= 0:
            raise ValueError("active_timeout_ns must be positive")
        self._held_response_ttl_ns = held_response_ttl_ns
        self._active_timeout_ns = active_timeout_ns
        self.active_task: Optional[VlmTask[T]] = None
        self.pending_task: Optional[VlmTask[T]] = None
        self.held_response: Optional[HeldResponse] = None
        self._unresolved_voice_ids: list[int] = []

    @property
    def stt_unresolved(self) -> bool:
        return bool(self._unresolved_voice_ids)

    @property
    def unresolved_voice_ids(self) -> tuple[int, ...]:
        return tuple(self._unresolved_voice_ids)

    def register_voice(self, voice_id: int) -> None:
        if voice_id in self._unresolved_voice_ids:
            raise ValueError(f"voice_id {voice_id} is already unresolved")
        self._unresolved_voice_ids.append(voice_id)

    def resolve_voice(
        self,
        voice_id: int,
        usable: bool,
        now_ns: int,
    ) -> VoiceResolution:
        if voice_id not in self._unresolved_voice_ids:
            return VoiceResolution(
                recognized=False,
                usable=usable,
                still_blocked=self.stt_unresolved,
            )

        self._unresolved_voice_ids.remove(voice_id)
        held = self.held_response
        discarded = None
        if held is not None and now_ns >= held.expires_at_ns:
            discarded = held
            self.held_response = None
            held = None

        released = None
        if usable:
            if held is not None:
                discarded = held
            self.held_response = None
        elif held is not None and held.voice_id == voice_id:
            if self._unresolved_voice_ids:
                self.held_response = HeldResponse(
                    vlm_task_id=held.vlm_task_id,
                    voice_id=self._unresolved_voice_ids[0],
                    speech=held.speech,
                    expires_at_ns=held.expires_at_ns,
                )
            else:
                released = held
                self.held_response = None

        return VoiceResolution(
            recognized=True,
            usable=usable,
            still_blocked=self.stt_unresolved,
            released=released,
            discarded=discarded,
        )

    def hold_response(
        self,
        vlm_task_id: int,
        speech: str,
        now_ns: int,
    ) -> HoldResult:
        if not self._unresolved_voice_ids:
            raise RuntimeError("Cannot hold a response without unresolved STT")
        normalized_speech = speech.strip()
        if not normalized_speech:
            raise ValueError("Cannot hold an empty response")
        held = HeldResponse(
            vlm_task_id=vlm_task_id,
            voice_id=self._unresolved_voice_ids[0],
            speech=normalized_speech,
            expires_at_ns=now_ns + self._held_response_ttl_ns,
        )
        replaced = self.held_response
        self.held_response = held
        return HoldResult(held=held, replaced=replaced)

    def expire_held_response(self, now_ns: int) -> Optional[HeldResponse]:
        held = self.held_response
        if held is None or now_ns < held.expires_at_ns:
            return None
        self.held_response = None
        return held

    def submit(self, task: VlmTask[T]) -> TaskSubmitResult[T]:
        if task.state != VlmTaskState.PENDING:
            raise ValueError("Only PENDING VLM tasks can be submitted")
        new_priority = self._priority(task.event_type)
        if (
            self.pending_task is not None
            and self._priority(self.pending_task.event_type) > new_priority
        ):
            return TaskSubmitResult(accepted=False)
        replaced = self.pending_task
        if replaced is not None:
            replaced.state = VlmTaskState.DONE
            replaced.payload = None  # type: ignore[assignment]
        self.pending_task = task
        return TaskSubmitResult(accepted=True, replaced=replaced)

    def begin_next(self, now_ns: int) -> BeginTaskResult[T]:
        if self.active_task is not None:
            return BeginTaskResult()
        if self.stt_unresolved:
            return BeginTaskResult(blocked_by_stt=True)
        task = self.pending_task
        if task is None:
            return BeginTaskResult()
        self.pending_task = None
        if now_ns >= task.deadline_ns:
            task.state = VlmTaskState.DONE
            task.payload = None  # type: ignore[assignment]
            return BeginTaskResult(expired=task)
        task.deadline_ns = now_ns + self._active_timeout_ns
        task.state = VlmTaskState.DISPATCHING
        self.active_task = task
        return BeginTaskResult(task=task)

    def mark_active(self, vlm_task_id: int, goal_handle: Any) -> bool:
        task = self.active_task
        if (
            task is None
            or task.vlm_task_id != vlm_task_id
            or task.state
            not in {VlmTaskState.DISPATCHING, VlmTaskState.CANCEL_REQUESTED}
        ):
            return False
        task.goal_handle = goal_handle
        if task.state == VlmTaskState.DISPATCHING:
            task.state = VlmTaskState.ACTIVE
        return True

    def result_is_current(self, vlm_task_id: int, now_ns: int) -> bool:
        task = self.active_task
        return bool(
            task is not None
            and task.vlm_task_id == vlm_task_id
            and task.state == VlmTaskState.ACTIVE
            and now_ns < task.deadline_ns
        )

    def discard_pending(self) -> Optional[VlmTask[T]]:
        task = self.pending_task
        if task is None:
            return None
        self.pending_task = None
        task.state = VlmTaskState.DONE
        task.payload = None  # type: ignore[assignment]
        return task

    def request_active_cancel(self) -> CancelTransition[T]:
        task = self.active_task
        if task is None or task.state == VlmTaskState.DONE:
            return CancelTransition()
        if task.state == VlmTaskState.CANCEL_REQUESTED:
            return CancelTransition(task=task, newly_requested=False)
        if task.state not in {VlmTaskState.DISPATCHING, VlmTaskState.ACTIVE}:
            return CancelTransition(task=task, newly_requested=False)
        task.state = VlmTaskState.CANCEL_REQUESTED
        task.cancel_requested = True
        return CancelTransition(task=task, newly_requested=True)

    def request_cancel_if_expired(self, now_ns: int) -> CancelTransition[T]:
        task = self.active_task
        if (
            task is None
            or task.state not in {VlmTaskState.DISPATCHING, VlmTaskState.ACTIVE}
            or now_ns < task.deadline_ns
        ):
            return CancelTransition(task=task, newly_requested=False)
        return self.request_active_cancel()

    def take_cancellation_dispatch(self) -> Optional[CancellationDispatch]:
        task = self.active_task
        if (
            task is None
            or task.state != VlmTaskState.CANCEL_REQUESTED
            or not task.cancel_requested
            or task.goal_handle is None
        ):
            return None
        task.cancel_requested = False
        return CancellationDispatch(task.vlm_task_id, task.goal_handle)

    def complete(self, vlm_task_id: int) -> Optional[VlmTask[T]]:
        task = self.active_task
        if task is None or task.vlm_task_id != vlm_task_id:
            return None
        task.state = VlmTaskState.DONE
        task.goal_handle = None
        task.payload = None  # type: ignore[assignment]
        self.active_task = None
        return task

    @staticmethod
    def _priority(event_type: str) -> int:
        return 1 if event_type in {"voice", "voice_motion"} else 0
