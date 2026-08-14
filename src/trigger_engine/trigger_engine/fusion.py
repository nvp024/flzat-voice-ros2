from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


@dataclass(frozen=True)
class VoiceWindow:
    request_id: int
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class MotionWindow:
    event_id: int
    start_ns: int
    end_ns: int
    deadline_ns: int
    payload: Any = None


class MotionAction(str, Enum):
    PENDING = "pending"
    MATCHED = "matched"
    SUPPRESSED = "suppressed"


@dataclass(frozen=True)
class MotionDecision:
    action: MotionAction
    voice_request_id: Optional[int] = None
    replaced_event_id: Optional[int] = None


def windows_overlap(
    first_start_ns: int,
    first_end_ns: int,
    second_start_ns: int,
    second_end_ns: int,
    tolerance_ns: int = 0,
) -> bool:
    """Return whether two closed time windows overlap within a tolerance."""
    return (
        first_start_ns <= second_end_ns + tolerance_ns
        and second_start_ns <= first_end_ns + tolerance_ns
    )


class FusionCoordinator:
    """Bounded timestamp policy for voice-priority audio/visual fusion."""

    def __init__(
        self,
        motion_hold_ns: int,
        overlap_tolerance_ns: int,
        recent_voice_limit: int = 8,
    ) -> None:
        if motion_hold_ns < 0 or overlap_tolerance_ns < 0:
            raise ValueError("Fusion timing values cannot be negative")
        if recent_voice_limit < 1:
            raise ValueError("recent_voice_limit must be at least one")
        self._motion_hold_ns = motion_hold_ns
        self._overlap_tolerance_ns = overlap_tolerance_ns
        self._voices: dict[int, VoiceWindow] = {}
        self._recent_voices: deque[VoiceWindow] = deque(maxlen=recent_voice_limit)
        self._pending_motion: Optional[MotionWindow] = None

    def register_voice(self, voice: VoiceWindow) -> Optional[MotionWindow]:
        self._voices[voice.request_id] = voice
        motion = self._pending_motion
        if motion is not None and self._overlaps(voice, motion):
            self._pending_motion = None
            return motion
        return None

    def discard_voice(self, request_id: int) -> None:
        self._voices.pop(request_id, None)

    def complete_voice(self, request_id: int) -> None:
        voice = self._voices.pop(request_id, None)
        if voice is not None:
            self._recent_voices.append(voice)

    def register_motion(
        self,
        event_id: int,
        start_ns: int,
        end_ns: int,
        received_ns: int,
        payload: Any = None,
    ) -> MotionDecision:
        motion = MotionWindow(
            event_id=event_id,
            start_ns=start_ns,
            end_ns=end_ns,
            deadline_ns=received_ns + self._motion_hold_ns,
            payload=payload,
        )
        for voice in reversed(tuple(self._voices.values())):
            if self._overlaps(voice, motion):
                return MotionDecision(MotionAction.MATCHED, voice.request_id)
        if any(self._overlaps(voice, motion) for voice in self._recent_voices):
            return MotionDecision(MotionAction.SUPPRESSED)

        replaced_event_id = None
        if self._pending_motion is not None:
            replaced_event_id = self._pending_motion.event_id
        self._pending_motion = motion
        return MotionDecision(
            MotionAction.PENDING,
            replaced_event_id=replaced_event_id,
        )

    def take_due_motion(self, now_ns: int) -> Optional[MotionWindow]:
        motion = self._pending_motion
        if motion is None or now_ns < motion.deadline_ns:
            return None
        self._pending_motion = None
        return motion

    def _overlaps(self, voice: VoiceWindow, motion: MotionWindow) -> bool:
        return windows_overlap(
            voice.start_ns,
            voice.end_ns,
            motion.start_ns,
            motion.end_ns,
            self._overlap_tolerance_ns,
        )
