from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, Optional, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class FrameCandidate(Generic[T]):
    """One bounded frame candidate with its ROS capture timestamp."""

    source: str
    stamp_ns: int
    frame: T


@dataclass(frozen=True)
class FrameSelection(Generic[T]):
    """Deterministic result of relevance-window frame selection."""

    selected: Optional[FrameCandidate[T]]
    window_start_ns: int
    window_end_ns: int
    used_baseline_fallback: bool
    rejected_sources: tuple[str, ...]


def relevance_window_end_ns(
    speech_end_ns: int,
    stt_result_ns: int,
    visual_after_ns: int,
) -> int:
    """Return the latest allowed ROS timestamp without waiting for the future."""

    if speech_end_ns < 0 or stt_result_ns < 0 or visual_after_ns < 0:
        raise ValueError("Frame relevance timestamps must be non-negative")
    return min(stt_result_ns, speech_end_ns + visual_after_ns)


def select_relevant_frame(
    candidates: Iterable[FrameCandidate[T]],
    baseline: Optional[FrameCandidate[T]],
    window_start_ns: int,
    window_end_ns: int,
) -> FrameSelection[T]:
    """Choose the newest in-window frame, or the retained baseline fallback."""

    candidates_tuple = tuple(candidates)
    valid = tuple(
        candidate
        for candidate in candidates_tuple
        if candidate.stamp_ns > 0
        and window_start_ns <= candidate.stamp_ns <= window_end_ns
    )
    source_priority = {"baseline": 0, "motion": 1, "refreshed": 2}
    selected = max(
        valid,
        key=lambda candidate: (
            candidate.stamp_ns,
            source_priority.get(candidate.source, -1),
        ),
        default=None,
    )
    used_fallback = False
    if selected is None and baseline is not None and baseline.stamp_ns > 0:
        selected = baseline
        used_fallback = True
    selected_identity = id(selected) if selected is not None else None
    rejected = tuple(
        candidate.source
        for candidate in candidates_tuple
        if id(candidate) != selected_identity
        and not (
            candidate.stamp_ns > 0
            and window_start_ns <= candidate.stamp_ns <= window_end_ns
        )
    )
    return FrameSelection(
        selected=selected,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        used_baseline_fallback=used_fallback,
        rejected_sources=rejected,
    )
