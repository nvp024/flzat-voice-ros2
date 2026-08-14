from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np


class MotionState(str, Enum):
    """Stable states used by the motion-episode detector."""

    WARMING_UP = "WARMING_UP"
    IDLE = "IDLE"
    MOTION = "MOTION"
    SETTLING = "SETTLING"


@dataclass(frozen=True)
class MotionResult:
    """Result produced for each processed camera frame."""

    state: MotionState
    score: float
    mask: np.ndarray
    boxes: tuple[tuple[int, int, int, int], ...]
    event_started: bool = False
    event_finished: bool = False
    motion_start_ns: Optional[int] = None
    motion_end_ns: Optional[int] = None
    peak_score: float = 0.0


class MotionDetector:
    """Detect one scene-change event per motion episode."""

    def __init__(
        self,
        processing_width: int,
        pixel_threshold: int,
        min_motion_ratio: float,
        start_frames: int,
        end_s: float,
        settle_s: float,
        warmup_frames: int,
        blur_size: int,
    ) -> None:
        if processing_width < 32:
            raise ValueError("processing_width must be at least 32")
        if not 1 <= pixel_threshold <= 255:
            raise ValueError("pixel_threshold must be between 1 and 255")
        if not 0.0 < min_motion_ratio <= 1.0:
            raise ValueError("min_motion_ratio must be between 0 and 1")
        if start_frames < 1 or warmup_frames < 1:
            raise ValueError("frame counts must be at least one")
        if end_s <= 0.0 or settle_s < 0.0:
            raise ValueError("motion timing parameters are invalid")
        if blur_size < 1 or blur_size % 2 == 0:
            raise ValueError("blur_size must be a positive odd number")

        self._processing_width = processing_width
        self._pixel_threshold = pixel_threshold
        self._min_motion_ratio = min_motion_ratio
        self._start_frames = start_frames
        self._end_ns = int(end_s * 1_000_000_000)
        self._settle_ns = int(settle_s * 1_000_000_000)
        self._warmup_frames = warmup_frames
        self._blur_size = blur_size
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        self._state = MotionState.WARMING_UP
        self._previous_gray: Optional[np.ndarray] = None
        self._warmup_count = 0
        self._candidate_count = 0
        self._candidate_start_ns = 0
        self._last_change_ns = 0
        self._settle_start_ns = 0
        self._motion_start_ns = 0
        self._motion_end_ns = 0
        self._peak_score = 0.0

    @property
    def state(self) -> MotionState:
        return self._state

    def process(self, frame: np.ndarray, stamp_ns: int) -> MotionResult:
        gray, scale_x, scale_y = self._preprocess(frame)
        empty_mask = np.zeros_like(gray)

        if self._state == MotionState.WARMING_UP:
            self._previous_gray = gray
            self._warmup_count += 1
            if self._warmup_count >= self._warmup_frames:
                self._state = MotionState.IDLE
            return MotionResult(self._state, 0.0, empty_mask, ())

        difference = cv2.absdiff(gray, self._previous_gray)
        self._previous_gray = gray
        _, mask = cv2.threshold(
            difference,
            self._pixel_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.dilate(mask, self._kernel, iterations=2)
        changed_pixels = int(cv2.countNonZero(mask))
        score = changed_pixels / float(mask.size)
        changed = score >= self._min_motion_ratio
        boxes = self._boxes(mask, scale_x, scale_y)

        event_started = False
        event_finished = False
        result_start_ns: Optional[int] = None
        result_end_ns: Optional[int] = None
        result_peak = self._peak_score

        if self._state == MotionState.IDLE:
            if changed:
                if self._candidate_count == 0:
                    self._candidate_start_ns = stamp_ns
                self._candidate_count += 1
                if self._candidate_count >= self._start_frames:
                    self._state = MotionState.MOTION
                    self._motion_start_ns = self._candidate_start_ns
                    self._last_change_ns = stamp_ns
                    self._peak_score = score
                    event_started = True
                    result_start_ns = self._motion_start_ns
            else:
                self._candidate_count = 0

        elif self._state == MotionState.MOTION:
            self._peak_score = max(self._peak_score, score)
            if changed:
                self._last_change_ns = stamp_ns
            elif stamp_ns - self._last_change_ns >= self._end_ns:
                self._motion_end_ns = self._last_change_ns
                self._settle_start_ns = stamp_ns
                self._state = MotionState.SETTLING

        elif self._state == MotionState.SETTLING:
            if changed:
                self._state = MotionState.MOTION
                self._last_change_ns = stamp_ns
                self._peak_score = max(self._peak_score, score)
            elif stamp_ns - self._settle_start_ns >= self._settle_ns:
                result_start_ns = self._motion_start_ns
                result_end_ns = self._motion_end_ns
                result_peak = self._peak_score
                event_finished = True
                self._state = MotionState.IDLE
                self._candidate_count = 0
                self._peak_score = 0.0

        return MotionResult(
            state=self._state,
            score=score,
            mask=mask,
            boxes=boxes,
            event_started=event_started,
            event_finished=event_finished,
            motion_start_ns=result_start_ns,
            motion_end_ns=result_end_ns,
            peak_score=result_peak,
        )

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        height, width = frame.shape[:2]
        working_height = max(1, round(height * self._processing_width / width))
        resized = cv2.resize(
            frame,
            (self._processing_width, working_height),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self._blur_size, self._blur_size), 0)
        return gray, width / self._processing_width, height / working_height

    @staticmethod
    def _boxes(
        mask: np.ndarray,
        scale_x: float,
        scale_y: float,
    ) -> tuple[tuple[int, int, int, int], ...]:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        boxes = []
        minimum_area = mask.size * 0.001
        for contour in contours:
            if cv2.contourArea(contour) < minimum_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            boxes.append(
                (
                    round(x * scale_x),
                    round(y * scale_y),
                    round(width * scale_x),
                    round(height * scale_y),
                )
            )
        return tuple(boxes)
