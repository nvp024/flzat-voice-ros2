import cv2
import numpy as np

from vision_pipeline.motion_detector import MotionDetector, MotionState


def _detector() -> MotionDetector:
    return MotionDetector(
        processing_width=160,
        pixel_threshold=15,
        min_motion_ratio=0.01,
        start_frames=2,
        end_s=0.2,
        settle_s=0.1,
        warmup_frames=2,
        blur_size=5,
    )


def test_one_event_is_emitted_after_motion_settles() -> None:
    detector = _detector()
    still = np.zeros((120, 160, 3), dtype=np.uint8)
    changed_one = still.copy()
    cv2.rectangle(changed_one, (20, 30), (80, 100), (255, 255, 255), -1)
    changed_two = still.copy()
    cv2.rectangle(changed_two, (40, 30), (100, 100), (255, 255, 255), -1)

    detector.process(still, 0)
    detector.process(still, 100_000_000)
    assert detector.state == MotionState.IDLE

    first = detector.process(changed_one, 200_000_000)
    started = detector.process(changed_two, 300_000_000)
    assert not first.event_started
    assert started.event_started
    assert detector.state == MotionState.MOTION

    detector.process(still, 400_000_000)
    detector.process(still, 500_000_000)
    detector.process(still, 600_000_000)
    assert detector.state == MotionState.SETTLING
    finished = detector.process(still, 700_000_000)

    assert finished.event_finished
    assert finished.motion_start_ns == 200_000_000
    assert finished.motion_end_ns == 400_000_000
    assert detector.state == MotionState.IDLE
