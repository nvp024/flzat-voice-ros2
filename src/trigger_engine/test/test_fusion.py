from trigger_engine.fusion import (
    FusionCoordinator,
    MotionAction,
    VoiceWindow,
    windows_overlap,
)


def test_windows_overlap_with_tolerance() -> None:
    assert windows_overlap(100, 200, 190, 300)
    assert not windows_overlap(100, 200, 211, 300, tolerance_ns=10)
    assert windows_overlap(100, 200, 210, 300, tolerance_ns=10)


def test_voice_claims_pending_motion() -> None:
    coordinator = FusionCoordinator(1_000, 0)
    decision = coordinator.register_motion(7, 100, 300, 1_000, payload="frame")
    assert decision.action == MotionAction.PENDING

    motion = coordinator.register_voice(VoiceWindow(2, 200, 400))

    assert motion is not None
    assert motion.event_id == 7
    assert motion.payload == "frame"
    assert coordinator.take_due_motion(10_000) is None


def test_motion_matches_active_voice_and_late_duplicate_is_suppressed() -> None:
    coordinator = FusionCoordinator(1_000, 0)
    coordinator.register_voice(VoiceWindow(5, 100, 300))

    decision = coordinator.register_motion(8, 200, 250, 1_000)
    assert decision.action == MotionAction.MATCHED
    assert decision.voice_request_id == 5

    coordinator.complete_voice(5)
    duplicate = coordinator.register_motion(9, 150, 280, 2_000)
    assert duplicate.action == MotionAction.SUPPRESSED


def test_unrelated_motion_is_held_and_newest_replaces_oldest() -> None:
    coordinator = FusionCoordinator(1_000, 0)
    first = coordinator.register_motion(10, 100, 200, 1_000)
    second = coordinator.register_motion(11, 400, 500, 1_100)

    assert first.action == MotionAction.PENDING
    assert second.replaced_event_id == 10
    assert coordinator.take_due_motion(2_099) is None
    due = coordinator.take_due_motion(2_100)
    assert due is not None
    assert due.event_id == 11
