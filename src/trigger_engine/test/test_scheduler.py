from trigger_engine.scheduler import (
    VlmScheduler,
    VlmTask,
    VlmTaskState,
    is_usable_transcript,
)


def test_usable_transcript_requires_letter_or_digit() -> None:
    assert is_usable_transcript("No")
    assert is_usable_transcript("  2  ")
    assert is_usable_transcript("Xin chào")
    assert not is_usable_transcript("")
    assert not is_usable_transcript("   ")
    assert not is_usable_transcript("...?!")


def test_unresolved_stt_blocks_new_vlm_dispatch() -> None:
    scheduler = VlmScheduler[str](held_response_ttl_ns=100)
    scheduler.submit(VlmTask(1, "motion", "frame", deadline_ns=1_000))
    scheduler.register_voice(10)

    blocked = scheduler.begin_next(now_ns=10)

    assert blocked.task is None
    assert blocked.blocked_by_stt is True
    assert scheduler.pending_task is not None


def test_invalid_transcript_releases_held_response() -> None:
    scheduler = VlmScheduler[str](held_response_ttl_ns=100)
    scheduler.register_voice(10)
    scheduler.hold_response(4, "old answer", now_ns=20)

    result = scheduler.resolve_voice(10, usable=False, now_ns=30)

    assert result.recognized is True
    assert result.released is not None
    assert result.released.speech == "old answer"
    assert scheduler.held_response is None
    assert scheduler.stt_unresolved is False


def test_usable_transcript_discards_held_response() -> None:
    scheduler = VlmScheduler[str](held_response_ttl_ns=100)
    scheduler.register_voice(10)
    scheduler.hold_response(4, "obsolete answer", now_ns=20)

    result = scheduler.resolve_voice(10, usable=True, now_ns=30)

    assert result.discarded is not None
    assert result.discarded.speech == "obsolete answer"
    assert result.released is None
    assert scheduler.held_response is None


def test_newest_completed_response_replaces_older_hold() -> None:
    scheduler = VlmScheduler[str](held_response_ttl_ns=100)
    scheduler.register_voice(10)
    scheduler.hold_response(4, "older answer", now_ns=20)

    outcome = scheduler.hold_response(5, "newer answer", now_ns=30)

    assert outcome.replaced is not None
    assert outcome.replaced.vlm_task_id == 4
    assert scheduler.held_response is not None
    assert scheduler.held_response.vlm_task_id == 5
    assert scheduler.held_response.speech == "newer answer"


def test_resolved_voice_callback_cannot_resolve_hold_twice() -> None:
    scheduler = VlmScheduler[str](held_response_ttl_ns=100)
    scheduler.register_voice(10)
    scheduler.hold_response(4, "old answer", now_ns=20)
    first = scheduler.resolve_voice(10, usable=False, now_ns=30)

    late = scheduler.resolve_voice(10, usable=True, now_ns=40)

    assert first.released is not None
    assert late.recognized is False
    assert late.released is None
    assert late.discarded is None


def test_invalid_first_voice_keeps_hold_for_next_unresolved_voice() -> None:
    scheduler = VlmScheduler[str](held_response_ttl_ns=100)
    scheduler.register_voice(10)
    scheduler.register_voice(11)
    scheduler.hold_response(4, "old answer", now_ns=20)

    first = scheduler.resolve_voice(10, usable=False, now_ns=30)

    assert first.released is None
    assert first.still_blocked is True
    assert scheduler.held_response is not None
    assert scheduler.held_response.voice_id == 11

    second = scheduler.resolve_voice(11, usable=False, now_ns=40)
    assert second.released is not None
    assert second.released.speech == "old answer"


def test_expired_held_response_is_never_released() -> None:
    scheduler = VlmScheduler[str](held_response_ttl_ns=100)
    scheduler.register_voice(10)
    scheduler.hold_response(4, "too old", now_ns=20)

    result = scheduler.resolve_voice(10, usable=False, now_ns=120)

    assert result.released is None
    assert result.discarded is not None
    assert result.discarded.speech == "too old"
    assert scheduler.held_response is None


def test_voice_replaces_pending_motion_and_task_id_owns_callbacks() -> None:
    scheduler = VlmScheduler[str](held_response_ttl_ns=100)
    motion = VlmTask(1, "motion", "motion frame", deadline_ns=1_000)
    voice = VlmTask(2, "voice", "voice frame", deadline_ns=1_000)

    scheduler.submit(motion)
    outcome = scheduler.submit(voice)
    assert outcome.replaced is motion
    assert motion.state == VlmTaskState.DONE
    assert motion.payload is None

    begin = scheduler.begin_next(now_ns=10)
    assert begin.task is voice
    assert voice.state == VlmTaskState.DISPATCHING
    assert scheduler.mark_active(2, goal_handle="goal") is True
    assert scheduler.result_is_current(2, now_ns=10) is True
    assert scheduler.result_is_current(1, now_ns=10) is False

    assert scheduler.complete(1) is None
    assert scheduler.active_task is voice
    assert scheduler.complete(2) is voice
    assert scheduler.active_task is None


def test_pending_voice_cannot_be_replaced_by_motion() -> None:
    scheduler = VlmScheduler[str](held_response_ttl_ns=100)
    voice = VlmTask(1, "voice", "voice", deadline_ns=1_000)
    motion = VlmTask(2, "motion", "motion", deadline_ns=1_000)

    scheduler.submit(voice)
    outcome = scheduler.submit(motion)

    assert outcome.accepted is False
    assert scheduler.pending_task is voice


def test_expired_pending_task_is_released() -> None:
    scheduler = VlmScheduler[str](held_response_ttl_ns=100)
    task = VlmTask(1, "motion", "frame", deadline_ns=50)
    scheduler.submit(task)

    result = scheduler.begin_next(now_ns=50)

    assert result.task is None
    assert result.expired is task
    assert task.state == VlmTaskState.DONE
    assert task.payload is None


def test_active_cancellation_is_dispatched_exactly_once() -> None:
    scheduler = VlmScheduler[str](held_response_ttl_ns=100)
    task = VlmTask(1, "motion", "frame", deadline_ns=1_000)
    scheduler.submit(task)
    scheduler.begin_next(now_ns=10)
    assert scheduler.mark_active(1, goal_handle="goal")

    first = scheduler.request_active_cancel()
    repeated = scheduler.request_active_cancel()
    dispatch = scheduler.take_cancellation_dispatch()

    assert first.newly_requested is True
    assert repeated.newly_requested is False
    assert dispatch is not None
    assert dispatch.vlm_task_id == 1
    assert dispatch.goal_handle == "goal"
    assert scheduler.take_cancellation_dispatch() is None
    assert task.state == VlmTaskState.CANCEL_REQUESTED
    assert scheduler.result_is_current(1, now_ns=20) is False


def test_cancellation_during_dispatch_waits_for_goal_handle() -> None:
    scheduler = VlmScheduler[str](held_response_ttl_ns=100)
    task = VlmTask(1, "motion", "frame", deadline_ns=1_000)
    scheduler.submit(task)
    scheduler.begin_next(now_ns=10)

    transition = scheduler.request_active_cancel()

    assert transition.newly_requested is True
    assert scheduler.take_cancellation_dispatch() is None
    assert scheduler.mark_active(1, goal_handle="late-goal") is True
    assert task.state == VlmTaskState.CANCEL_REQUESTED
    dispatch = scheduler.take_cancellation_dispatch()
    assert dispatch is not None
    assert dispatch.goal_handle == "late-goal"


def test_active_deadline_requests_cancel_and_blocks_replacement() -> None:
    scheduler = VlmScheduler[str](
        held_response_ttl_ns=100,
        active_timeout_ns=50,
    )
    task = VlmTask(1, "motion", "frame", deadline_ns=1_000)
    scheduler.submit(task)
    scheduler.begin_next(now_ns=10)
    scheduler.mark_active(1, goal_handle="goal")
    scheduler.submit(VlmTask(2, "voice", "new", deadline_ns=1_000))

    assert scheduler.result_is_current(1, now_ns=59) is True
    transition = scheduler.request_cancel_if_expired(now_ns=60)

    assert transition.newly_requested is True
    assert task.state == VlmTaskState.CANCEL_REQUESTED
    assert scheduler.begin_next(now_ns=70).task is None
    assert scheduler.pending_task is not None


def test_discard_pending_releases_obsolete_payload() -> None:
    scheduler = VlmScheduler[str](held_response_ttl_ns=100)
    task = VlmTask(3, "voice", "obsolete", deadline_ns=1_000)
    scheduler.submit(task)

    discarded = scheduler.discard_pending()

    assert discarded is task
    assert task.state == VlmTaskState.DONE
    assert task.payload is None
    assert scheduler.pending_task is None
