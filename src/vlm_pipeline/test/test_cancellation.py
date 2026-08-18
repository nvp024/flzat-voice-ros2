import threading

from vlm_pipeline.cancellation import (
    CancellationStoppingCriteria,
    GoalCancellationRegistry,
)


class _GoalId:
    def __init__(self, value: bytes) -> None:
        self.uuid = value


class _GoalHandle:
    def __init__(self, value: bytes) -> None:
        self.goal_id = _GoalId(value)


def test_cancel_callback_handle_signals_registered_generation() -> None:
    registry = GoalCancellationRegistry()
    execute_handle = _GoalHandle(b"one-goal")
    cancel_callback_handle = _GoalHandle(b"one-goal")
    token = threading.Event()
    registry.register(execute_handle, token)

    assert registry.request_cancel(cancel_callback_handle) is True
    assert token.is_set()

    registry.unregister(execute_handle, token)
    assert registry.active_count == 0


def test_stopping_criteria_observes_cancel_during_generation() -> None:
    registry = GoalCancellationRegistry()
    handle = _GoalHandle(b"active-generation")
    token = threading.Event()
    criterion = CancellationStoppingCriteria(token)
    generation_started = threading.Event()
    cancellation_observed = threading.Event()

    def _fake_generate() -> None:
        generation_started.set()
        while not criterion(None, None):
            token.wait(0.01)
        cancellation_observed.set()

    registry.register(handle, token)
    worker = threading.Thread(target=_fake_generate)
    worker.start()
    assert generation_started.wait(1.0)

    assert registry.request_cancel(handle) is True
    assert cancellation_observed.wait(1.0)
    worker.join(timeout=1.0)
    registry.unregister(handle, token)

    assert not worker.is_alive()
    assert registry.active_count == 0


def test_unregister_does_not_remove_a_replacement_token() -> None:
    registry = GoalCancellationRegistry()
    handle = _GoalHandle(b"goal")
    old_token = threading.Event()
    replacement_token = threading.Event()
    registry.register(handle, old_token)
    registry.unregister(handle, old_token)
    registry.register(handle, replacement_token)

    registry.unregister(handle, old_token)

    assert registry.active_count == 1
    assert registry.request_cancel(handle) is True
    assert replacement_token.is_set()
