from trigger_engine.work_queue import LatestPriorityWorkQueue


def test_queue_keeps_one_active_and_newest_pending() -> None:
    queue = LatestPriorityWorkQueue[str]()
    queue.submit("first")
    assert queue.begin_next() == "first"

    queue.submit("second")
    result = queue.submit("third")
    assert result.replaced == "second"
    assert queue.begin_next() is None

    queue.complete()
    assert queue.begin_next() == "third"


def test_voice_priority_cannot_be_replaced_by_motion() -> None:
    queue = LatestPriorityWorkQueue[str]()
    queue.submit("voice", priority=1)
    result = queue.submit("motion", priority=0)
    assert result.accepted is False
    assert queue.begin_next() == "voice"


def test_voice_replaces_pending_motion() -> None:
    queue = LatestPriorityWorkQueue[str]()
    queue.submit("motion", priority=0)
    result = queue.submit("voice", priority=1)
    assert result.accepted is True
    assert result.replaced == "motion"
    assert queue.begin_next() == "voice"
