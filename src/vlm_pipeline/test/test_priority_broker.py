import threading

from vlm_pipeline.priority_broker import PriorityInferenceBroker, TicketState


def test_one_active_and_latest_equal_priority_pending():
    broker = PriorityInferenceBroker()
    active = broker.submit("active", 20, threading.Event())
    old_pending = broker.submit("old", 20, threading.Event())
    new_pending = broker.submit("new", 20, threading.Event())

    assert active.state == TicketState.ACTIVE
    assert old_pending.state == TicketState.REPLACED
    assert old_pending.cancel_event.is_set()
    assert new_pending.state == TicketState.QUEUED
    assert broker.pending_job_id == "new"

    broker.complete(active)
    assert new_pending.state == TicketState.ACTIVE
    assert new_pending.ready_event.is_set()


def test_higher_priority_preempts_active_and_lower_priority_is_rejected():
    broker = PriorityInferenceBroker()
    background = broker.submit("environment", 20, threading.Event())
    voice = broker.submit("voice", 30, threading.Event())
    motion = broker.submit("motion", 10, threading.Event())

    assert background.cancel_event.is_set()
    assert "higher-priority" in background.reason
    assert voice.state == TicketState.QUEUED
    assert motion.state == TicketState.REJECTED

    broker.complete(background)
    assert voice.state == TicketState.ACTIVE


def test_pending_ticket_can_be_cancelled_without_touching_active():
    broker = PriorityInferenceBroker()
    active = broker.submit("active", 30, threading.Event())
    pending = broker.submit("pending", 20, threading.Event())

    broker.cancel(pending, "caller cancelled")

    assert active.state == TicketState.ACTIVE
    assert pending.state == TicketState.CANCELLED
    assert broker.pending_job_id is None
