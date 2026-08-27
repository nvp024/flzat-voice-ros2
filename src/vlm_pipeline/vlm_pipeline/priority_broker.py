from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TicketState(str, Enum):
    ACTIVE = "active"
    QUEUED = "queued"
    REJECTED = "rejected"
    REPLACED = "replaced"
    CANCELLED = "cancelled"
    COMPLETE = "complete"


@dataclass
class InferenceTicket:
    job_id: str
    priority: int
    cancel_event: threading.Event
    state: TicketState
    reason: str = ""
    ready_event: threading.Event = field(default_factory=threading.Event)


class PriorityInferenceBroker:
    """One active and one latest pending inference across all VLM actions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: Optional[InferenceTicket] = None
        self._pending: Optional[InferenceTicket] = None

    def submit(
        self,
        job_id: str,
        priority: int,
        cancel_event: threading.Event,
    ) -> InferenceTicket:
        with self._lock:
            ticket = InferenceTicket(
                job_id=job_id,
                priority=priority,
                cancel_event=cancel_event,
                state=TicketState.QUEUED,
            )
            if self._active is None:
                self._activate(ticket)
                return ticket

            if self._pending is not None and self._pending.priority > priority:
                ticket.state = TicketState.REJECTED
                ticket.reason = "a higher-priority request is already pending"
                ticket.ready_event.set()
                return ticket

            if self._pending is not None:
                replaced = self._pending
                replaced.state = TicketState.REPLACED
                replaced.reason = (
                    "superseded by a newer equal-or-higher priority request"
                )
                replaced.cancel_event.set()
                replaced.ready_event.set()

            self._pending = ticket
            if priority > self._active.priority:
                self._active.reason = "preempted by a higher-priority request"
                self._active.cancel_event.set()
            return ticket

    def cancel(self, ticket: InferenceTicket, reason: str) -> None:
        with self._lock:
            ticket.reason = reason
            ticket.cancel_event.set()
            if self._pending is ticket:
                self._pending = None
                ticket.state = TicketState.CANCELLED
                ticket.ready_event.set()

    def complete(self, ticket: InferenceTicket) -> None:
        with self._lock:
            if self._active is ticket:
                ticket.state = TicketState.COMPLETE
                self._active = None
                if self._pending is not None:
                    pending = self._pending
                    self._pending = None
                    self._activate(pending)
            elif self._pending is ticket:
                self._pending = None
                ticket.state = TicketState.CANCELLED
                ticket.ready_event.set()

    @property
    def active_job_id(self) -> Optional[str]:
        with self._lock:
            return None if self._active is None else self._active.job_id

    @property
    def pending_job_id(self) -> Optional[str]:
        with self._lock:
            return None if self._pending is None else self._pending.job_id

    def _activate(self, ticket: InferenceTicket) -> None:
        self._active = ticket
        ticket.state = TicketState.ACTIVE
        ticket.ready_event.set()
