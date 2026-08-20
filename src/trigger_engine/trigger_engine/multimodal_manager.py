from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from robot_interfaces.action import RunVlm, SpeechToText, TextToSpeech
from robot_interfaces.msg import MultimodalEvent, SpeechAudio, VisualEvent
from robot_interfaces.srv import GetFramesAround
from trigger_engine.fusion import (
    FusionCoordinator,
    MotionAction,
    MotionWindow,
    VoiceWindow,
)
from trigger_engine.frame_selection import (
    FrameCandidate,
    FrameSelection,
    relevance_window_end_ns,
    select_relevant_frame,
)
from trigger_engine.output_policy import speech_from_vlm_response
from trigger_engine.scheduler import (
    CancellationDispatch,
    HeldResponse,
    VlmScheduler,
    VlmTask,
    VlmTaskState,
    is_usable_transcript,
)
from trigger_engine.work_queue import LatestPriorityWorkQueue


def _stamp_to_ns(stamp: Time) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _time_from_ns(stamp_ns: int) -> Time:
    stamp = Time()
    stamp.sec = stamp_ns // 1_000_000_000
    stamp.nanosec = stamp_ns % 1_000_000_000
    return stamp


@dataclass
class FrameRequestSlot:
    frame: Optional[CompressedImage] = None
    requested: bool = False
    request_started_ns: int = 0
    done: bool = False
    error: str = ""


@dataclass
class VoiceRequest:
    request_id: int
    audio: SpeechAudio
    start_ns: int
    end_ns: int
    received_ns: int
    fusion_deadline_ns: int
    stt_state: str = "waiting"
    transcript: str = ""
    transcript_usable: bool = False
    stt_error: str = ""
    stt_result_ns: int = 0
    stt_completed_ns: int = 0
    baseline: FrameRequestSlot = field(default_factory=FrameRequestSlot)
    refreshed: FrameRequestSlot = field(default_factory=FrameRequestSlot)
    motion: Optional[VisualEvent] = None


@dataclass(frozen=True)
class PendingSpeech:
    vlm_task_id: int
    speech: str


class MultimodalManager(Node):
    """Fuse events, expose previews, and optionally run asynchronous VLM/TTS."""

    def __init__(self) -> None:
        super().__init__("multimodal_manager")
        self._cb_group = ReentrantCallbackGroup()
        self._lock = threading.RLock()

        self._declare_parameters()
        self._mode = self.get_parameter("mode").value.strip().lower()
        if self._mode not in {"mock", "vlm"}:
            raise ValueError("Parameter 'mode' must be 'mock' or 'vlm'.")
        motion_hold_s = self._positive_parameter("motion_hold_s", allow_zero=True)
        fusion_wait_s = self._positive_parameter("voice_fusion_wait_s", allow_zero=True)
        overlap_s = self._positive_parameter("overlap_tolerance_s", allow_zero=True)
        self._frame_before_s = self._positive_parameter("frame_before_s")
        self._frame_after_s = self._positive_parameter(
            "frame_after_s", allow_zero=True
        )
        self._voice_visual_after_ns = int(
            self._positive_parameter(
                "voice_visual_after_s",
                allow_zero=True,
            ) * 1_000_000_000
        )
        self._frame_timeout_ns = int(
            self._positive_parameter("frame_timeout_s") * 1_000_000_000
        )
        self._stt_timeout_ns = int(
            self._positive_parameter("stt_server_timeout_s") * 1_000_000_000
        )
        self._fusion_wait_ns = int(fusion_wait_s * 1_000_000_000)
        self._motion_vlm_cooldown_ns = int(
            self._positive_parameter(
                "motion_vlm_cooldown_s", allow_zero=True
            ) * 1_000_000_000
        )
        self._held_response_ttl_ns = int(
            self._positive_parameter("held_response_ttl_s") * 1_000_000_000
        )
        self._pending_motion_ttl_ns = int(
            self._positive_parameter("pending_motion_ttl_s") * 1_000_000_000
        )
        self._pending_voice_ttl_ns = int(
            self._positive_parameter("pending_voice_ttl_s") * 1_000_000_000
        )
        self._active_vlm_timeout_ns = int(
            self._positive_parameter("active_vlm_timeout_s") * 1_000_000_000
        )
        self._vlm_cancel_grace_ns = int(
            self._positive_parameter("vlm_cancel_grace_s") * 1_000_000_000
        )
        self._next_motion_vlm_ns = 0
        self._coordinator = FusionCoordinator(
            motion_hold_ns=int(motion_hold_s * 1_000_000_000),
            overlap_tolerance_ns=int(overlap_s * 1_000_000_000),
        )

        payload_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        preview_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._payload_pub = self.create_publisher(
            MultimodalEvent,
            "/multimodal/mock_input",
            payload_qos,
        )
        self._preview_pub = self.create_publisher(
            Image,
            "/multimodal/mock_frame",
            preview_qos,
        )
        self._summary_pub = self.create_publisher(
            String,
            "/multimodal/mock_summary",
            preview_qos,
        )
        self._vlm_response_pub = self.create_publisher(
            String,
            "/multimodal/vlm_response",
            payload_qos,
        )
        self._scheduler_status_pub = self.create_publisher(
            String,
            "/multimodal/scheduler_status",
            preview_qos,
        )
        self._audio_sub = self.create_subscription(
            SpeechAudio,
            "/audio_events",
            self._on_audio,
            10,
            callback_group=self._cb_group,
        )
        self._motion_sub = self.create_subscription(
            VisualEvent,
            "/vision/events",
            self._on_motion,
            10,
            callback_group=self._cb_group,
        )
        self._stt_client = ActionClient(
            self,
            SpeechToText,
            "/stt_action",
            callback_group=self._cb_group,
        )
        self._vlm_client = ActionClient(
            self,
            RunVlm,
            "/vlm/run",
            callback_group=self._cb_group,
        )
        self._tts_client = ActionClient(
            self,
            TextToSpeech,
            "/tts_action",
            callback_group=self._cb_group,
        )
        self._frame_client = self.create_client(
            GetFramesAround,
            "/vision/get_frames_around",
            callback_group=self._cb_group,
        )

        self._voices: dict[int, VoiceRequest] = {}
        self._active_stt_id: Optional[int] = None
        self._queued_stt_id: Optional[int] = None
        self._next_request_id = 1
        self._next_vlm_task_id = 1
        self._scheduler: VlmScheduler[MultimodalEvent] = VlmScheduler(
            self._held_response_ttl_ns,
            self._active_vlm_timeout_ns,
        )
        self._cancel_task_id: Optional[int] = None
        self._cancel_started_ns = 0
        self._cancel_degraded = False
        self._next_cancel_warning_ns = 0
        self._tts_queue: LatestPriorityWorkQueue[PendingSpeech] = (
            LatestPriorityWorkQueue()
        )
        self._timer = self.create_timer(
            0.05,
            self._on_timer,
            callback_group=self._cb_group,
        )
        if self._mode == "mock":
            self.get_logger().info(
                "Multimodal manager ready in mock mode — VLM and TTS are disabled."
            )
        else:
            self.get_logger().info(
                "Multimodal manager ready in VLM mode — asynchronous VLM and TTS enabled."
            )

    def _declare_parameters(self) -> None:
        self.declare_parameter("mode", "mock")
        self.declare_parameter("motion_hold_s", 1.0)
        self.declare_parameter("voice_fusion_wait_s", 1.0)
        self.declare_parameter("overlap_tolerance_s", 0.25)
        self.declare_parameter("frame_before_s", 2.0)
        self.declare_parameter("frame_after_s", 0.1)
        self.declare_parameter("voice_visual_after_s", 0.75)
        self.declare_parameter("frame_timeout_s", 2.0)
        self.declare_parameter("stt_server_timeout_s", 10.0)
        self.declare_parameter("motion_vlm_cooldown_s", 5.0)
        self.declare_parameter("held_response_ttl_s", 10.0)
        self.declare_parameter("pending_motion_ttl_s", 3.0)
        self.declare_parameter("pending_voice_ttl_s", 10.0)
        self.declare_parameter("active_vlm_timeout_s", 25.0)
        self.declare_parameter("vlm_cancel_grace_s", 3.0)

    def _positive_parameter(self, name: str, allow_zero: bool = False) -> float:
        value = self.get_parameter(name).get_parameter_value().double_value
        if value < 0.0 or (value == 0.0 and not allow_zero):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"Parameter '{name}' must be {qualifier}.")
        return value

    def _on_audio(self, audio: SpeechAudio) -> None:
        if audio.sample_rate <= 0 or not audio.audio_data:
            self.get_logger().warn("Ignoring invalid or empty audio event.")
            return
        end_ns = _stamp_to_ns(audio.header.stamp)
        if end_ns == 0:
            end_ns = self.get_clock().now().nanoseconds
            audio.header.stamp.sec = end_ns // 1_000_000_000
            audio.header.stamp.nanosec = end_ns % 1_000_000_000
        duration_ns = int(len(audio.audio_data) / audio.sample_rate * 1_000_000_000)
        start_ns = max(0, end_ns - duration_ns)
        now_ns = time.monotonic_ns()
        moved_pending_speech: Optional[PendingSpeech] = None
        replaced_held: Optional[HeldResponse] = None
        released_from_replaced: Optional[HeldResponse] = None

        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            self._scheduler.register_voice(request_id)
            voice = VoiceRequest(
                request_id=request_id,
                audio=audio,
                start_ns=start_ns,
                end_ns=end_ns,
                received_ns=now_ns,
                fusion_deadline_ns=now_ns + self._fusion_wait_ns,
            )
            self._voices[request_id] = voice
            matched = self._coordinator.register_voice(
                VoiceWindow(request_id, start_ns, end_ns)
            )
            if matched is not None:
                voice.motion = matched.payload

            moved_pending_speech = self._tts_queue.take_pending()
            if moved_pending_speech is not None:
                hold = self._scheduler.hold_response(
                    moved_pending_speech.vlm_task_id,
                    moved_pending_speech.speech,
                    now_ns,
                )
                replaced_held = hold.replaced

            if self._active_stt_id is None:
                self._active_stt_id = request_id
            else:
                replaced = self._queued_stt_id
                self._queued_stt_id = request_id
                if replaced is not None:
                    self._voices.pop(replaced, None)
                    self._coordinator.discard_voice(replaced)
                    resolution = self._scheduler.resolve_voice(
                        replaced,
                        usable=False,
                        now_ns=now_ns,
                    )
                    released_from_replaced = resolution.released
                    self.get_logger().warn(
                        f"Replaced queued voice request {replaced} with {request_id}."
                    )

        self.get_logger().info(
            f"Voice {request_id}: interval={self._format_ns(start_ns)}.."
            f"{self._format_ns(end_ns)}, samples={len(audio.audio_data)}; "
            "requesting baseline frame and STT; downstream dispatch paused."
        )
        if moved_pending_speech is not None:
            self.get_logger().info(
                f"Voice {request_id}: moved pending TTS response from VLM task "
                f"{moved_pending_speech.vlm_task_id} into the held slot."
            )
        if replaced_held is not None:
            self.get_logger().warn(
                f"Held response from VLM task {replaced_held.vlm_task_id} was "
                "replaced by a newer pending response."
            )
        if released_from_replaced is not None:
            self._enqueue_tts(
                released_from_replaced.speech,
                released_from_replaced.vlm_task_id,
            )
        self._request_frame(request_id, "baseline")
        self._pump_stt()

    def _on_motion(self, event: VisualEvent) -> None:
        if not event.frames:
            self.get_logger().warn(
                f"Ignoring motion event {event.event_id} without a frame."
            )
            return
        if len(event.frames) > 1:
            event.frames = [event.frames[0]]
            self.get_logger().warn(
                f"Motion event {event.event_id} contained multiple frames; "
                "retained only the first bounded candidate."
            )
        start_ns = _stamp_to_ns(event.motion_start)
        end_ns = _stamp_to_ns(event.motion_end)
        if start_ns == 0 or end_ns == 0:
            start_ns = end_ns = _stamp_to_ns(event.header.stamp)
        if start_ns > end_ns:
            start_ns, end_ns = end_ns, start_ns

        with self._lock:
            decision = self._coordinator.register_motion(
                event.event_id,
                start_ns,
                end_ns,
                time.monotonic_ns(),
                payload=event,
            )
            if decision.action == MotionAction.MATCHED:
                voice = self._voices.get(decision.voice_request_id)
                if voice is not None:
                    voice.motion = event

        if decision.action == MotionAction.MATCHED:
            self.get_logger().info(
                f"Motion {event.event_id} matched voice "
                f"{decision.voice_request_id}; voice has priority."
            )
        elif decision.action == MotionAction.SUPPRESSED:
            self.get_logger().info(
                f"Motion {event.event_id} suppressed: overlapping voice already published."
            )
        elif decision.replaced_event_id is not None:
            self.get_logger().warn(
                f"Motion {event.event_id} replaced pending motion "
                f"{decision.replaced_event_id}."
            )
        else:
            self.get_logger().info(
                f"Motion {event.event_id} held for voice fusion."
            )

    @staticmethod
    def _frame_slot(voice: VoiceRequest, source: str) -> FrameRequestSlot:
        if source == "baseline":
            return voice.baseline
        if source == "refreshed":
            return voice.refreshed
        raise ValueError(f"Unknown voice frame source: {source}")

    def _request_frame(self, request_id: int, source: str) -> None:
        if not self._frame_client.service_is_ready():
            return
        with self._lock:
            voice = self._voices.get(request_id)
            if voice is None:
                return
            slot = self._frame_slot(voice, source)
            if slot.done or slot.requested:
                return
            if source == "refreshed" and voice.stt_result_ns <= 0:
                return
            slot.requested = True
            slot.request_started_ns = time.monotonic_ns()
            target_ns = (
                voice.end_ns
                if source == "baseline"
                else relevance_window_end_ns(
                    voice.end_ns,
                    voice.stt_result_ns,
                    self._voice_visual_after_ns,
                )
            )
            request = GetFramesAround.Request()
            request.target_stamp = _time_from_ns(target_ns)
            request.before_s = self._frame_before_s
            request.after_s = self._frame_after_s if source == "baseline" else 0.0
            request.max_frames = 1
        try:
            future = self._frame_client.call_async(request)
            future.add_done_callback(
                lambda completed, request_id=request_id, source=source: (
                    self._on_frame_result(request_id, source, completed)
                )
            )
        except Exception as exc:
            with self._lock:
                voice = self._voices.get(request_id)
                if voice is not None:
                    self._frame_slot(voice, source).requested = False
            self.get_logger().error(
                f"Voice {request_id}: {source} frame request failed to send: {exc}"
            )

    def _on_frame_result(self, request_id: int, source: str, future) -> None:
        response = None
        error = ""
        try:
            response = future.result()
        except Exception as exc:
            error = str(exc)
        selected_frame = None
        frame_error = ""
        with self._lock:
            voice = self._voices.get(request_id)
            if voice is None:
                self.get_logger().debug(
                    f"Ignoring stale {source} frame callback for voice {request_id}."
                )
                return
            slot = self._frame_slot(voice, source)
            if slot.done:
                return
            slot.requested = False
            slot.done = True
            if response is not None and response.success and response.frames:
                slot.frame = response.frames[0]
                selected_frame = slot.frame
            else:
                slot.error = error or (
                    response.message if response is not None else "unknown service error"
                )
                frame_error = slot.error
        if selected_frame is not None:
            self.get_logger().info(
                f"Voice {request_id}: {source} candidate timestamp="
                f"{self._format_ns(_stamp_to_ns(selected_frame.header.stamp))}."
            )
        else:
            self.get_logger().warn(
                f"Voice {request_id}: no {source} frame candidate: {frame_error}"
            )

    def _pump_stt(self) -> None:
        now_ns = time.monotonic_ns()
        timed_out_id = None
        with self._lock:
            request_id = self._active_stt_id
            voice = self._voices.get(request_id) if request_id is not None else None
            if voice is None:
                self._active_stt_id = None
                self._promote_queued_locked()
                request_id = self._active_stt_id
                voice = self._voices.get(request_id) if request_id is not None else None
            if voice is None or voice.stt_state != "waiting":
                return
            if now_ns - voice.received_ns >= self._stt_timeout_ns:
                timed_out_id = voice.request_id
            elif not self._stt_client.server_is_ready():
                return
            else:
                voice.stt_state = "sending"
                goal = SpeechToText.Goal()
                goal.audio_packet = voice.audio

        if timed_out_id is not None:
            self._finish_stt(timed_out_id, "", "STT server timeout")
            return
        try:
            future = self._stt_client.send_goal_async(goal)
            future.add_done_callback(
                lambda completed, request_id=request_id: self._on_stt_goal(
                    request_id, completed
                )
            )
        except Exception as exc:
            self._finish_stt(request_id, "", f"STT send failed: {exc}")

    def _on_stt_goal(self, request_id: int, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._finish_stt(request_id, "", f"STT goal error: {exc}")
            return
        if not goal_handle.accepted:
            self._finish_stt(request_id, "", "STT goal rejected")
            return
        with self._lock:
            voice = self._voices.get(request_id)
            if voice is None:
                return
            voice.stt_state = "active"
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda completed, request_id=request_id: self._on_stt_result(
                request_id, completed
            )
        )

    def _on_stt_result(self, request_id: int, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self._finish_stt(request_id, "", f"STT result error: {exc}")
            return
        if response.status != GoalStatus.STATUS_SUCCEEDED:
            self._finish_stt(
                request_id,
                "",
                f"STT ended with status {response.status}",
            )
            return
        self._finish_stt(request_id, response.result.transcript.strip(), "")

    def _finish_stt(self, request_id: int, transcript: str, error: str) -> None:
        now_ns = time.monotonic_ns()
        stt_result_ns = self.get_clock().now().nanoseconds
        released: Optional[HeldResponse] = None
        discarded: Optional[HeldResponse] = None
        invalid_motion: Optional[VisualEvent] = None
        discarded_pending_vlm: Optional[VlmTask[MultimodalEvent]] = None
        discarded_pending_tts: Optional[PendingSpeech] = None
        cancel_dispatch: Optional[CancellationDispatch] = None
        cancel_task_id: Optional[int] = None
        usable = False
        with self._lock:
            voice = self._voices.get(request_id)
            if voice is None or voice.stt_state == "done":
                return
            usable = not error and is_usable_transcript(transcript)
            voice.stt_state = "done"
            voice.transcript = transcript
            voice.transcript_usable = usable
            voice.stt_error = error
            voice.stt_result_ns = stt_result_ns
            voice.stt_completed_ns = now_ns
            resolution = self._scheduler.resolve_voice(
                request_id,
                usable=usable,
                now_ns=now_ns,
            )
            released = resolution.released
            discarded = resolution.discarded
            if self._active_stt_id == request_id:
                self._active_stt_id = None
                self._promote_queued_locked()
            elif self._queued_stt_id == request_id:
                self._queued_stt_id = None

            if not usable:
                self._voices.pop(request_id, None)
                self._coordinator.discard_voice(request_id)
                invalid_motion = voice.motion
            else:
                discarded_pending_vlm = self._scheduler.discard_pending()
                discarded_pending_tts = self._tts_queue.take_pending()
                cancel_transition = self._scheduler.request_active_cancel()
                if cancel_transition.newly_requested and cancel_transition.task:
                    cancel_task_id = cancel_transition.task.vlm_task_id
                    self._start_cancel_diagnostic_locked(cancel_task_id, now_ns)
                cancel_dispatch = self._scheduler.take_cancellation_dispatch()

        if usable:
            self.get_logger().info(
                f"Voice {request_id}: usable final transcript confirmed: "
                f"'{transcript}'."
            )
        elif error:
            self.get_logger().error(f"Voice {request_id}: {error}.")
        else:
            self.get_logger().info(
                f"Voice {request_id}: final transcript is not usable; "
                "no voice VLM request will be created."
            )
        if discarded is not None:
            self.get_logger().info(
                f"Voice {request_id}: discarded held response from VLM task "
                f"{discarded.vlm_task_id}; it is obsolete or expired."
            )
        if discarded_pending_vlm is not None:
            self.get_logger().info(
                f"Voice {request_id}: discarded pending obsolete VLM task "
                f"{discarded_pending_vlm.vlm_task_id}."
            )
        if discarded_pending_tts is not None:
            self.get_logger().info(
                f"Voice {request_id}: discarded pending TTS response from VLM task "
                f"{discarded_pending_tts.vlm_task_id}."
            )
        if cancel_task_id is not None:
            self.get_logger().info(
                f"Voice {request_id}: requesting cancellation of obsolete VLM task "
                f"{cancel_task_id}."
            )
        if cancel_dispatch is not None:
            self._dispatch_vlm_cancel(cancel_dispatch)
        if released is not None:
            self.get_logger().info(
                f"Voice {request_id}: STT was not usable; releasing held response "
                f"from VLM task {released.vlm_task_id} to TTS."
            )
            self._enqueue_tts(released.speech, released.vlm_task_id)
        if invalid_motion is not None:
            self.get_logger().info(
                f"Voice {request_id}: returning matched motion "
                f"{invalid_motion.event_id} to normal fusion."
            )
            self._on_motion(invalid_motion)
        if usable:
            self._request_frame(request_id, "refreshed")
        self._pump_stt()
        self._pump_vlm()
        self._pump_tts()

    def _promote_queued_locked(self) -> None:
        if self._active_stt_id is None and self._queued_stt_id is not None:
            self._active_stt_id = self._queued_stt_id
            self._queued_stt_id = None

    def _on_timer(self) -> None:
        self._pump_frame_requests()
        self._pump_stt()
        with self._lock:
            expired_held = self._scheduler.expire_held_response(time.monotonic_ns())
        if expired_held is not None:
            self.get_logger().warn(
                f"Discarded expired held response from VLM task "
                f"{expired_held.vlm_task_id}."
            )
        self._monitor_vlm_cancellation()
        self._pump_vlm()
        self._pump_tts()
        now_ns = time.monotonic_ns()
        ready_voices: list[
            tuple[VoiceRequest, FrameSelection[CompressedImage]]
        ] = []
        dropped_voices: list[
            tuple[VoiceRequest, FrameSelection[CompressedImage]]
        ] = []
        motion: Optional[MotionWindow]

        with self._lock:
            motion = self._coordinator.take_due_motion(now_ns)
            for request_id, voice in list(self._voices.items()):
                selection = self._select_voice_frame(voice)
                fusion_ready = (
                    voice.motion is not None or now_ns >= voice.fusion_deadline_ns
                )
                candidates_done = voice.baseline.done and voice.refreshed.done
                if (
                    voice.stt_state == "done"
                    and voice.transcript_usable
                    and candidates_done
                    and selection.selected is not None
                    and fusion_ready
                ):
                    ready_voices.append((voice, selection))
                    self._voices.pop(request_id, None)
                    self._coordinator.complete_voice(request_id)
                elif (
                    voice.stt_state == "done"
                    and voice.transcript_usable
                    and candidates_done
                    and selection.selected is None
                    and fusion_ready
                ):
                    dropped_voices.append((voice, selection))
                    self._voices.pop(request_id, None)
                    self._coordinator.complete_voice(request_id)

        for voice, selection in ready_voices:
            selected = selection.selected
            assert selected is not None
            event_type = "voice_motion" if voice.motion is not None else "voice"
            reason = "speech_with_motion" if voice.motion is not None else "speech"
            self._log_voice_frame_selection(voice, selection)
            try:
                self._publish_event(
                    voice.audio.header.stamp,
                    event_type,
                    voice.transcript,
                    reason,
                    selected.frame,
                    selected.source,
                    voice.start_ns,
                    voice.end_ns,
                )
            finally:
                self._release_voice_frames(voice)
        for voice, selection in dropped_voices:
            self._log_voice_frame_selection(voice, selection)
            self.get_logger().error(
                f"Voice {voice.request_id} dropped: no relevant frame; "
                f"baseline_error='{voice.baseline.error}', "
                f"refreshed_error='{voice.refreshed.error}'."
            )
            self._release_voice_frames(voice)
        if motion is not None:
            event = motion.payload
            if event is not None and event.frames:
                self._publish_event(
                    event.header.stamp,
                    "motion",
                    "",
                    event.trigger_reason or "scene_change",
                    event.frames[0],
                    "motion",
                    motion.start_ns,
                    motion.end_ns,
                )

    def _pump_frame_requests(self) -> None:
        now_ns = time.monotonic_ns()
        requests: list[tuple[int, str]] = []
        with self._lock:
            for voice in self._voices.values():
                sources = [("baseline", voice.baseline, voice.received_ns)]
                if voice.stt_state == "done" and voice.transcript_usable:
                    sources.append(
                        ("refreshed", voice.refreshed, voice.stt_completed_ns)
                    )
                for source, slot, eligible_since_ns in sources:
                    if slot.done:
                        continue
                    if slot.requested:
                        if (
                            now_ns - slot.request_started_ns
                            >= self._frame_timeout_ns
                        ):
                            slot.requested = False
                            slot.done = True
                            slot.error = "frame service timeout"
                        continue
                    if now_ns - eligible_since_ns >= self._frame_timeout_ns:
                        slot.done = True
                        slot.error = "frame service unavailable"
                    else:
                        requests.append((voice.request_id, source))
        for request_id, source in requests:
            self._request_frame(request_id, source)

    def _select_voice_frame(
        self,
        voice: VoiceRequest,
    ) -> FrameSelection[CompressedImage]:
        candidates: list[FrameCandidate[CompressedImage]] = []
        baseline = self._candidate_from_frame("baseline", voice.baseline.frame)
        if baseline is not None:
            candidates.append(baseline)
        refreshed = self._candidate_from_frame("refreshed", voice.refreshed.frame)
        if refreshed is not None:
            candidates.append(refreshed)
        motion_frame = (
            voice.motion.frames[0]
            if voice.motion is not None and voice.motion.frames
            else None
        )
        motion = self._candidate_from_frame("motion", motion_frame)
        if motion is not None:
            candidates.append(motion)
        window_end_ns = relevance_window_end_ns(
            voice.end_ns,
            voice.stt_result_ns,
            self._voice_visual_after_ns,
        ) if voice.stt_result_ns > 0 else voice.end_ns
        return select_relevant_frame(
            candidates,
            baseline,
            voice.start_ns,
            window_end_ns,
        )

    @staticmethod
    def _candidate_from_frame(
        source: str,
        frame: Optional[CompressedImage],
    ) -> Optional[FrameCandidate[CompressedImage]]:
        if frame is None:
            return None
        return FrameCandidate(source, _stamp_to_ns(frame.header.stamp), frame)

    def _log_voice_frame_selection(
        self,
        voice: VoiceRequest,
        selection: FrameSelection[CompressedImage],
    ) -> None:
        def describe(frame: Optional[CompressedImage]) -> str:
            if frame is None:
                return "none"
            return self._format_ns(_stamp_to_ns(frame.header.stamp))

        motion_frame = (
            voice.motion.frames[0]
            if voice.motion is not None and voice.motion.frames
            else None
        )
        selected = selection.selected
        selected_text = (
            f"{selected.source}@{self._format_ns(selected.stamp_ns)}"
            if selected is not None
            else "none"
        )
        self.get_logger().info(
            f"Voice {voice.request_id} visual selection: window="
            f"{self._format_ns(selection.window_start_ns)}.."
            f"{self._format_ns(selection.window_end_ns)}, candidates="
            f"baseline:{describe(voice.baseline.frame)}, "
            f"refreshed:{describe(voice.refreshed.frame)}, "
            f"motion:{describe(motion_frame)}, selected={selected_text}, "
            f"baseline_fallback={selection.used_baseline_fallback}."
        )

    @staticmethod
    def _release_voice_frames(voice: VoiceRequest) -> None:
        voice.baseline.frame = None
        voice.refreshed.frame = None
        voice.motion = None

    def _publish_event(
        self,
        stamp: Time,
        event_type: str,
        transcript: str,
        trigger_reason: str,
        frame: CompressedImage,
        frame_source: str,
        start_ns: int,
        end_ns: int,
    ) -> None:
        payload = MultimodalEvent()
        payload.header.stamp = stamp
        payload.header.frame_id = "multimodal_manager"
        payload.event_type = event_type
        payload.transcript = transcript
        payload.trigger_reason = trigger_reason
        payload.frames = [frame]
        self._payload_pub.publish(payload)

        frame_ns = _stamp_to_ns(frame.header.stamp)
        summary = {
            "event_type": event_type,
            "event_timestamp": self._format_ns(_stamp_to_ns(stamp)),
            "source_interval": [self._format_ns(start_ns), self._format_ns(end_ns)],
            "transcript": transcript,
            "frame_timestamp": self._format_ns(frame_ns),
            "frame_source": frame_source,
            "frame_bytes": len(frame.data),
            "trigger_reason": trigger_reason,
        }
        summary_text = json.dumps(summary, ensure_ascii=False)
        self._summary_pub.publish(String(data=summary_text))
        self._publish_preview(frame, event_type, transcript, trigger_reason)
        self.get_logger().info(f"Multimodal input: {summary_text}")
        if self._mode == "vlm":
            self._enqueue_vlm(payload)

    def _enqueue_vlm(self, payload: MultimodalEvent) -> None:
        now_ns = time.monotonic_ns()
        with self._lock:
            if payload.event_type == "motion":
                if now_ns < self._next_motion_vlm_ns:
                    remaining_s = (self._next_motion_vlm_ns - now_ns) / 1_000_000_000
                    self.get_logger().info(
                        f"Motion-only VLM input skipped by cooldown ({remaining_s:.1f} s left)."
                    )
                    return
                self._next_motion_vlm_ns = now_ns + self._motion_vlm_cooldown_ns
            task_id = self._next_vlm_task_id
            self._next_vlm_task_id += 1
            pending_ttl_ns = (
                self._pending_voice_ttl_ns
                if payload.event_type in {"voice", "voice_motion"}
                else self._pending_motion_ttl_ns
            )
            task = VlmTask(
                vlm_task_id=task_id,
                event_type=payload.event_type,
                payload=payload,
                deadline_ns=now_ns + pending_ttl_ns,
            )
            outcome = self._scheduler.submit(task)
        if not outcome.accepted:
            self.get_logger().info(
                "Ignored pending motion VLM input because a voice input is waiting."
            )
            return
        if outcome.replaced is not None:
            self.get_logger().warn(
                f"Replaced pending {outcome.replaced.event_type} VLM input with "
                f"newer {payload.event_type} task {task_id}."
            )
        self._pump_vlm()

    def _pump_vlm(self) -> None:
        if self._mode != "vlm" or not self._vlm_client.server_is_ready():
            return
        with self._lock:
            begin = self._scheduler.begin_next(time.monotonic_ns())
        if begin.expired is not None:
            self.get_logger().warn(
                f"Discarded expired pending {begin.expired.event_type} VLM task "
                f"{begin.expired.vlm_task_id}."
            )
            return
        task = begin.task
        if task is None:
            return
        payload = task.payload
        if payload is None:
            self.get_logger().error(
                f"VLM task {task.vlm_task_id} has no payload; discarding it."
            )
            self._complete_vlm(task.vlm_task_id)
            return

        goal = RunVlm.Goal()
        goal.input = payload
        self.get_logger().info(
            f"Dispatching VLM task {task.vlm_task_id}: "
            f"{payload.event_type}, one selected frame."
        )
        try:
            future = self._vlm_client.send_goal_async(goal)
            future.add_done_callback(
                lambda completed, task_id=task.vlm_task_id: self._on_vlm_goal(
                    task_id, completed
                )
            )
        except Exception as exc:
            self.get_logger().error(
                f"Failed to send VLM task {task.vlm_task_id}: {exc}"
            )
            self._complete_vlm(task.vlm_task_id)

    def _on_vlm_goal(self, vlm_task_id: int, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"VLM task {vlm_task_id} goal error: {exc}")
            self._complete_vlm(vlm_task_id)
            return
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn(f"VLM task {vlm_task_id} was rejected.")
            self._complete_vlm(vlm_task_id)
            return
        with self._lock:
            became_active = self._scheduler.mark_active(vlm_task_id, goal_handle)
            task = self._scheduler.active_task
            event_type = task.event_type if became_active and task is not None else ""
            cancel_dispatch = (
                self._scheduler.take_cancellation_dispatch()
                if became_active
                else None
            )
        if not became_active:
            self.get_logger().warn(
                f"Ignoring accepted goal callback for stale VLM task {vlm_task_id}."
            )
            self._dispatch_vlm_cancel(
                CancellationDispatch(vlm_task_id, goal_handle)
            )
            return
        if cancel_dispatch is not None:
            self.get_logger().info(
                f"VLM task {vlm_task_id} was cancelled while dispatching; "
                "sending cancellation now that its goal handle is available."
            )
            self._dispatch_vlm_cancel(cancel_dispatch)
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda completed, task_id=vlm_task_id, kind=event_type: self._on_vlm_result(
                task_id, kind, completed
            )
        )

    def _on_vlm_result(self, vlm_task_id: int, event_type: str, future) -> None:
        try:
            response = future.result()
            result = response.result
        except Exception as exc:
            self.get_logger().error(f"VLM task {vlm_task_id} result failed: {exc}")
            self._complete_vlm(vlm_task_id)
            return

        if response.status != GoalStatus.STATUS_SUCCEEDED or not result.success:
            with self._lock:
                active = self._scheduler.active_task
                owns_active = bool(
                    active is not None and active.vlm_task_id == vlm_task_id
                )
                cancellation_expected = bool(
                    owns_active
                    and active is not None
                    and active.state == VlmTaskState.CANCEL_REQUESTED
                )
                completed = (
                    self._scheduler.complete(vlm_task_id)
                    if owns_active
                    else None
                )
                recovered_from_degraded = (
                    self._clear_cancel_diagnostic_locked(vlm_task_id)
                    if completed is not None
                    else False
                )
            if not owns_active:
                self.get_logger().warn(
                    f"Ignoring terminal result for stale VLM task {vlm_task_id}."
                )
                return
            message = result.error_message or f"status {response.status}"
            if cancellation_expected or response.status == GoalStatus.STATUS_CANCELED:
                self.get_logger().info(
                    f"VLM task {vlm_task_id} reached cancelled terminal state: "
                    f"{message}."
                )
            else:
                self.get_logger().error(
                    f"VLM task {vlm_task_id} result failed: {message}"
                )
            if recovered_from_degraded:
                self._publish_scheduler_status("ready", vlm_task_id)
            self._pump_vlm()
            return

        try:
            decision = speech_from_vlm_response(result.response_text, event_type)
        except Exception as exc:
            self.get_logger().error(
                f"VLM task {vlm_task_id} response policy failed: {exc}"
            )
            self._complete_vlm(vlm_task_id)
            return

        held = None
        tts_outcome = None
        recovered_from_degraded = False
        now_ns = time.monotonic_ns()
        with self._lock:
            active = self._scheduler.active_task
            owns_active = bool(
                active is not None and active.vlm_task_id == vlm_task_id
            )
            if not self._scheduler.result_is_current(vlm_task_id, now_ns):
                current = False
            else:
                current = True
                if decision.should_speak:
                    if self._scheduler.stt_unresolved:
                        held = self._scheduler.hold_response(
                            vlm_task_id,
                            decision.speech,
                            time.monotonic_ns(),
                        )
                    else:
                        tts_outcome = self._tts_queue.submit(
                            PendingSpeech(vlm_task_id, decision.speech)
                        )
            if owns_active:
                self._scheduler.complete(vlm_task_id)
                recovered_from_degraded = self._clear_cancel_diagnostic_locked(
                    vlm_task_id
                )

        if not owns_active:
            self.get_logger().warn(
                f"Discarding stale result callback for VLM task {vlm_task_id}."
            )
            return
        if recovered_from_degraded:
            self._publish_scheduler_status("ready", vlm_task_id)
        if not current:
            self.get_logger().warn(
                f"Discarded cancelled or expired result from VLM task "
                f"{vlm_task_id}."
            )
            self._pump_vlm()
            self._pump_tts()
            return

        self._vlm_response_pub.publish(String(data=result.response_text))
        self.get_logger().info(
            f"VLM task {vlm_task_id}: decision={decision.decision}, "
            f"should_speak={decision.should_speak}."
        )
        if held is not None:
            self.get_logger().info(
                f"Held response from VLM task {vlm_task_id} while STT for voice "
                f"{held.held.voice_id} is unresolved."
            )
            if held.replaced is not None:
                self.get_logger().warn(
                    f"Held response from VLM task {held.replaced.vlm_task_id} was "
                    f"replaced by newer task {vlm_task_id}."
                )
        if tts_outcome is not None and tts_outcome.replaced is not None:
            self.get_logger().warn(
                f"Replaced pending TTS response from VLM task "
                f"{tts_outcome.replaced.vlm_task_id} with task {vlm_task_id}."
            )
        self._pump_vlm()
        self._pump_tts()

    def _complete_vlm(self, vlm_task_id: int) -> None:
        with self._lock:
            completed = self._scheduler.complete(vlm_task_id)
            recovered_from_degraded = (
                self._clear_cancel_diagnostic_locked(vlm_task_id)
                if completed is not None
                else False
            )
        if completed is None:
            self.get_logger().warn(
                f"Late completion for VLM task {vlm_task_id} did not own active state."
            )
        elif recovered_from_degraded:
            self._publish_scheduler_status("ready", vlm_task_id)
        self._pump_vlm()

    def _start_cancel_diagnostic_locked(
        self,
        vlm_task_id: int,
        now_ns: int,
    ) -> None:
        self._cancel_task_id = vlm_task_id
        self._cancel_started_ns = now_ns
        self._cancel_degraded = False
        self._next_cancel_warning_ns = now_ns + self._vlm_cancel_grace_ns

    def _clear_cancel_diagnostic_locked(self, vlm_task_id: int) -> bool:
        if self._cancel_task_id != vlm_task_id:
            return False
        was_degraded = self._cancel_degraded
        self._cancel_task_id = None
        self._cancel_started_ns = 0
        self._cancel_degraded = False
        self._next_cancel_warning_ns = 0
        return was_degraded

    def _monitor_vlm_cancellation(self) -> None:
        now_ns = time.monotonic_ns()
        timeout_task_id = None
        degraded_task_id = None
        publish_degraded = False
        with self._lock:
            transition = self._scheduler.request_cancel_if_expired(now_ns)
            if transition.newly_requested and transition.task is not None:
                timeout_task_id = transition.task.vlm_task_id
                self._start_cancel_diagnostic_locked(timeout_task_id, now_ns)
            cancel_dispatch = self._scheduler.take_cancellation_dispatch()
            active = self._scheduler.active_task
            if (
                active is not None
                and active.state == VlmTaskState.CANCEL_REQUESTED
                and self._cancel_task_id == active.vlm_task_id
                and now_ns - self._cancel_started_ns >= self._vlm_cancel_grace_ns
                and now_ns >= self._next_cancel_warning_ns
            ):
                degraded_task_id = active.vlm_task_id
                publish_degraded = not self._cancel_degraded
                self._cancel_degraded = True
                self._next_cancel_warning_ns = now_ns + self._vlm_cancel_grace_ns

        if timeout_task_id is not None:
            self.get_logger().warn(
                f"VLM task {timeout_task_id} exceeded the active inference "
                "deadline; requesting cooperative cancellation."
            )
        if cancel_dispatch is not None:
            self._dispatch_vlm_cancel(cancel_dispatch)
        if degraded_task_id is not None:
            self.get_logger().error(
                f"VLM task {degraded_task_id} has exceeded the cancellation "
                "grace period; scheduler is degraded and will not start another "
                "generation until the old task becomes terminal."
            )
            if publish_degraded:
                self._publish_scheduler_status(
                    "degraded_cancellation",
                    degraded_task_id,
                )

    def _dispatch_vlm_cancel(self, dispatch: CancellationDispatch) -> None:
        try:
            future = dispatch.goal_handle.cancel_goal_async()
            future.add_done_callback(
                lambda completed, task_id=dispatch.vlm_task_id: (
                    self._on_vlm_cancel_response(task_id, completed)
                )
            )
        except Exception as exc:
            self.get_logger().error(
                f"Could not send cancellation for VLM task "
                f"{dispatch.vlm_task_id}: {exc}. The active slot remains occupied."
            )

    def _on_vlm_cancel_response(self, vlm_task_id: int, future) -> None:
        try:
            response = future.result()
            accepted = bool(response.goals_canceling)
        except Exception as exc:
            self.get_logger().error(
                f"VLM task {vlm_task_id} cancellation response failed: {exc}. "
                "The active slot remains occupied."
            )
            return
        if accepted:
            self.get_logger().info(
                f"VLM task {vlm_task_id} cancellation was accepted; waiting for "
                "terminal cleanup."
            )
        else:
            self.get_logger().warn(
                f"VLM task {vlm_task_id} cancellation was not accepted; waiting "
                "for its normal terminal result."
            )

    def _publish_scheduler_status(self, state: str, vlm_task_id: int) -> None:
        status = json.dumps(
            {"state": state, "vlm_task_id": vlm_task_id},
            ensure_ascii=False,
        )
        self._scheduler_status_pub.publish(String(data=status))

    def _enqueue_tts(self, speech: str, vlm_task_id: int) -> None:
        held = None
        with self._lock:
            if self._scheduler.stt_unresolved:
                held = self._scheduler.hold_response(
                    vlm_task_id,
                    speech,
                    time.monotonic_ns(),
                )
                outcome = None
            else:
                outcome = self._tts_queue.submit(
                    PendingSpeech(vlm_task_id=vlm_task_id, speech=speech)
                )
        if held is not None:
            self.get_logger().info(
                f"Held response from VLM task {vlm_task_id} while STT for voice "
                f"{held.held.voice_id} is unresolved."
            )
            if held.replaced is not None:
                self.get_logger().warn(
                    f"Held response from VLM task {held.replaced.vlm_task_id} was "
                    f"replaced by newer task {vlm_task_id}."
                )
            return
        assert outcome is not None
        if outcome.replaced is not None:
            self.get_logger().warn(
                f"Replaced pending TTS response from VLM task "
                f"{outcome.replaced.vlm_task_id} with task {vlm_task_id}."
            )
        self._pump_tts()

    def _pump_tts(self) -> None:
        if self._mode != "vlm" or not self._tts_client.server_is_ready():
            return
        with self._lock:
            if self._scheduler.stt_unresolved:
                return
            pending_speech = self._tts_queue.begin_next()
        if pending_speech is None:
            return

        goal = TextToSpeech.Goal()
        goal.text = pending_speech.speech
        self.get_logger().info(
            f"Sending speech from VLM task {pending_speech.vlm_task_id} to TTS: "
            f"'{pending_speech.speech[:160]}'"
        )
        try:
            future = self._tts_client.send_goal_async(goal)
            future.add_done_callback(self._on_tts_goal)
        except Exception as exc:
            self.get_logger().error(f"Failed to send TTS goal: {exc}")
            self._complete_tts()

    def _on_tts_goal(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"TTS goal error: {exc}")
            self._complete_tts()
            return
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("TTS goal was rejected.")
            self._complete_tts()
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_tts_result)

    def _on_tts_result(self, future) -> None:
        try:
            response = future.result()
            if response.status != GoalStatus.STATUS_SUCCEEDED or not response.result.success:
                raise RuntimeError(f"status {response.status}")
            self.get_logger().info("VLM response finished speaking.")
        except Exception as exc:
            self.get_logger().error(f"TTS result failed: {exc}")
        finally:
            self._complete_tts()

    def _complete_tts(self) -> None:
        with self._lock:
            if self._tts_queue.active:
                self._tts_queue.complete()
        self._pump_tts()

    def _publish_preview(
        self,
        compressed: CompressedImage,
        event_type: str,
        transcript: str,
        trigger_reason: str,
    ) -> None:
        encoded = np.frombuffer(bytes(compressed.data), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().error("Could not decode selected mock-input frame.")
            return
        self._annotate_preview(image, event_type, transcript, trigger_reason)
        message = Image()
        message.header = compressed.header
        message.height = image.shape[0]
        message.width = image.shape[1]
        message.encoding = "bgr8"
        message.is_bigendian = False
        message.step = image.shape[1] * image.shape[2]
        message.data = np.ascontiguousarray(image).tobytes()
        self._preview_pub.publish(message)

    @classmethod
    def _annotate_preview(
        cls,
        image: np.ndarray,
        event_type: str,
        transcript: str,
        trigger_reason: str,
    ) -> None:
        lines = [
            f"Event: {event_type} ({trigger_reason})",
            *cls._wrap_preview_text(
                f"Transcript: {transcript or '<no speech>'}",
                max(24, image.shape[1] // 12),
            ),
        ]
        overlay_height = min(image.shape[0], 12 + 28 * len(lines))
        cv2.rectangle(
            image,
            (0, 0),
            (image.shape[1], overlay_height),
            (18, 18, 18),
            -1,
        )
        for line_number, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (10, 25 + 28 * line_number),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def _wrap_preview_text(text: str, max_chars: int) -> list[str]:
        words = text.split()
        if not words:
            return [""]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines[:3]

    @staticmethod
    def _format_ns(stamp_ns: int) -> str:
        return f"{stamp_ns // 1_000_000_000}.{stamp_ns % 1_000_000_000:09d}"

    def destroy_node(self) -> None:
        if hasattr(self, "_stt_client"):
            self._stt_client.destroy()
        if hasattr(self, "_vlm_client"):
            self._vlm_client.destroy()
        if hasattr(self, "_tts_client"):
            self._tts_client.destroy()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node: Optional[MultimodalManager] = None
    executor: Optional[MultiThreadedExecutor] = None
    try:
        node = MultimodalManager()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("MultimodalManager shutting down …")
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
