from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
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
from trigger_engine.output_policy import speech_from_vlm_response
from trigger_engine.work_queue import LatestPriorityWorkQueue


def _stamp_to_ns(stamp: Time) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


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
    stt_error: str = ""
    frame: Optional[CompressedImage] = None
    frame_requested: bool = False
    frame_request_started_ns: int = 0
    frame_done: bool = False
    frame_error: str = ""
    motion: Optional[VisualEvent] = None


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
        self._vlm_queue: LatestPriorityWorkQueue[MultimodalEvent] = (
            LatestPriorityWorkQueue()
        )
        self._tts_queue: LatestPriorityWorkQueue[str] = LatestPriorityWorkQueue()
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
        self.declare_parameter("frame_timeout_s", 2.0)
        self.declare_parameter("stt_server_timeout_s", 10.0)
        self.declare_parameter("motion_vlm_cooldown_s", 5.0)

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

        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1
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

            if self._active_stt_id is None:
                self._active_stt_id = request_id
            else:
                replaced = self._queued_stt_id
                self._queued_stt_id = request_id
                if replaced is not None:
                    self._voices.pop(replaced, None)
                    self._coordinator.discard_voice(replaced)
                    self.get_logger().warn(
                        f"Replaced queued voice request {replaced} with {request_id}."
                    )

        self.get_logger().info(
            f"Voice {request_id}: interval={self._format_ns(start_ns)}.."
            f"{self._format_ns(end_ns)}, samples={len(audio.audio_data)}; "
            "requesting frame and STT."
        )
        self._request_frame(request_id)
        self._pump_stt()

    def _on_motion(self, event: VisualEvent) -> None:
        if not event.frames:
            self.get_logger().warn(
                f"Ignoring motion event {event.event_id} without a frame."
            )
            return
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

    def _request_frame(self, request_id: int) -> None:
        if not self._frame_client.service_is_ready():
            return
        with self._lock:
            voice = self._voices.get(request_id)
            if voice is None or voice.frame_done or voice.frame_requested:
                return
            voice.frame_requested = True
            voice.frame_request_started_ns = time.monotonic_ns()
            request = GetFramesAround.Request()
            request.target_stamp = voice.audio.header.stamp
            request.before_s = self._frame_before_s
            request.after_s = self._frame_after_s
            request.max_frames = 1
        try:
            future = self._frame_client.call_async(request)
            future.add_done_callback(
                lambda completed, request_id=request_id: self._on_frame_result(
                    request_id, completed
                )
            )
        except Exception as exc:
            with self._lock:
                voice = self._voices.get(request_id)
                if voice is not None:
                    voice.frame_requested = False
            self.get_logger().error(
                f"Voice {request_id}: frame request failed to send: {exc}"
            )

    def _on_frame_result(self, request_id: int, future) -> None:
        response = None
        error = ""
        try:
            response = future.result()
        except Exception as exc:
            error = str(exc)
        with self._lock:
            voice = self._voices.get(request_id)
            if voice is None or voice.frame_done:
                return
            voice.frame_requested = False
            voice.frame_done = True
            if response is not None and response.success and response.frames:
                voice.frame = response.frames[0]
            else:
                voice.frame_error = error or (
                    response.message if response is not None else "unknown service error"
                )
        if voice.frame is None:
            self.get_logger().warn(
                f"Voice {request_id}: no synchronized frame: {voice.frame_error}"
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
        with self._lock:
            voice = self._voices.get(request_id)
            if voice is None or voice.stt_state == "done":
                return
            voice.stt_state = "done"
            voice.transcript = transcript
            voice.stt_error = error
            if self._active_stt_id == request_id:
                self._active_stt_id = None
                self._promote_queued_locked()
        if error:
            self.get_logger().error(f"Voice {request_id}: {error}.")
        else:
            self.get_logger().info(
                f"Voice {request_id}: transcript='{transcript}'."
            )
        self._pump_stt()

    def _promote_queued_locked(self) -> None:
        if self._active_stt_id is None and self._queued_stt_id is not None:
            self._active_stt_id = self._queued_stt_id
            self._queued_stt_id = None

    def _on_timer(self) -> None:
        self._pump_frame_requests()
        self._pump_stt()
        self._pump_vlm()
        self._pump_tts()
        now_ns = time.monotonic_ns()
        ready_voices: list[tuple[VoiceRequest, CompressedImage]] = []
        dropped_voices: list[VoiceRequest] = []
        motion: Optional[MotionWindow]

        with self._lock:
            motion = self._coordinator.take_due_motion(now_ns)
            for request_id, voice in list(self._voices.items()):
                frame = self._selected_frame(voice)
                fusion_ready = (
                    voice.motion is not None or now_ns >= voice.fusion_deadline_ns
                )
                if voice.stt_state == "done" and frame is not None and fusion_ready:
                    ready_voices.append((voice, frame))
                    self._voices.pop(request_id, None)
                    self._coordinator.complete_voice(request_id)
                elif (
                    voice.stt_state == "done"
                    and voice.frame_done
                    and frame is None
                    and fusion_ready
                ):
                    dropped_voices.append(voice)
                    self._voices.pop(request_id, None)
                    self._coordinator.complete_voice(request_id)

        for voice, frame in ready_voices:
            event_type = "voice_motion" if voice.motion is not None else "voice"
            reason = "speech_with_motion" if voice.motion is not None else "speech"
            self._publish_event(
                voice.audio.header.stamp,
                event_type,
                voice.transcript,
                reason,
                frame,
                voice.start_ns,
                voice.end_ns,
            )
        for voice in dropped_voices:
            self.get_logger().error(
                f"Voice {voice.request_id} dropped: no synchronized frame "
                f"({voice.frame_error or 'frame unavailable'})."
            )
        if motion is not None:
            event = motion.payload
            if event is not None and event.frames:
                self._publish_event(
                    event.header.stamp,
                    "motion",
                    "",
                    event.trigger_reason or "scene_change",
                    event.frames[0],
                    motion.start_ns,
                    motion.end_ns,
                )

    def _pump_frame_requests(self) -> None:
        now_ns = time.monotonic_ns()
        request_ids: list[int] = []
        with self._lock:
            for voice in self._voices.values():
                if voice.motion is not None or voice.frame_done:
                    continue
                age_ns = now_ns - voice.received_ns
                if voice.frame_requested:
                    if now_ns - voice.frame_request_started_ns >= self._frame_timeout_ns:
                        voice.frame_requested = False
                        voice.frame_done = True
                        voice.frame_error = "frame service timeout"
                    continue
                if age_ns >= self._frame_timeout_ns:
                    voice.frame_done = True
                    voice.frame_error = "frame service unavailable"
                else:
                    request_ids.append(voice.request_id)
        for request_id in request_ids:
            self._request_frame(request_id)

    @staticmethod
    def _selected_frame(voice: VoiceRequest) -> Optional[CompressedImage]:
        if voice.motion is not None and voice.motion.frames:
            return voice.motion.frames[0]
        return voice.frame

    def _publish_event(
        self,
        stamp: Time,
        event_type: str,
        transcript: str,
        trigger_reason: str,
        frame: CompressedImage,
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
        if payload.event_type == "motion":
            with self._lock:
                if now_ns < self._next_motion_vlm_ns:
                    remaining_s = (self._next_motion_vlm_ns - now_ns) / 1_000_000_000
                    self.get_logger().info(
                        f"Motion-only VLM input skipped by cooldown ({remaining_s:.1f} s left)."
                    )
                    return
                self._next_motion_vlm_ns = now_ns + self._motion_vlm_cooldown_ns

        priority = 1 if payload.event_type in {"voice", "voice_motion"} else 0
        with self._lock:
            outcome = self._vlm_queue.submit(payload, priority)
        if not outcome.accepted:
            self.get_logger().info(
                "Ignored pending motion VLM input because a voice input is waiting."
            )
            return
        if outcome.replaced is not None:
            self.get_logger().warn(
                f"Replaced pending {outcome.replaced.event_type} VLM input with "
                f"newer {payload.event_type} input."
            )
        self._pump_vlm()

    def _pump_vlm(self) -> None:
        if self._mode != "vlm" or not self._vlm_client.server_is_ready():
            return
        with self._lock:
            payload = self._vlm_queue.begin_next()
        if payload is None:
            return

        goal = RunVlm.Goal()
        goal.input = payload
        self.get_logger().info(
            f"Sending {payload.event_type} event to VLM (one selected frame)."
        )
        try:
            future = self._vlm_client.send_goal_async(goal)
            future.add_done_callback(
                lambda completed, event_type=payload.event_type: self._on_vlm_goal(
                    event_type, completed
                )
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to send VLM goal: {exc}")
            self._complete_vlm()

    def _on_vlm_goal(self, event_type: str, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"VLM goal error: {exc}")
            self._complete_vlm()
            return
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("VLM goal was rejected.")
            self._complete_vlm()
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda completed, event_type=event_type: self._on_vlm_result(
                event_type, completed
            )
        )

    def _on_vlm_result(self, event_type: str, future) -> None:
        try:
            response = future.result()
            result = response.result
            if response.status != GoalStatus.STATUS_SUCCEEDED or not result.success:
                raise RuntimeError(result.error_message or f"status {response.status}")
            decision = speech_from_vlm_response(result.response_text, event_type)
            self._vlm_response_pub.publish(String(data=result.response_text))
            self.get_logger().info(
                f"VLM decision={decision.decision}, should_speak={decision.should_speak}."
            )
            if decision.should_speak:
                self._enqueue_tts(decision.speech)
        except Exception as exc:
            self.get_logger().error(f"VLM result failed: {exc}")
        finally:
            self._complete_vlm()

    def _complete_vlm(self) -> None:
        with self._lock:
            if self._vlm_queue.active:
                self._vlm_queue.complete()
        self._pump_vlm()

    def _enqueue_tts(self, speech: str) -> None:
        with self._lock:
            outcome = self._tts_queue.submit(speech)
        if outcome.replaced is not None:
            self.get_logger().warn("Replaced one pending TTS response with the newest response.")
        self._pump_tts()

    def _pump_tts(self) -> None:
        if self._mode != "vlm" or not self._tts_client.server_is_ready():
            return
        with self._lock:
            speech = self._tts_queue.begin_next()
        if speech is None:
            return

        goal = TextToSpeech.Goal()
        goal.text = speech
        self.get_logger().info(f"Sending VLM speech to TTS: '{speech[:160]}'")
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
