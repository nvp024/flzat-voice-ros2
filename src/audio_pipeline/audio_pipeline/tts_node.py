from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool

from robot_interfaces.action import TextToSpeech

TTS_RATE: int = 175
TTS_VOLUME: float = 1.0
DEFAULT_TTS_ACTIVE_TOPIC: str = "/voice/tts_active"


class _Pyttsx3Engine:
    """Small, cancellable wrapper around the local pyttsx3 engine."""

    def __init__(self, stop_event: threading.Event, rate: int, volume: float) -> None:
        self._stop_event = stop_event
        self._rate = rate
        self._volume = volume
        self._engine = None
        self._engine_lock = threading.Lock()
        self.last_error: Optional[str] = None

    def request_stop(self) -> None:
        self._stop_event.set()
        with self._engine_lock:
            engine = self._engine
        if engine is not None:
            try:
                engine.stop()
            except Exception as exc:
                self.last_error = f"Unable to stop pyttsx3 engine: {exc}"

    def speak(self, text: str, on_word_callback: Optional[Callable[[str], None]] = None) -> bool:
        # DROP-IN: replace this method body with a different TTS engine if needed.
        try:
            import pyttsx3
        except ImportError:
            self.last_error = "pyttsx3 is not installed"
            return False

        engine = None
        try:
            engine = pyttsx3.init()
            with self._engine_lock:
                self._engine = engine
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)

            words = text.split()
            word_index = [0]

            def _on_word(name, location, length):
                if self._stop_event.is_set():
                    engine.stop()
                    return
                if word_index[0] < len(words):
                    if on_word_callback is not None:
                        on_word_callback(words[word_index[0]])
                    word_index[0] += 1

            engine.connect("started-word", _on_word)
            engine.say(text)
            engine.runAndWait()
            return not self._stop_event.is_set()
        except Exception as exc:
            self.last_error = str(exc)
            return False
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
            with self._engine_lock:
                self._engine = None


class TtsNode(Node):
    """A single-speaker TTS action server with explicit VAD muting state."""

    def __init__(self) -> None:
        super().__init__("tts_node")

        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._goal_reserved = False
        self._active_token: Optional[object] = None
        self._active_engine: Optional[_Pyttsx3Engine] = None
        self._cb_group = ReentrantCallbackGroup()

        self.declare_parameter("tts_rate", TTS_RATE)
        self.declare_parameter("tts_volume", TTS_VOLUME)
        self.declare_parameter("speech_timeout_s", 60.0)
        self.declare_parameter("cancel_grace_s", 2.0)
        self.declare_parameter("tts_active_topic", DEFAULT_TTS_ACTIVE_TOPIC)
        self._tts_rate = self.get_parameter("tts_rate").get_parameter_value().integer_value
        self._tts_volume = self.get_parameter("tts_volume").get_parameter_value().double_value
        self._speech_timeout_s = self.get_parameter(
            "speech_timeout_s"
        ).get_parameter_value().double_value
        self._cancel_grace_s = self.get_parameter(
            "cancel_grace_s"
        ).get_parameter_value().double_value
        tts_active_topic = self.get_parameter(
            "tts_active_topic"
        ).get_parameter_value().string_value
        if self._tts_rate <= 0:
            raise ValueError("Parameter 'tts_rate' must be greater than zero.")
        if not 0.0 <= self._tts_volume <= 1.0:
            raise ValueError("Parameter 'tts_volume' must be between 0.0 and 1.0.")
        if self._speech_timeout_s <= 0.0 or self._cancel_grace_s <= 0.0:
            raise ValueError("TTS timeout parameters must be greater than zero.")

        voice_state_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._tts_active_pub = self.create_publisher(Bool, tts_active_topic, voice_state_qos)

        self._action_server = ActionServer(
            self,
            TextToSpeech,
            "/tts_action",
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb_group,
        )
        self._publish_tts_active(False)
        self.get_logger().info(
            f"TtsNode ready — /tts_action active; VAD state on {tts_active_topic}."
        )

    def _publish_tts_active(self, active: bool) -> None:
        self._tts_active_pub.publish(Bool(data=active))

    def _goal_callback(self, goal_request) -> GoalResponse:
        text = goal_request.text.strip()
        if not text:
            self.get_logger().warn("Rejected empty TTS goal.")
            return GoalResponse.REJECT

        with self._state_lock:
            if self._goal_reserved:
                self.get_logger().warn("Rejected TTS goal because another speech request is active.")
                return GoalResponse.REJECT
            self._goal_reserved = True

        self.get_logger().info(f"TTS goal accepted: '{text[:60]}'")
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().info("TTS cancel request — stopping engine.")
        self._stop_event.set()
        with self._state_lock:
            engine = self._active_engine
        if engine is not None:
            engine.request_stop()
        return CancelResponse.ACCEPT

    def _release_goal(self, token: object) -> None:
        """Release the single-speaker slot only if it belongs to this goal."""
        with self._state_lock:
            if self._active_token is not token:
                return
            self._active_engine = None
            self._active_token = None
            self._goal_reserved = False
        self._publish_tts_active(False)

    def _execute_callback(self, goal_handle):
        text = goal_handle.request.text.strip()
        token = object()
        self.get_logger().info(f"TTS executing: '{text}'")

        self._stop_event.clear()
        with self._state_lock:
            self._active_token = token
        self._publish_tts_active(True)

        if goal_handle.is_cancel_requested:
            self._stop_event.set()
            self._release_goal(token)
            goal_handle.canceled()
            result = TextToSpeech.Result()
            result.success = False
            return result

        feedback_msg = TextToSpeech.Feedback()

        def _on_word(word: str) -> None:
            feedback_msg.current_word = word
            goal_handle.publish_feedback(feedback_msg)

        success_container: list[bool] = [False]
        speech_done = threading.Event()
        timed_out = threading.Event()
        engine = _Pyttsx3Engine(self._stop_event, self._tts_rate, self._tts_volume)

        def _speech_thread() -> None:
            try:
                success_container[0] = engine.speak(text, on_word_callback=_on_word)
            finally:
                speech_done.set()
                # If the action returned after an unresponsive engine, this
                # worker owns the eventual cleanup and VAD re-enable.
                if timed_out.is_set():
                    self._release_goal(token)

        with self._state_lock:
            self._active_engine = engine
        speech_thread = threading.Thread(target=_speech_thread, daemon=True)
        speech_thread.start()

        deadline = time.monotonic() + self._speech_timeout_s
        while not speech_done.wait(timeout=0.1):
            if goal_handle.is_cancel_requested:
                engine.request_stop()
            if time.monotonic() >= deadline:
                timed_out.set()
                engine.request_stop()
                if not speech_done.wait(timeout=self._cancel_grace_s):
                    self.get_logger().error(
                        "TTS engine did not stop before the timeout grace period; "
                        "rejecting future TTS goals until its worker exits."
                    )
                break

        if timed_out.is_set() and not speech_done.is_set():
            goal_handle.abort()
            result = TextToSpeech.Result()
            result.success = False
            return result

        self._release_goal(token)

        result = TextToSpeech.Result()
        if goal_handle.is_cancel_requested:
            self.get_logger().info("TTS goal cancelled.")
            goal_handle.canceled()
            result.success = False
        elif timed_out.is_set():
            self.get_logger().error("TTS goal timed out.")
            goal_handle.abort()
            result.success = False
        elif not success_container[0]:
            self.get_logger().error(f"TTS engine failed: {engine.last_error or 'unknown error'}")
            goal_handle.abort()
            result.success = False
        else:
            self.get_logger().info("TTS goal completed.")
            goal_handle.succeed()
            result.success = True
        return result

    def destroy_node(self) -> None:
        self._stop_event.set()
        with self._state_lock:
            engine = self._active_engine
        if engine is not None:
            engine.request_stop()
        if hasattr(self, "_action_server"):
            self._action_server.destroy()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = TtsNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("TtsNode shutting down …")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
