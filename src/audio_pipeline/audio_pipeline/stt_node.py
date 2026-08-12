from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from robot_interfaces.action import SpeechToText
from robot_interfaces.msg import SpeechAudio

EXPECTED_SAMPLE_RATE: int = 16_000


def _default_whisper_model_path() -> str:
    """Find the conventional model path without requiring a sourced shell."""
    colcon_prefixes = os.environ.get("COLCON_PREFIX_PATH", "")
    for prefix_value in filter(None, colcon_prefixes.split(os.pathsep)):
        prefix = Path(prefix_value).expanduser()
        # Colcon may use <workspace>/install or
        # <workspace>/install/<package> as an environment prefix.
        for models_dir in (prefix / "models", prefix.parent / "models"):
            if models_dir.is_dir():
                return str(models_dir / "tiny.pt")
        if prefix.parent.name == "install":
            models_dir = prefix.parent.parent / "models"
            if models_dir.is_dir():
                return str(models_dir / "tiny.pt")

    # This also works with --symlink-install because resolve() reaches src/.
    models_dir = Path(__file__).resolve().parents[3] / "models"
    return str(models_dir / "tiny.pt")


def _load_stt_model(model_path: str):
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Whisper model not found at {path}. "
            "Set the 'whisper_model_path' ROS parameter to a local .pt model file."
        )
    import whisper
    model = whisper.load_model(str(path))
    return model


class SttNode(Node):
    def __init__(self) -> None:
        super().__init__("stt_node")

        self._cb_group = ReentrantCallbackGroup()
        self._goal_lock = threading.Lock()
        self._goal_active = False

        self.get_logger().info("Loading Whisper model …")
        self.declare_parameter("whisper_model_path", _default_whisper_model_path())
        self.declare_parameter("language", "en")
        self.declare_parameter("expected_sample_rate", EXPECTED_SAMPLE_RATE)
        self.declare_parameter("max_audio_duration_s", 30.0)
        model_path = self.get_parameter("whisper_model_path").get_parameter_value().string_value
        language = self.get_parameter("language").get_parameter_value().string_value.strip()
        self._language = language or None
        self._expected_sample_rate = self.get_parameter(
            "expected_sample_rate"
        ).get_parameter_value().integer_value
        self._max_audio_duration_s = self.get_parameter(
            "max_audio_duration_s"
        ).get_parameter_value().double_value
        if self._expected_sample_rate != EXPECTED_SAMPLE_RATE:
            raise ValueError(
                f"Whisper input must be {EXPECTED_SAMPLE_RATE} Hz; "
                "resampling is not implemented in this node."
            )
        if self._max_audio_duration_s <= 0.0:
            raise ValueError("Parameter 'max_audio_duration_s' must be greater than zero.")
        self.get_logger().info(f"Whisper model path: {model_path}")
        self._stt_model = _load_stt_model(model_path)
        self.get_logger().info(
            f"Whisper model ready; language={self._language or 'auto'}."
        )

        self._action_server = ActionServer(
            self,
            SpeechToText,
            "/stt_action",
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb_group,
        )
        self.get_logger().info("SttNode ready — /stt_action active.")

    def _goal_callback(self, goal_request) -> GoalResponse:
        packet = goal_request.audio_packet
        sample_count = len(packet.audio_data)
        if sample_count == 0:
            self.get_logger().warn("Rejected empty STT goal.")
            return GoalResponse.REJECT
        if packet.sample_rate != self._expected_sample_rate:
            self.get_logger().warn(
                f"Rejected STT goal with sample rate {packet.sample_rate}; "
                f"expected {self._expected_sample_rate} Hz."
            )
            return GoalResponse.REJECT
        max_samples = int(self._expected_sample_rate * self._max_audio_duration_s)
        if sample_count > max_samples:
            self.get_logger().warn(
                f"Rejected STT goal with {sample_count} samples; "
                f"limit is {max_samples} ({self._max_audio_duration_s:.1f} s)."
            )
            return GoalResponse.REJECT

        with self._goal_lock:
            if self._goal_active:
                self.get_logger().warn("Rejected STT goal because inference is already active.")
                return GoalResponse.REJECT
            self._goal_active = True

        self.get_logger().info(
            f"STT goal accepted: {sample_count} samples @ {packet.sample_rate} Hz"
        )
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().info("STT cancel request received.")
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        try:
            if goal_handle.is_cancel_requested:
                self.get_logger().info("STT goal cancelled before inference started.")
                goal_handle.canceled()
                result = SpeechToText.Result()
                result.transcript = ""
                return result

            self.get_logger().info("Transcribing …")
            audio_packet: SpeechAudio = goal_handle.request.audio_packet
            audio_i16 = np.array(audio_packet.audio_data, dtype=np.int16)
            audio_f32 = audio_i16.astype(np.float32) / 32768.0

            feedback_msg = SpeechToText.Feedback()
            feedback_msg.status = "inference_started"
            goal_handle.publish_feedback(feedback_msg)

            transcribe_options: dict[str, object] = {"fp16": False}
            if self._language is not None:
                transcribe_options["language"] = self._language
            result_data = self._stt_model.transcribe(audio_f32, **transcribe_options)
            transcript = result_data["text"].strip()

            # Whisper cannot be safely interrupted mid-inference. The action is
            # nevertheless marked cancelled as soon as that inference returns.
            if goal_handle.is_cancel_requested:
                self.get_logger().info("STT goal cancelled after inference completed.")
                goal_handle.canceled()
                result = SpeechToText.Result()
                result.transcript = ""
                return result

            feedback_msg.status = "inference_complete"
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f"📝 Transcript: '{transcript}'")
            goal_handle.succeed()
            result = SpeechToText.Result()
            result.transcript = transcript
            return result
        except Exception as exc:
            self.get_logger().error(f"STT inference failed: {exc}")
            goal_handle.abort()
            result = SpeechToText.Result()
            result.transcript = ""
            return result
        finally:
            with self._goal_lock:
                self._goal_active = False

    def destroy_node(self) -> None:
        if hasattr(self, "_action_server"):
            self._action_server.destroy()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = SttNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("SttNode shutting down …")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
