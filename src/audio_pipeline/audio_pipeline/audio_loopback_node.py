from __future__ import annotations

import threading
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from robot_interfaces.action import SpeechToText, TextToSpeech
from robot_interfaces.msg import SpeechAudio


class AudioLoopbackNode(Node):
    """Connect VAD → STT → TTS for the standalone audio test launch."""

    def __init__(self) -> None:
        super().__init__("audio_loopback_node")
        self._cb_group = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self._active = False
        self._pending_audio: Optional[SpeechAudio] = None
        self._audio_sub = self.create_subscription(
            SpeechAudio,
            "/audio_events",
            self._on_audio,
            10,
            callback_group=self._cb_group,
        )
        self._stt_client = ActionClient(
            self,
            SpeechToText,
            "/stt_action",
            callback_group=self._cb_group,
        )
        self._tts_client = ActionClient(
            self,
            TextToSpeech,
            "/tts_action",
            callback_group=self._cb_group,
        )
        self.get_logger().info("Audio loopback ready — VAD → STT → TTS.")

    def _on_audio(self, audio: SpeechAudio) -> None:
        with self._lock:
            if self._active:
                self._pending_audio = audio
                self.get_logger().warn(
                    "Audio pipeline busy — retained only the newest speech segment."
                )
                return
            self._active = True
        self._send_stt(audio)

    def _send_stt(self, audio: SpeechAudio) -> None:
        if not self._stt_client.server_is_ready():
            self.get_logger().error("Cannot transcribe: /stt_action is unavailable.")
            self._complete()
            return
        goal = SpeechToText.Goal()
        goal.audio_packet = audio
        try:
            future = self._stt_client.send_goal_async(goal)
            future.add_done_callback(self._on_stt_goal)
        except Exception as exc:
            self.get_logger().error(f"Could not send STT goal: {exc}")
            self._complete()

    def _on_stt_goal(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"STT goal error: {exc}")
            self._complete()
            return
        if not goal_handle.accepted:
            self.get_logger().warn("STT goal rejected.")
            self._complete()
            return
        goal_handle.get_result_async().add_done_callback(self._on_stt_result)

    def _on_stt_result(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"STT result error: {exc}")
            self._complete()
            return
        if response.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warn(f"STT failed with status {response.status}.")
            self._complete()
            return
        transcript = response.result.transcript.strip()
        if not transcript:
            self.get_logger().warn("STT returned an empty transcript.")
            self._complete()
            return
        self.get_logger().info(f"Audio transcript: '{transcript}'")
        self._send_tts(transcript)

    def _send_tts(self, transcript: str) -> None:
        if not self._tts_client.server_is_ready():
            self.get_logger().error("Cannot speak: /tts_action is unavailable.")
            self._complete()
            return
        goal = TextToSpeech.Goal()
        goal.text = transcript
        try:
            future = self._tts_client.send_goal_async(goal)
            future.add_done_callback(self._on_tts_goal)
        except Exception as exc:
            self.get_logger().error(f"Could not send TTS goal: {exc}")
            self._complete()

    def _on_tts_goal(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"TTS goal error: {exc}")
            self._complete()
            return
        if not goal_handle.accepted:
            self.get_logger().warn("TTS goal rejected.")
            self._complete()
            return
        goal_handle.get_result_async().add_done_callback(self._on_tts_result)

    def _on_tts_result(self, future) -> None:
        try:
            response = future.result()
            if (
                response.status != GoalStatus.STATUS_SUCCEEDED
                or not response.result.success
            ):
                self.get_logger().warn("TTS did not complete successfully.")
        except Exception as exc:
            self.get_logger().error(f"TTS result error: {exc}")
        finally:
            self._complete()

    def _complete(self) -> None:
        with self._lock:
            audio = self._pending_audio
            self._pending_audio = None
            if audio is None:
                self._active = False
        if audio is not None:
            self._send_stt(audio)

    def destroy_node(self) -> None:
        self._stt_client.destroy()
        self._tts_client.destroy()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node: Optional[AudioLoopbackNode] = None
    executor: Optional[MultiThreadedExecutor] = None
    try:
        node = AudioLoopbackNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("AudioLoopbackNode shutting down …")
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
