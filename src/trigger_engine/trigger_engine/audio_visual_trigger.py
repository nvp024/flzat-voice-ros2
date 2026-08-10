from __future__ import annotations

import threading
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from robot_interfaces.action import SpeechToText, TextToSpeech
from robot_interfaces.msg import SpeechAudio


def _build_llm_response(transcript: str) -> str:
    # DROP-IN: replace with a real LLM call (OpenAI, Ollama, etc.)
    return transcript


class AudioVisualTrigger(Node):
    def __init__(self) -> None:
        super().__init__("audio_visual_trigger")

        self._cb_group = ReentrantCallbackGroup()

        self._audio_sub = self.create_subscription(
            SpeechAudio,
            "/audio_events",
            self._on_audio_event,
            qos_profile=10,
            callback_group=self._cb_group,
        )

        self._stt_client = ActionClient(self, SpeechToText, "/stt_action", callback_group=self._cb_group)
        self._tts_client = ActionClient(self, TextToSpeech, "/tts_action", callback_group=self._cb_group)

        self._tts_goal_handle: Optional[object] = None
        self._tts_lock = threading.Lock()

        self._stt_in_progress = False
        self._stt_lock = threading.Lock()
        self._pending_audio: Optional[SpeechAudio] = None

        self.get_logger().info("AudioVisualTrigger ready — waiting for action servers …")
        self._wait_for_servers()
        self._server_health_timer = self.create_timer(5.0, self._report_missing_servers)

    def _wait_for_servers(self) -> None:
        if not self._stt_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("/stt_action server not available.")
        else:
            self.get_logger().info("/stt_action server found.")

        if not self._tts_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("/tts_action server not available.")
        else:
            self.get_logger().info("/tts_action server found.")

    def _report_missing_servers(self) -> None:
        if not self._stt_client.server_is_ready():
            self.get_logger().warn("/stt_action server is unavailable; STT requests will be skipped.")
        if not self._tts_client.server_is_ready():
            self.get_logger().warn("/tts_action server is unavailable; TTS requests will be skipped.")

    def _on_audio_event(self, msg: SpeechAudio) -> None:
        self.get_logger().info(f"Audio event: {len(msg.audio_data)} samples @ {msg.sample_rate} Hz")

        with self._stt_lock:
            if self._stt_in_progress:
                # Keep only the newest completed utterance. This bounds memory
                # while preserving a follow-up spoken after a long transcription.
                self._pending_audio = msg
                self.get_logger().warn("STT in progress — queued latest speech segment.")
                return
            self._stt_in_progress = True

        self._send_stt_goal(msg)

    def _complete_stt_slot(self) -> None:
        """Start one queued segment, or mark the single Whisper slot idle."""
        with self._stt_lock:
            next_audio = self._pending_audio
            self._pending_audio = None
            if next_audio is None:
                self._stt_in_progress = False
        if next_audio is not None:
            self.get_logger().info("Starting queued speech segment …")
            self._send_stt_goal(next_audio)

    def _send_stt_goal(self, audio_msg: SpeechAudio) -> None:
        if not self._stt_client.server_is_ready():
            self.get_logger().error("Cannot send STT goal: /stt_action server is unavailable.")
            self._complete_stt_slot()
            return
        goal_msg = SpeechToText.Goal()
        goal_msg.audio_packet = audio_msg
        self.get_logger().info("Sending audio to STT …")
        send_future = self._stt_client.send_goal_async(goal_msg, feedback_callback=self._on_stt_feedback)
        send_future.add_done_callback(self._on_stt_goal_accepted)

    def _on_stt_feedback(self, feedback_msg) -> None:
        self.get_logger().debug(f"STT feedback: {feedback_msg.feedback.status}")

    def _on_stt_goal_accepted(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"STT goal send failed: {exc}")
            self._complete_stt_slot()
            return

        if not goal_handle.accepted:
            self.get_logger().warn("STT goal rejected.")
            self._complete_stt_slot()
            return

        self.get_logger().info("STT goal accepted — waiting for result …")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_stt_result)

    def _on_stt_result(self, future) -> None:
        try:
            result_response = future.result()
        except Exception as exc:
            self.get_logger().error(f"STT result error: {exc}")
        else:
            from action_msgs.msg import GoalStatus
            if result_response.status != GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().warn(f"STT did not succeed (status={result_response.status}).")
            else:
                transcript = result_response.result.transcript
                if not transcript.strip():
                    self.get_logger().warn("Empty transcript — skipping TTS.")
                else:
                    self.get_logger().info(f"Transcript: '{transcript}'")
                    response = _build_llm_response(transcript)
                    self._send_tts_goal(response)
        finally:
            self._complete_stt_slot()

    def _send_tts_goal(self, text: str) -> None:
        if not self._tts_client.server_is_ready():
            self.get_logger().error("Cannot send TTS goal: /tts_action server is unavailable.")
            return
        goal_msg = TextToSpeech.Goal()
        goal_msg.text = text
        self.get_logger().info(f"Sending to TTS: '{text}'")
        send_future = self._tts_client.send_goal_async(goal_msg, feedback_callback=self._on_tts_feedback)
        send_future.add_done_callback(self._on_tts_goal_accepted)

    def _on_tts_feedback(self, feedback_msg) -> None:
        self.get_logger().debug(f"TTS word: '{feedback_msg.feedback.current_word}'")

    def _on_tts_goal_accepted(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"TTS goal send failed: {exc}")
            return

        if not goal_handle.accepted:
            self.get_logger().warn("TTS goal rejected.")
            return

        self.get_logger().info("TTS goal accepted — speaking …")
        with self._tts_lock:
            self._tts_goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_tts_result)

    def _on_tts_result(self, future) -> None:
        with self._tts_lock:
            self._tts_goal_handle = None
        try:
            result_response = future.result()
            self.get_logger().info(f"TTS done — success={result_response.result.success}")
        except Exception as exc:
            self.get_logger().error(f"TTS result error: {exc}")

    def destroy_node(self) -> None:
        self._stt_client.destroy()
        self._tts_client.destroy()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AudioVisualTrigger()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("AudioVisualTrigger shutting down …")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
