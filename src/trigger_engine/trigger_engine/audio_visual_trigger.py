from __future__ import annotations

import threading
import time
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

        self._conversation_active = False
        self._conversation_lock = threading.Lock()
        self._pending_audio: Optional[tuple[SpeechAudio, float]] = None
        self._conversation_started_at = 0.0
        self._tts_first_word_seen = False

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
        received_at = time.monotonic()
        self.get_logger().info(f"Audio event: {len(msg.audio_data)} samples @ {msg.sample_rate} Hz")

        with self._conversation_lock:
            if self._conversation_active:
                # Keep only the newest completed utterance. This bounds memory
                # while preserving a follow-up after the current reply finishes.
                self._pending_audio = (msg, received_at)
                self.get_logger().warn(
                    "Conversation in progress — queued latest speech segment."
                )
                return
            self._conversation_active = True
            self._conversation_started_at = received_at

        self._send_stt_goal(msg)

    def _complete_conversation(self) -> None:
        """Start one queued segment after TTS, or return to listening."""
        with self._conversation_lock:
            next_request = self._pending_audio
            self._pending_audio = None
            if next_request is None:
                self._conversation_active = False
            else:
                self._conversation_started_at = next_request[1]
        if next_request is not None:
            self.get_logger().info("Reply finished — starting queued speech segment …")
            self._send_stt_goal(next_request[0])

    def _send_stt_goal(self, audio_msg: SpeechAudio) -> None:
        if not self._stt_client.server_is_ready():
            self.get_logger().error("Cannot send STT goal: /stt_action server is unavailable.")
            self._complete_conversation()
            return
        goal_msg = SpeechToText.Goal()
        goal_msg.audio_packet = audio_msg
        self.get_logger().info("Sending audio to STT …")
        try:
            send_future = self._stt_client.send_goal_async(
                goal_msg,
                feedback_callback=self._on_stt_feedback,
            )
        except Exception as exc:
            self.get_logger().error(f"Unable to send STT goal: {exc}")
            self._complete_conversation()
            return
        send_future.add_done_callback(self._on_stt_goal_accepted)

    def _on_stt_feedback(self, feedback_msg) -> None:
        self.get_logger().debug(f"STT feedback: {feedback_msg.feedback.status}")

    def _on_stt_goal_accepted(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"STT goal send failed: {exc}")
            self._complete_conversation()
            return

        if not goal_handle.accepted:
            self.get_logger().warn("STT goal rejected.")
            self._complete_conversation()
            return

        self.get_logger().info("STT goal accepted — waiting for result …")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_stt_result)

    def _on_stt_result(self, future) -> None:
        tts_started = False
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
                    tts_started = self._send_tts_goal(response)
        finally:
            # Keep the conversation slot until speech has actually finished.
            if not tts_started:
                self._complete_conversation()

    def _send_tts_goal(self, text: str) -> bool:
        if not self._tts_client.server_is_ready():
            self.get_logger().error("Cannot send TTS goal: /tts_action server is unavailable.")
            return False
        goal_msg = TextToSpeech.Goal()
        goal_msg.text = text
        self._tts_first_word_seen = False
        self.get_logger().info(f"Sending to TTS: '{text}'")
        try:
            send_future = self._tts_client.send_goal_async(
                goal_msg,
                feedback_callback=self._on_tts_feedback,
            )
        except Exception as exc:
            self.get_logger().error(f"Unable to send TTS goal: {exc}")
            return False
        send_future.add_done_callback(self._on_tts_goal_accepted)
        return True

    def _on_tts_feedback(self, feedback_msg) -> None:
        with self._tts_lock:
            first_word = not self._tts_first_word_seen
            if first_word:
                self._tts_first_word_seen = True
        if first_word:
            now = time.monotonic()
            self.get_logger().info(
                f"Latency: post-VAD response="
                f"{now - self._conversation_started_at:.3f} s"
            )
        self.get_logger().debug(f"TTS word: '{feedback_msg.feedback.current_word}'")

    def _on_tts_goal_accepted(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"TTS goal send failed: {exc}")
            self._complete_conversation()
            return

        if not goal_handle.accepted:
            self.get_logger().warn("TTS goal rejected.")
            self._complete_conversation()
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
            from action_msgs.msg import GoalStatus
            if result_response.status != GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().warn(
                    f"TTS did not succeed (status={result_response.status})."
                )
            elif not result_response.result.success:
                self.get_logger().warn("TTS completed without speaking successfully.")
            else:
                self.get_logger().info("TTS done — returning to listening.")
        except Exception as exc:
            self.get_logger().error(f"TTS result error: {exc}")
        finally:
            self._complete_conversation()

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
