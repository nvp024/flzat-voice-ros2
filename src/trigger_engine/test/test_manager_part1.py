import time

import pytest


rclpy = pytest.importorskip("rclpy")

from robot_interfaces.msg import MultimodalEvent, SpeechAudio
from trigger_engine.multimodal_manager import MultimodalManager, PendingSpeech
from trigger_engine.scheduler import VlmTask, VlmTaskState


def _audio() -> SpeechAudio:
    message = SpeechAudio()
    message.sample_rate = 16_000
    message.audio_data = [1] * 512
    return message


def test_manager_confirms_only_usable_final_transcript(monkeypatch) -> None:
    monkeypatch.setenv("ROS_LOG_DIR", "/tmp/ros_test_v11_part1")
    rclpy.init()
    node = MultimodalManager()
    try:
        node._on_audio(_audio())
        invalid_voice_id = node._active_stt_id
        assert invalid_voice_id is not None
        node._scheduler.hold_response(90, "old answer", now_ns=time.monotonic_ns())

        node._finish_stt(invalid_voice_id, "...?!", "")

        assert invalid_voice_id not in node._voices
        assert node._scheduler.stt_unresolved is False
        released = node._tts_queue.take_pending()
        assert released is not None
        assert released.vlm_task_id == 90
        assert released.speech == "old answer"

        node._on_audio(_audio())
        usable_voice_id = node._active_stt_id
        assert usable_voice_id is not None
        node._scheduler.hold_response(
            91,
            "obsolete answer",
            now_ns=time.monotonic_ns(),
        )

        node._finish_stt(usable_voice_id, "No", "")

        voice = node._voices.get(usable_voice_id)
        assert voice is not None
        assert voice.transcript == "No"
        assert voice.transcript_usable is True
        assert node._scheduler.held_response is None
        assert node._scheduler.stt_unresolved is False
        assert node._tts_queue.take_pending() is None
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


class _CancelResponse:
    goals_canceling = [object()]


class _CompletedCancelFuture:
    def add_done_callback(self, callback) -> None:
        callback(self)

    @staticmethod
    def result() -> _CancelResponse:
        return _CancelResponse()


class _GoalHandle:
    def __init__(self) -> None:
        self.cancel_count = 0

    def cancel_goal_async(self) -> _CompletedCancelFuture:
        self.cancel_count += 1
        return _CompletedCancelFuture()


class _VlmResult:
    success = True
    response_text = "obsolete model response"
    error_message = ""


class _SucceededVlmResponse:
    status = 4
    result = _VlmResult()


class _SucceededVlmFuture:
    @staticmethod
    def result() -> _SucceededVlmResponse:
        return _SucceededVlmResponse()


def test_usable_stt_cancels_obsolete_vlm_once(monkeypatch) -> None:
    monkeypatch.setenv("ROS_LOG_DIR", "/tmp/ros_test_v11_part2")
    rclpy.init()
    node = MultimodalManager()
    try:
        now_ns = time.monotonic_ns()
        goal_handle = _GoalHandle()
        active = VlmTask(
            20,
            "motion",
            MultimodalEvent(),
            deadline_ns=now_ns + 1_000_000_000,
        )
        pending = VlmTask(
            21,
            "motion",
            MultimodalEvent(),
            deadline_ns=now_ns + 1_000_000_000,
        )
        node._scheduler.submit(active)
        node._scheduler.begin_next(now_ns)
        node._scheduler.mark_active(20, goal_handle)
        node._scheduler.submit(pending)
        node._tts_queue.submit(PendingSpeech(20, "obsolete speech"))

        node._on_audio(_audio())
        voice_id = node._active_stt_id
        assert voice_id is not None
        node._finish_stt(voice_id, "Look over there", "")
        node._finish_stt(voice_id, "Look over there", "")

        assert goal_handle.cancel_count == 1
        assert active.state == VlmTaskState.CANCEL_REQUESTED
        assert node._scheduler.active_task is active
        assert node._scheduler.pending_task is None
        assert node._scheduler.held_response is None
        assert node._tts_queue.take_pending() is None
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def test_invalid_stt_does_not_cancel_active_vlm(monkeypatch) -> None:
    monkeypatch.setenv("ROS_LOG_DIR", "/tmp/ros_test_v11_part2_invalid")
    rclpy.init()
    node = MultimodalManager()
    try:
        now_ns = time.monotonic_ns()
        goal_handle = _GoalHandle()
        active = VlmTask(
            30,
            "motion",
            MultimodalEvent(),
            deadline_ns=now_ns + 1_000_000_000,
        )
        node._scheduler.submit(active)
        node._scheduler.begin_next(now_ns)
        node._scheduler.mark_active(30, goal_handle)

        node._on_audio(_audio())
        voice_id = node._active_stt_id
        assert voice_id is not None
        node._finish_stt(voice_id, "...?!", "")

        assert goal_handle.cancel_count == 0
        assert active.state == VlmTaskState.ACTIVE
        assert node._scheduler.active_task is active
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def test_successful_result_after_cancel_never_reaches_tts(monkeypatch) -> None:
    monkeypatch.setenv("ROS_LOG_DIR", "/tmp/ros_test_v11_part2_stale")
    rclpy.init()
    node = MultimodalManager()
    try:
        now_ns = time.monotonic_ns()
        task = VlmTask(
            40,
            "motion",
            MultimodalEvent(),
            deadline_ns=now_ns + 1_000_000_000,
        )
        node._scheduler.submit(task)
        node._scheduler.begin_next(now_ns)
        node._scheduler.mark_active(40, _GoalHandle())
        node._scheduler.request_active_cancel()

        node._on_vlm_result(40, "motion", _SucceededVlmFuture())

        assert node._scheduler.active_task is None
        assert node._tts_queue.take_pending() is None
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
