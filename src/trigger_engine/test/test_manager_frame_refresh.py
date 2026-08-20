import pytest


rclpy = pytest.importorskip("rclpy")

from sensor_msgs.msg import CompressedImage

from robot_interfaces.msg import SpeechAudio, VisualEvent
from trigger_engine.multimodal_manager import MultimodalManager, VoiceRequest


def _frame(stamp_ns: int, value: int = 1) -> CompressedImage:
    frame = CompressedImage()
    frame.header.stamp.sec = stamp_ns // 1_000_000_000
    frame.header.stamp.nanosec = stamp_ns % 1_000_000_000
    frame.format = "jpeg"
    frame.data = bytes([value])
    return frame


def _voice() -> VoiceRequest:
    return VoiceRequest(
        request_id=7,
        audio=SpeechAudio(),
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
        received_ns=10,
        fusion_deadline_ns=20,
        stt_state="done",
        transcript="What changed?",
        transcript_usable=True,
        stt_result_ns=2_600_000_000,
        stt_completed_ns=30,
    )


class _PendingFuture:
    def __init__(self) -> None:
        self.callback = None

    def add_done_callback(self, callback) -> None:
        self.callback = callback


class _FrameClient:
    def __init__(self) -> None:
        self.request = None
        self.future = _PendingFuture()

    @staticmethod
    def service_is_ready() -> bool:
        return True

    def call_async(self, request) -> _PendingFuture:
        self.request = request
        return self.future


class _FrameResponse:
    success = True
    message = "ok"

    def __init__(self, frame: CompressedImage) -> None:
        self.frames = [frame]


class _CompletedFrameFuture:
    def __init__(self, frame: CompressedImage) -> None:
        self._response = _FrameResponse(frame)

    def result(self) -> _FrameResponse:
        return self._response


def test_refresh_targets_relevance_end_without_future_wait(monkeypatch) -> None:
    monkeypatch.setenv("ROS_LOG_DIR", "/tmp/ros_test_v11_part3_request")
    rclpy.init()
    node = MultimodalManager()
    original_client = node._frame_client
    try:
        voice = _voice()
        voice.stt_result_ns = 3_000_000_000
        node._voices[voice.request_id] = voice
        fake_client = _FrameClient()
        node._frame_client = fake_client

        node._request_frame(voice.request_id, "refreshed")

        request = fake_client.request
        assert request is not None
        target_ns = request.target_stamp.sec * 1_000_000_000
        target_ns += request.target_stamp.nanosec
        assert target_ns == 2_750_000_000
        assert request.after_s == 0.0
        assert request.max_frames == 1
    finally:
        node._frame_client = original_client
        node.destroy_node()
        rclpy.try_shutdown()


def test_manager_selects_newest_relevant_candidate_and_releases_rest(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROS_LOG_DIR", "/tmp/ros_test_v11_part3_select")
    rclpy.init()
    node = MultimodalManager()
    try:
        voice = _voice()
        voice.baseline.frame = _frame(900_000_000, 1)
        voice.refreshed.frame = _frame(2_500_000_000, 2)
        voice.baseline.done = True
        voice.refreshed.done = True
        motion = VisualEvent()
        motion.frames = [_frame(2_700_000_000, 3)]
        voice.motion = motion

        selection = node._select_voice_frame(voice)

        assert selection.selected is not None
        assert selection.selected.source == "refreshed"
        assert selection.selected.stamp_ns == 2_500_000_000
        node._release_voice_frames(voice)
        assert voice.baseline.frame is None
        assert voice.refreshed.frame is None
        assert voice.motion is None
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def test_late_voice_callback_is_ignored(monkeypatch) -> None:
    monkeypatch.setenv("ROS_LOG_DIR", "/tmp/ros_test_v11_part3_stale")
    rclpy.init()
    node = MultimodalManager()
    try:
        node._on_frame_result(
            999,
            "refreshed",
            _CompletedFrameFuture(_frame(2_000_000_000)),
        )

        assert 999 not in node._voices
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def test_timed_out_slot_is_not_overwritten_by_late_callback(monkeypatch) -> None:
    monkeypatch.setenv("ROS_LOG_DIR", "/tmp/ros_test_v11_part3_timeout")
    rclpy.init()
    node = MultimodalManager()
    try:
        voice = _voice()
        voice.refreshed.done = True
        voice.refreshed.error = "frame service timeout"
        node._voices[voice.request_id] = voice

        node._on_frame_result(
            voice.request_id,
            "refreshed",
            _CompletedFrameFuture(_frame(2_500_000_000)),
        )

        assert voice.refreshed.frame is None
        assert voice.refreshed.error == "frame service timeout"
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
