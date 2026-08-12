from __future__ import annotations

import collections
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool
from std_msgs.msg import Header

import torch

from robot_interfaces.msg import SpeechAudio

SAMPLE_RATE: int     = 16_000
CHUNK_SAMPLES: int   = 512
VAD_THRESHOLD: float = 0.5
SILENCE_MS: int      = 500
MIN_SPEECH_MS: int   = 250
PRE_ROLL_MS: float   = 200
DEFAULT_MIC_DEVICE: str = "sysdefault"
DEFAULT_TTS_ACTIVE_TOPIC: str = "/voice/tts_active"
DEFAULT_MAX_SEGMENT_DURATION_S: float = 30.0
DEFAULT_TTS_RESUME_DELAY_S: float = 0.3


def _default_silero_jit_path() -> str:
    """Find the conventional model path without requiring a sourced shell."""
    colcon_prefixes = os.environ.get("COLCON_PREFIX_PATH", "")
    for prefix_value in filter(None, colcon_prefixes.split(os.pathsep)):
        prefix = Path(prefix_value).expanduser()
        # Colcon may use <workspace>/install or
        # <workspace>/install/<package> as an environment prefix.
        for models_dir in (prefix / "models", prefix.parent / "models"):
            if models_dir.is_dir():
                return str(
                    models_dir
                    / "snakers4_silero-vad_master/src/silero_vad/data/silero_vad.jit"
                )
        if prefix.parent.name == "install":
            models_dir = prefix.parent.parent / "models"
            if models_dir.is_dir():
                return str(
                    models_dir
                    / "snakers4_silero-vad_master/src/silero_vad/data/silero_vad.jit"
                )

    # This also works with --symlink-install because resolve() reaches src/.
    models_dir = Path(__file__).resolve().parents[3] / "models"
    return str(models_dir / "snakers4_silero-vad_master/src/silero_vad/data/silero_vad.jit")


def _load_vad_model(jit_path: str):
    p = Path(jit_path)
    if p.exists():
        model = torch.jit.load(str(p), map_location="cpu")
        model.eval()
        return model
    raise FileNotFoundError(
        f"Silero-VAD .jit not found at {p}. "
        "Set the 'silero_jit_path' ROS parameter to the model file."
    )


class _VADStream:
    def __init__(
        self,
        model,
        max_segment_duration_s: float,
        silence_ms: int,
    ) -> None:
        self._model = model
        self._torch = torch

        chunk_ms = CHUNK_SAMPLES / SAMPLE_RATE * 1000
        self._silence_chunks    = max(1, int(silence_ms / chunk_ms))
        self._min_speech_chunks = max(1, int(MIN_SPEECH_MS / chunk_ms))
        self._max_segment_chunks = max(
            1,
            int(max_segment_duration_s * SAMPLE_RATE / CHUNK_SAMPLES),
        )
        pre_roll_chunks = max(1, int(PRE_ROLL_MS / chunk_ms))

        self._ring_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=pre_roll_chunks)
        self._in_speech     = False
        self._silence_count = 0
        self._speech_frames: list[np.ndarray] = []
        self._speech_count  = 0
        self.reset()

    def reset(self) -> None:
        """Discard partial speech when the microphone is intentionally muted."""
        self._ring_buffer.clear()
        self._in_speech = False
        self._silence_count = 0
        self._speech_frames = []
        self._speech_count = 0
        self._model.reset_states()

    def _finish_segment(self, event: str):
        frames = self._speech_frames.copy()
        speech_count = self._speech_count
        self._in_speech = False
        self._speech_frames = []
        self._speech_count = 0
        self._silence_count = 0
        if event == "max_duration":
            self._model.reset_states()
        if speech_count >= self._min_speech_chunks:
            return (event, frames)
        return None

    def process(self, chunk: np.ndarray):
        tensor = self._torch.from_numpy(chunk.astype(np.float32))
        with self._torch.no_grad():
            confidence = self._model(tensor, SAMPLE_RATE).item()

        is_voice = confidence >= VAD_THRESHOLD

        if is_voice:
            if not self._in_speech:
                self._in_speech     = True
                self._speech_frames = list(self._ring_buffer)
                self._speech_frames.append(chunk)
                self._ring_buffer.clear()
                self._speech_count  = 1
                self._silence_count = 0
                return ("start", [])
            self._speech_frames.append(chunk)
            self._speech_count  += 1
            self._silence_count  = 0
            if len(self._speech_frames) >= self._max_segment_chunks:
                return self._finish_segment("max_duration")
        else:
            if self._in_speech:
                self._speech_frames.append(chunk)
                self._silence_count += 1
                if len(self._speech_frames) >= self._max_segment_chunks:
                    return self._finish_segment("max_duration")
                if self._silence_count >= self._silence_chunks:
                    return self._finish_segment("end")
            else:
                self._ring_buffer.append(chunk)

        return None


class _MicReader:
    def __init__(self, sample_rate: int, chunk_samples: int, device: str, queue_size: int) -> None:
        self._sample_rate   = sample_rate
        self._chunk_samples = chunk_samples
        self._device = device
        self._queue: collections.deque[np.ndarray] = collections.deque(maxlen=queue_size)
        self._queue_lock = threading.Lock()
        self._dropped_chunks = 0
        self._stream = None

    def _callback(self, indata, frames, time_info, status) -> None:
        chunk = indata[:, 0].copy()
        # No resampling needed — sysdefault routes through PulseAudio which
        # handles sample rate conversion automatically (same trick as voice_pipeline.py)
        with self._queue_lock:
            if status:
                # The node will continue with the next callback, but the status is
                # exposed through the queue-overflow diagnostic below.
                self._dropped_chunks += 1
            if len(self._queue) == self._queue.maxlen:
                self._queue.popleft()
                self._dropped_chunks += 1
            self._queue.append(chunk[: self._chunk_samples])

    def start(self) -> None:
        try:
            import sounddevice as sd
            self._stream = sd.InputStream(
                device     = self._device,
                samplerate = self._sample_rate,
                channels   = 1,
                dtype      = "float32",
                blocksize  = self._chunk_samples,
                callback   = self._callback,
            )
            self._stream.start()
        except Exception as exc:
            raise RuntimeError(
                f"Unable to open microphone '{self._device}' at "
                f"{self._sample_rate} Hz: {exc}"
            ) from exc

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def pop(self) -> Optional[np.ndarray]:
        with self._queue_lock:
            return self._queue.popleft() if self._queue else None

    def consume_dropped_chunks(self) -> int:
        with self._queue_lock:
            dropped_chunks = self._dropped_chunks
            self._dropped_chunks = 0
            return dropped_chunks


class VadNode(Node):
    TIMER_PERIOD_S: float = CHUNK_SAMPLES / SAMPLE_RATE / 2

    def __init__(self) -> None:
        super().__init__("vad_node")

        self._pub = self.create_publisher(SpeechAudio, "/audio_events", qos_profile=10)

        self.declare_parameter("silero_jit_path", _default_silero_jit_path())
        self.declare_parameter("mic_device", DEFAULT_MIC_DEVICE)
        self.declare_parameter("mic_queue_size", 200)
        self.declare_parameter("silence_ms", SILENCE_MS)
        self.declare_parameter("tts_active_topic", DEFAULT_TTS_ACTIVE_TOPIC)
        self.declare_parameter(
            "max_segment_duration_s",
            DEFAULT_MAX_SEGMENT_DURATION_S,
        )
        self.declare_parameter("tts_resume_delay_s", DEFAULT_TTS_RESUME_DELAY_S)
        jit_path = self.get_parameter("silero_jit_path").get_parameter_value().string_value
        mic_device = self.get_parameter("mic_device").get_parameter_value().string_value
        mic_queue_size = self.get_parameter("mic_queue_size").get_parameter_value().integer_value
        silence_ms = self.get_parameter("silence_ms").get_parameter_value().integer_value
        tts_active_topic = self.get_parameter("tts_active_topic").get_parameter_value().string_value
        max_segment_duration_s = self.get_parameter(
            "max_segment_duration_s"
        ).get_parameter_value().double_value
        self._tts_resume_delay_s = self.get_parameter(
            "tts_resume_delay_s"
        ).get_parameter_value().double_value
        if mic_queue_size < 1:
            raise ValueError("Parameter 'mic_queue_size' must be at least 1.")
        chunk_ms = CHUNK_SAMPLES / SAMPLE_RATE * 1000
        if silence_ms < chunk_ms:
            raise ValueError(
                f"Parameter 'silence_ms' must be at least {chunk_ms:.0f} ms."
            )
        if max_segment_duration_s < MIN_SPEECH_MS / 1000:
            raise ValueError(
                "Parameter 'max_segment_duration_s' must not be shorter "
                "than MIN_SPEECH_MS."
            )
        if self._tts_resume_delay_s < 0.0:
            raise ValueError("Parameter 'tts_resume_delay_s' cannot be negative.")

        voice_state_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._tts_active = False
        self._resume_vad_at = 0.0
        self._tts_active_sub = self.create_subscription(
            Bool,
            tts_active_topic,
            self._on_tts_active,
            voice_state_qos,
        )

        self.get_logger().info(f"Loading Silero-VAD model from: {jit_path}")
        self._vad_model = _load_vad_model(jit_path)
        self._vad_stream = _VADStream(
            self._vad_model,
            max_segment_duration_s,
            silence_ms,
        )
        effective_silence_ms = self._vad_stream._silence_chunks * chunk_ms
        self.get_logger().info(
            f"VAD model ready — endpoint silence={effective_silence_ms:.0f} ms."
        )

        self._mic = _MicReader(SAMPLE_RATE, CHUNK_SAMPLES, mic_device, mic_queue_size)
        self._mic.start()

        self._timer = self.create_timer(self.TIMER_PERIOD_S, self._timer_callback)
        self.get_logger().info(
            f"VadNode started — listening on microphone '{mic_device}' at {SAMPLE_RATE} Hz."
        )

    def _on_tts_active(self, msg: Bool) -> None:
        if msg.data and not self._tts_active:
            self._vad_stream.reset()
            self.get_logger().info("TTS active — VAD paused to prevent speaker feedback.")
        elif not msg.data and self._tts_active:
            self._vad_stream.reset()
            self._resume_vad_at = time.monotonic() + self._tts_resume_delay_s
            self.get_logger().info(
                f"TTS inactive — VAD resumes after "
                f"{self._tts_resume_delay_s:.2f} s cooldown."
            )
        self._tts_active = msg.data

    def _timer_callback(self) -> None:
        chunk = self._mic.pop()
        if chunk is None:
            return

        dropped_chunks = self._mic.consume_dropped_chunks()
        if dropped_chunks:
            self.get_logger().warn(
                f"Microphone queue overflow/status: dropped {dropped_chunks} audio chunk(s)."
            )

        if self._tts_active or time.monotonic() < self._resume_vad_at:
            return

        result = self._vad_stream.process(chunk)
        if result is None:
            return

        event, frames = result
        if event == "start":
            self.get_logger().info("🟢 Voice detected — listening …")
        elif event == "max_duration":
            self.get_logger().warn(
                "Maximum speech duration reached — transcribing current segment."
            )
            self._publish_segment(frames)
        elif event == "end":
            self._publish_segment(frames)

    def _publish_segment(self, frames: list[np.ndarray]) -> None:
        if not frames:
            return

        audio_f32 = np.concatenate(frames).astype(np.float32)
        audio_i16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)

        msg = SpeechAudio()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "vad_node"
        msg.audio_data = audio_i16.tolist()
        msg.sample_rate = SAMPLE_RATE

        self._pub.publish(msg)
        self.get_logger().info(
            f"⏹️  Published segment — {len(audio_i16) / SAMPLE_RATE * 1000:.0f} ms → transcribing …"
        )

    def destroy_node(self) -> None:
        if hasattr(self, "_mic"):
            self._mic.stop()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = VadNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("VadNode shutting down …")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
