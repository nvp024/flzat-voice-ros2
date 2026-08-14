from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage, Image

from robot_interfaces.msg import VisualEvent
from robot_interfaces.srv import GetFrameNear, GetFramesAround
from vision_pipeline.frame_buffer import BufferedFrame, FrameRingBuffer
from vision_pipeline.motion_detector import MotionDetector, MotionResult


@dataclass(frozen=True)
class CapturedFrame:
    """Raw camera frame passed from capture to processing."""

    stamp_ns: int
    image: np.ndarray


def _time_from_ns(stamp_ns: int) -> Time:
    message = Time()
    message.sec = stamp_ns // 1_000_000_000
    message.nanosec = stamp_ns % 1_000_000_000
    return message


class CameraNode(Node):
    """Non-blocking camera capture with motion events and a frame buffer."""

    def __init__(self) -> None:
        super().__init__("camera_node")

        self._declare_parameters()
        camera_index = self._integer_parameter("camera_index")
        self._camera_width = self._integer_parameter("camera_width")
        self._camera_height = self._integer_parameter("camera_height")
        self._camera_fps = self._double_parameter("camera_fps")
        queue_size = self._integer_parameter("capture_queue_size")
        buffer_duration_s = self._double_parameter("buffer_duration_s")
        self._jpeg_quality = self._integer_parameter("jpeg_quality")
        self._debug_fps = self._double_parameter("debug_fps")
        self._retry_s = self._double_parameter("camera_retry_s")
        self._save_event_images = self._boolean_parameter("save_event_images")
        self._event_image_dir = Path(
            self._string_parameter("event_image_dir")
        ).expanduser()
        self._validate_parameters(queue_size, buffer_duration_s)

        max_buffer_frames = max(
            1,
            int(buffer_duration_s * self._camera_fps) + 1,
        )
        self._frame_buffer = FrameRingBuffer(
            buffer_duration_s,
            max_buffer_frames,
        )
        self._motion_detector = MotionDetector(
            processing_width=self._integer_parameter("processing_width"),
            pixel_threshold=self._integer_parameter("pixel_threshold"),
            min_motion_ratio=self._double_parameter("min_motion_ratio"),
            start_frames=self._integer_parameter("motion_start_frames"),
            end_s=self._double_parameter("motion_end_s"),
            settle_s=self._double_parameter("settle_s"),
            warmup_frames=self._integer_parameter("warmup_frames"),
            blur_size=self._integer_parameter("blur_size"),
        )

        debug_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        event_image_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        event_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._debug_pub = self.create_publisher(
            CompressedImage,
            "/vision/debug/compressed",
            debug_qos,
        )
        self._debug_raw_pub = self.create_publisher(
            Image,
            "/vision/debug",
            debug_qos,
        )
        self._event_frame_pub = self.create_publisher(
            CompressedImage,
            "/vision/event_frame/compressed",
            event_image_qos,
        )
        self._event_frame_raw_pub = self.create_publisher(
            Image,
            "/vision/event_frame",
            event_image_qos,
        )
        self._event_pub = self.create_publisher(
            VisualEvent,
            "/vision/events",
            event_qos,
        )
        self._frame_service = self.create_service(
            GetFrameNear,
            "/vision/get_frame_near",
            self._on_get_frame_near,
        )
        self._frames_service = self.create_service(
            GetFramesAround,
            "/vision/get_frames_around",
            self._on_get_frames_around,
        )

        self._queue: collections.deque[CapturedFrame] = collections.deque(
            maxlen=queue_size
        )
        self._queue_condition = threading.Condition()
        self._stop_event = threading.Event()
        self._capture_lock = threading.Lock()
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_count = 0
        self._processed_count = 0
        self._dropped_count = 0
        self._metric_lock = threading.Lock()
        self._last_debug_ns = 0
        self._event_id = 0

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(camera_index,),
            name="camera-capture",
            daemon=True,
        )
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name="camera-processing",
            daemon=True,
        )
        self._capture_thread.start()
        self._processing_thread.start()
        self._stats_timer = self.create_timer(5.0, self._report_stats)
        self.get_logger().info(
            f"Vision Stage 1 started: camera={camera_index}, "
            f"requested={self._camera_width}x{self._camera_height} "
            f"@ {self._camera_fps:.1f} FPS, buffer={buffer_duration_s:.1f} s."
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_fps", 15.0)
        self.declare_parameter("camera_retry_s", 2.0)
        self.declare_parameter("capture_queue_size", 2)
        self.declare_parameter("buffer_duration_s", 5.0)
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("debug_fps", 5.0)
        self.declare_parameter("processing_width", 320)
        self.declare_parameter("pixel_threshold", 25)
        self.declare_parameter("min_motion_ratio", 0.02)
        self.declare_parameter("motion_start_frames", 3)
        self.declare_parameter("motion_end_s", 0.7)
        self.declare_parameter("settle_s", 0.3)
        self.declare_parameter("warmup_frames", 20)
        self.declare_parameter("blur_size", 9)
        self.declare_parameter("save_event_images", False)
        self.declare_parameter("event_image_dir", "/tmp/vision_events")

    def _validate_parameters(self, queue_size: int, buffer_duration_s: float) -> None:
        if self._camera_width < 1 or self._camera_height < 1:
            raise ValueError("Camera dimensions must be positive.")
        if self._camera_fps <= 0.0 or self._debug_fps < 0.0:
            raise ValueError("Camera FPS must be positive and debug FPS non-negative.")
        if queue_size < 1:
            raise ValueError("capture_queue_size must be at least one.")
        if buffer_duration_s <= 0.0 or self._retry_s <= 0.0:
            raise ValueError("Buffer duration and camera retry must be positive.")
        if not 1 <= self._jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100.")

    def _capture_loop(self, camera_index: int) -> None:
        last_open_error_log = 0.0
        minimum_interval_ns = int(900_000_000 / self._camera_fps)
        last_accepted_ns = 0
        while not self._stop_event.is_set():
            capture = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
            if not capture.isOpened():
                capture.release()
                now = time.monotonic()
                if now - last_open_error_log >= 10.0:
                    self.get_logger().error(
                        f"Cannot open camera index {camera_index}; retrying."
                    )
                    last_open_error_log = now
                self._stop_event.wait(self._retry_s)
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera_height)
            capture.set(cv2.CAP_PROP_FPS, self._camera_fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            with self._capture_lock:
                if self._stop_event.is_set():
                    capture.release()
                    return
                self._capture = capture
            actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = capture.get(cv2.CAP_PROP_FPS)
            self.get_logger().info(
                f"Camera opened: {actual_width}x{actual_height} "
                f"@ {actual_fps:.1f} FPS."
            )

            consecutive_failures = 0
            while not self._stop_event.is_set():
                with self._capture_lock:
                    if self._stop_event.is_set():
                        break
                    success, image = capture.read()
                if not success or image is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        self.get_logger().error(
                            "Camera read failed repeatedly; reopening device."
                        )
                        break
                    continue
                consecutive_failures = 0
                stamp_ns = self.get_clock().now().nanoseconds
                if stamp_ns - last_accepted_ns < minimum_interval_ns:
                    continue
                last_accepted_ns = stamp_ns
                captured = CapturedFrame(stamp_ns=stamp_ns, image=image)
                with self._queue_condition:
                    if len(self._queue) == self._queue.maxlen:
                        self._queue.popleft()
                        with self._metric_lock:
                            self._dropped_count += 1
                    self._queue.append(captured)
                    self._queue_condition.notify()
                with self._metric_lock:
                    self._capture_count += 1

            self._release_active_capture()
            if not self._stop_event.is_set():
                self._stop_event.wait(self._retry_s)

    def _release_active_capture(self) -> None:
        """Release the camera without racing an in-progress VideoCapture.read()."""
        with self._capture_lock:
            capture = self._capture
            self._capture = None
            if capture is not None:
                capture.release()

    def _processing_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._queue_condition:
                frame_available = self._queue_condition.wait_for(
                    lambda: bool(self._queue) or self._stop_event.is_set(),
                    timeout=0.5,
                )
                if self._stop_event.is_set():
                    return
                if not frame_available:
                    continue
                captured = self._queue.pop()
                skipped = len(self._queue)
                self._queue.clear()
            if skipped:
                with self._metric_lock:
                    self._dropped_count += skipped

            try:
                self._process_frame(captured)
            except Exception as exc:
                self.get_logger().error(f"Frame processing failed: {exc}")

    def _process_frame(self, captured: CapturedFrame) -> None:
        jpeg_data = self._encode_jpeg(captured.image)
        buffered = BufferedFrame(
            stamp_ns=captured.stamp_ns,
            jpeg_data=jpeg_data,
            frame_id="camera",
        )
        self._frame_buffer.append(buffered)
        result = self._motion_detector.process(captured.image, captured.stamp_ns)

        if result.event_started:
            self.get_logger().info(f"Motion started: score={result.score:.3f}.")
        if result.event_finished:
            self._publish_motion_event(buffered, captured.image, result)
        self._publish_debug_if_due(captured, result)
        with self._metric_lock:
            self._processed_count += 1

    def _publish_motion_event(
        self,
        frame: BufferedFrame,
        raw_image: np.ndarray,
        result: MotionResult,
    ) -> None:
        self._event_id += 1
        image_message = self._compressed_message(frame)
        event = VisualEvent()
        event.header.stamp = _time_from_ns(frame.stamp_ns)
        event.header.frame_id = frame.frame_id
        event.event_id = self._event_id
        event.event_type = "motion"
        event.motion_start = _time_from_ns(result.motion_start_ns or frame.stamp_ns)
        event.motion_end = _time_from_ns(result.motion_end_ns or frame.stamp_ns)
        event.trigger_reason = "scene_change"
        event.motion_score = float(result.peak_score)
        event.frames = [image_message]
        self._event_pub.publish(event)
        self._event_frame_pub.publish(image_message)
        self._event_frame_raw_pub.publish(
            self._raw_image_message(raw_image, frame.stamp_ns, frame.frame_id)
        )
        saved_path = self._save_event_image(frame)
        saved_text = f", saved={saved_path}" if saved_path is not None else ""
        self.get_logger().info(
            f"Motion event {self._event_id}: settled frame published, "
            f"peak_score={result.peak_score:.3f}, "
            f"buffer={len(self._frame_buffer)}{saved_text}."
        )

    def _save_event_image(self, frame: BufferedFrame) -> Optional[Path]:
        if not self._save_event_images:
            return None
        try:
            self._event_image_dir.mkdir(parents=True, exist_ok=True)
            output = self._event_image_dir / (
                f"motion_event_{self._event_id:06d}_{frame.stamp_ns}.jpg"
            )
            output.write_bytes(frame.jpeg_data)
            return output
        except OSError as exc:
            self.get_logger().error(f"Could not save motion event image: {exc}")
            return None

    def _publish_debug_if_due(
        self,
        captured: CapturedFrame,
        result: MotionResult,
    ) -> None:
        if self._debug_fps == 0.0:
            return
        interval_ns = int(1_000_000_000 / self._debug_fps)
        if captured.stamp_ns - self._last_debug_ns < interval_ns:
            return
        self._last_debug_ns = captured.stamp_ns
        annotated = captured.image.copy()
        for x, y, width, height in result.boxes:
            cv2.rectangle(
                annotated,
                (x, y),
                (x + width, y + height),
                (0, 220, 80),
                2,
            )
        cv2.rectangle(
            annotated,
            (0, 0),
            (annotated.shape[1], 62),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            annotated,
            f"{result.state.value}  motion={result.score:.3f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"buffer={len(self._frame_buffer)} frames",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        debug_frame = BufferedFrame(
            captured.stamp_ns,
            self._encode_jpeg(annotated),
            "camera",
        )
        self._debug_pub.publish(self._compressed_message(debug_frame))
        self._debug_raw_pub.publish(
            self._raw_image_message(annotated, captured.stamp_ns, "camera")
        )

    def _on_get_frame_near(self, request, response):
        target_ns = request.target_stamp.sec * 1_000_000_000
        target_ns += request.target_stamp.nanosec
        frame = self._frame_buffer.nearest(target_ns, request.max_age_s)
        if frame is None:
            response.success = False
            response.message = "No buffered frame within the requested age."
            return response
        response.success = True
        response.message = "Nearest buffered frame selected."
        response.frame = self._compressed_message(frame)
        return response

    def _on_get_frames_around(self, request, response):
        target_ns = request.target_stamp.sec * 1_000_000_000
        target_ns += request.target_stamp.nanosec
        frames = self._frame_buffer.around(
            target_ns,
            request.before_s,
            request.after_s,
            request.max_frames,
        )
        if not frames:
            response.success = False
            response.message = "No buffered frames in the requested window."
            return response
        response.success = True
        response.message = f"Selected {len(frames)} buffered frame(s)."
        response.frames = [self._compressed_message(frame) for frame in frames]
        return response

    def _report_stats(self) -> None:
        with self._metric_lock:
            capture_count = self._capture_count
            processed_count = self._processed_count
            dropped_count = self._dropped_count
            self._capture_count = 0
            self._processed_count = 0
            self._dropped_count = 0
        self.get_logger().info(
            f"Vision health: capture={capture_count / 5.0:.1f} FPS, "
            f"processing={processed_count / 5.0:.1f} FPS, "
            f"dropped={dropped_count}, buffer={len(self._frame_buffer)}."
        )

    def _encode_jpeg(self, image: np.ndarray) -> bytes:
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        if not success:
            raise RuntimeError("OpenCV could not encode JPEG frame")
        return encoded.tobytes()

    @staticmethod
    def _compressed_message(frame: BufferedFrame) -> CompressedImage:
        message = CompressedImage()
        message.header.stamp = _time_from_ns(frame.stamp_ns)
        message.header.frame_id = frame.frame_id
        message.format = "bgr8; jpeg compressed bgr8"
        message.data = frame.jpeg_data
        return message

    @staticmethod
    def _raw_image_message(
        image: np.ndarray,
        stamp_ns: int,
        frame_id: str,
    ) -> Image:
        contiguous = np.ascontiguousarray(image)
        message = Image()
        message.header.stamp = _time_from_ns(stamp_ns)
        message.header.frame_id = frame_id
        message.height = contiguous.shape[0]
        message.width = contiguous.shape[1]
        message.encoding = "bgr8"
        message.is_bigendian = False
        message.step = contiguous.shape[1] * contiguous.shape[2]
        message.data = contiguous.tobytes()
        return message

    def _integer_parameter(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    def _double_parameter(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _boolean_parameter(self, name: str) -> bool:
        return self.get_parameter(name).get_parameter_value().bool_value

    def _string_parameter(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    @staticmethod
    def _join_worker(worker: threading.Thread) -> None:
        deadline = time.monotonic() + 3.0
        while worker.is_alive() and time.monotonic() < deadline:
            try:
                worker.join(timeout=0.2)
            except KeyboardInterrupt:
                continue

    def destroy_node(self) -> None:
        self._stop_event.set()
        self._release_active_capture()
        with self._queue_condition:
            self._queue_condition.notify_all()
        if hasattr(self, "_capture_thread"):
            self._join_worker(self._capture_thread)
        if hasattr(self, "_processing_thread"):
            self._join_worker(self._processing_thread)
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node: Optional[CameraNode] = None
    try:
        node = CameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("CameraNode shutting down …")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
