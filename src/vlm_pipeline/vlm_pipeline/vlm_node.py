from __future__ import annotations

import time
from typing import Optional

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from robot_interfaces.action import RunVlm
from vlm_pipeline.backends import available_backends, create_backend
from vlm_pipeline.backends.base import BackendConfig, GenerationRequest
from vlm_pipeline.image_io import decode_compressed_images
from vlm_pipeline.job_gate import VlmJobGate
from vlm_pipeline.prompting import PromptBuilder


class VlmNode(Node):
    """Standalone one-at-a-time VLM action server with a replaceable backend."""

    def __init__(self) -> None:
        super().__init__("vlm_node")
        self._callback_group = ReentrantCallbackGroup()
        self._job_gate = VlmJobGate()
        self._declare_parameters()

        backend_name = self._string_parameter("backend")
        model_id = self._string_parameter("model_id")
        self._max_new_tokens = self._integer_parameter("max_new_tokens")
        self._max_input_bytes = self._integer_parameter("max_input_bytes")
        self._max_image_pixels = self._integer_parameter("max_image_pixels")
        self._default_prompt = self._string_parameter("default_prompt").strip()
        if self._max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least one")
        if self._max_input_bytes < 1 or self._max_image_pixels < 1:
            raise ValueError("image input limits must be positive")

        prompt_profile = self._string_parameter("prompt_profile")
        prompt_directory = self._string_parameter("prompt_directory")
        self._prompt_builder = PromptBuilder(prompt_profile, prompt_directory)
        raw_response_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._raw_response_pub = self.create_publisher(
            String,
            "/vlm/raw_response",
            raw_response_qos,
        )
        self.get_logger().info(
            f"Loaded VLM prompt profile '{prompt_profile}' from "
            f"{self._prompt_builder.root}."
        )

        config = BackendConfig(
            model_id=model_id,
            device=self._string_parameter("device"),
            dtype=self._string_parameter("dtype"),
            quantization=self._string_parameter("quantization"),
            trust_remote_code=self._boolean_parameter("trust_remote_code"),
            local_files_only=self._boolean_parameter("local_files_only"),
        )
        self.get_logger().info(
            f"Loading VLM: backend={backend_name}, model={model_id}, "
            f"device={config.device}, dtype={config.dtype}, "
            f"quantization={config.quantization}."
        )
        started = time.perf_counter()
        try:
            self._backend = create_backend(backend_name, config)
            self._backend.load()
        except Exception as exc:
            available = ", ".join(available_backends())
            self.get_logger().fatal(
                f"VLM initialization failed: {exc}. Available backends: {available}."
            )
            raise RuntimeError(f"VLM initialization failed: {exc}") from exc
        self.get_logger().info(
            f"VLM ready: backend={self._backend.name}, "
            f"device={self._backend.device_description}, "
            f"initialization={time.perf_counter() - started:.2f} s."
        )

        self._action_server = ActionServer(
            self,
            RunVlm,
            "/vlm/run",
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._callback_group,
        )
        self.get_logger().info("Standalone VLM action ready on /vlm/run.")

    def _declare_parameters(self) -> None:
        self.declare_parameter("backend", "smolvlm2")
        self.declare_parameter(
            "model_id",
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        )
        self.declare_parameter("device", "auto")
        self.declare_parameter("dtype", "auto")
        self.declare_parameter("quantization", "none")
        self.declare_parameter("max_new_tokens", 128)
        self.declare_parameter("max_input_bytes", 10_000_000)
        self.declare_parameter("max_image_pixels", 16_000_000)
        self.declare_parameter("trust_remote_code", False)
        self.declare_parameter("local_files_only", False)
        self.declare_parameter("prompt_profile", "companion_robot_v1")
        self.declare_parameter("prompt_directory", "")
        self.declare_parameter(
            "default_prompt",
            "Describe the image clearly and briefly.",
        )

    def _on_goal(self, goal_request: RunVlm.Goal) -> GoalResponse:
        payload = goal_request.input
        if len(payload.frames) != 1:
            self.get_logger().warn(
                f"Rejected VLM goal with {len(payload.frames)} frames; V1 requires one."
            )
            return GoalResponse.REJECT
        if not payload.frames[0].data:
            self.get_logger().warn("Rejected VLM goal with empty image data.")
            return GoalResponse.REJECT
        if payload.event_type in {"voice", "voice_motion"} and not payload.transcript.strip():
            self.get_logger().warn("Rejected voice VLM goal without a transcript.")
            return GoalResponse.REJECT
        if not self._job_gate.try_reserve():
            self.get_logger().warn("Rejected VLM goal because inference is busy.")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, _goal_handle) -> CancelResponse:
        self.get_logger().info(
            "VLM cancel requested; active generation will stop after model.generate returns."
        )
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        result = RunVlm.Result()
        images = ()
        inference_started = 0.0
        try:
            payload = goal_handle.request.input
            prompts = self._prompt_builder.build(
                payload.event_type,
                payload.transcript,
                payload.trigger_reason,
                self._default_prompt,
            )
            self._publish_feedback(goal_handle, "decoding_image")
            images = decode_compressed_images(
                list(payload.frames),
                self._max_input_bytes,
                self._max_image_pixels,
            )
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.error_message = "VLM request cancelled before inference"
                return result

            image = images[0]
            image_bytes = len(payload.frames[0].data)
            self.get_logger().info(
                f"VLM input: event_type={payload.event_type or 'unspecified'}, "
                f"trigger_reason={payload.trigger_reason or 'unspecified'}, "
                f"human_command='{payload.transcript[:120]}', "
                f"prompt_profile={self._prompt_builder.profile}, "
                f"image={image.width}x{image.height}, "
                f"compressed_bytes={image_bytes}."
            )
            self._publish_feedback(goal_handle, "inference_started")
            inference_started = time.perf_counter()
            raw_response = self._backend.generate(
                GenerationRequest(
                    images=images,
                    system_prompt=prompts.system_prompt,
                    user_prompt=prompts.user_prompt,
                    max_new_tokens=self._max_new_tokens,
                )
            )
            inference_s = time.perf_counter() - inference_started
            self._raw_response_pub.publish(String(data=raw_response))
            response = raw_response.strip()
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.error_message = "VLM request cancelled after inference"
                return result
            if not response:
                raise RuntimeError("VLM backend returned an empty response")

            self._publish_feedback(goal_handle, "inference_complete")
            result.success = True
            result.response_text = response
            result.error_message = ""
            goal_handle.succeed()
            self.get_logger().info(
                f"VLM response ({inference_s:.2f} s): '{response[:500]}'"
            )
            return result
        except Exception as exc:
            elapsed = (
                time.perf_counter() - inference_started
                if inference_started > 0.0
                else 0.0
            )
            self.get_logger().error(
                f"VLM request failed after {elapsed:.2f} s: {exc}"
            )
            goal_handle.abort()
            result.success = False
            result.response_text = ""
            result.error_message = str(exc)
            return result
        finally:
            for image in images:
                image.close()
            self._job_gate.release()

    @staticmethod
    def _publish_feedback(goal_handle, status: str) -> None:
        feedback = RunVlm.Feedback()
        feedback.status = status
        goal_handle.publish_feedback(feedback)

    def _integer_parameter(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    def _string_parameter(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _boolean_parameter(self, name: str) -> bool:
        return self.get_parameter(name).get_parameter_value().bool_value

    def destroy_node(self) -> None:
        if hasattr(self, "_action_server"):
            self._action_server.destroy()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node: Optional[VlmNode] = None
    executor: Optional[MultiThreadedExecutor] = None
    try:
        node = VlmNode()
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("VlmNode shutting down …")
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
