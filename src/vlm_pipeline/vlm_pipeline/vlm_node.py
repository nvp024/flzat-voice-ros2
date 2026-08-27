from __future__ import annotations

import threading
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

from robot_interfaces.action import AnalyzeEnvironment, RunVlm
from vlm_pipeline.backends import available_backends, create_backend
from vlm_pipeline.backends.base import (
    BackendConfig,
    GenerationCancelled,
    GenerationRequest,
)
from vlm_pipeline.cancellation import GoalCancellationRegistry
from vlm_pipeline.environment_prompting import EnvironmentPromptBuilder
from vlm_pipeline.environment_schema import (
    EnvironmentSchemaError,
    parse_environment_response,
)
from vlm_pipeline.image_io import decode_compressed_images
from vlm_pipeline.priority_broker import (
    InferenceTicket,
    PriorityInferenceBroker,
    TicketState,
)
from vlm_pipeline.prompting import PromptBuilder


VOICE_PRIORITY = 30
ENVIRONMENT_PRIORITY = 20
MOTION_PRIORITY = 10


class VlmNode(Node):
    """One shared VLM backend serving conversation and environment actions."""

    def __init__(self) -> None:
        super().__init__("vlm_node")
        self._callback_group = ReentrantCallbackGroup()
        self._broker = PriorityInferenceBroker()
        self._cancel_tokens = GoalCancellationRegistry()
        self._declare_parameters()

        backend_name = self._string_parameter("backend")
        model_id = self._string_parameter("model_id")
        self._max_new_tokens = self._integer_parameter("max_new_tokens")
        self._environment_max_new_tokens = self._integer_parameter(
            "environment_max_new_tokens"
        )
        self._max_input_bytes = self._integer_parameter("max_input_bytes")
        self._max_image_pixels = self._integer_parameter("max_image_pixels")
        self._default_prompt = self._string_parameter("default_prompt").strip()
        if self._max_new_tokens < 1 or self._environment_max_new_tokens < 1:
            raise ValueError("VLM token limits must be at least one")
        if self._max_input_bytes < 1 or self._max_image_pixels < 1:
            raise ValueError("image input limits must be positive")

        prompt_profile = self._string_parameter("prompt_profile")
        prompt_directory = self._string_parameter("prompt_directory")
        self._prompt_builder = PromptBuilder(prompt_profile, prompt_directory)
        self._environment_prompt_builder = EnvironmentPromptBuilder(
            self._string_parameter("environment_prompt_directory")
        )
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
            f"Loaded companion prompt '{prompt_profile}' and environment prompt "
            f"from {self._environment_prompt_builder.root}."
        )

        config = BackendConfig(
            model_id=model_id,
            device=self._string_parameter("device"),
            dtype=self._string_parameter("dtype"),
            quantization=self._string_parameter("quantization"),
            trust_remote_code=self._boolean_parameter("trust_remote_code"),
            local_files_only=self._boolean_parameter("local_files_only"),
            do_image_splitting=self._boolean_parameter("do_image_splitting"),
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

        self._run_action_server = ActionServer(
            self,
            RunVlm,
            "/vlm/run",
            execute_callback=self._execute_run,
            goal_callback=self._on_run_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._callback_group,
        )
        self._environment_action_server = ActionServer(
            self,
            AnalyzeEnvironment,
            "/vlm/analyze_environment",
            execute_callback=self._execute_environment,
            goal_callback=self._on_environment_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._callback_group,
        )
        self.get_logger().info(
            "Shared VLM actions ready on /vlm/run and /vlm/analyze_environment."
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("backend", "smolvlm2")
        self.declare_parameter(
            "model_id", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        )
        self.declare_parameter("device", "auto")
        self.declare_parameter("dtype", "auto")
        self.declare_parameter("quantization", "none")
        self.declare_parameter("max_new_tokens", 48)
        self.declare_parameter("environment_max_new_tokens", 256)
        self.declare_parameter("max_input_bytes", 10_000_000)
        self.declare_parameter("max_image_pixels", 16_000_000)
        self.declare_parameter("trust_remote_code", False)
        self.declare_parameter("local_files_only", False)
        self.declare_parameter("do_image_splitting", False)
        self.declare_parameter("prompt_profile", "companion_robot_v1")
        self.declare_parameter("prompt_directory", "")
        self.declare_parameter("environment_prompt_directory", "")
        self.declare_parameter(
            "default_prompt", "Describe the image clearly and briefly."
        )

    def _on_run_goal(self, goal_request: RunVlm.Goal) -> GoalResponse:
        payload = goal_request.input
        if len(payload.frames) != 1:
            self.get_logger().warn(
                f"Rejected VLM goal with {len(payload.frames)} frames; V1 requires one."
            )
            return GoalResponse.REJECT
        if not payload.frames[0].data:
            self.get_logger().warn("Rejected VLM goal with empty image data.")
            return GoalResponse.REJECT
        if (
            payload.event_type in {"voice", "voice_motion"}
            and not payload.transcript.strip()
        ):
            self.get_logger().warn("Rejected voice VLM goal without a transcript.")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_environment_goal(
        self, goal_request: AnalyzeEnvironment.Goal
    ) -> GoalResponse:
        if not goal_request.observation_id.strip() or not goal_request.image.data:
            self.get_logger().warn(
                "Rejected environment goal without observation ID or image data."
            )
            return GoalResponse.REJECT
        if len(goal_request.detections) > 8:
            self.get_logger().warn("Rejected environment goal with more than 8 boxes.")
            return GoalResponse.REJECT
        detection_ids = set()
        for detection in goal_request.detections:
            if detection.detection_id in detection_ids:
                self.get_logger().warn("Rejected duplicate detection ID.")
                return GoalResponse.REJECT
            detection_ids.add(detection.detection_id)
            if (
                detection.x_min < 0
                or detection.y_min < 0
                or detection.x_max <= detection.x_min
                or detection.y_max <= detection.y_min
                or not 0.0 <= detection.confidence <= 1.0
            ):
                self.get_logger().warn("Rejected invalid detector metadata.")
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle) -> CancelResponse:
        if self._cancel_tokens.request_cancel(goal_handle):
            self.get_logger().info("Cooperative VLM cancellation was signalled.")
        return CancelResponse.ACCEPT

    def _execute_run(self, goal_handle):
        result = RunVlm.Result()
        images = ()
        token = threading.Event()
        ticket = None
        token_registered = False
        inference_started = 0.0
        try:
            self._cancel_tokens.register(goal_handle, token)
            token_registered = True
            payload = goal_handle.request.input
            ticket = self._broker.submit(
                self._goal_key(goal_handle),
                self._run_priority(payload.event_type),
                token,
            )
            self._publish_run_feedback(goal_handle, "queued")
            if not self._wait_for_turn(goal_handle, ticket):
                return self._finish_run_without_inference(goal_handle, result, ticket)

            prompts = self._prompt_builder.build(
                payload.event_type,
                payload.transcript,
                payload.trigger_reason,
                self._default_prompt,
            )
            self._publish_run_feedback(goal_handle, "decoding_image")
            images = decode_compressed_images(
                list(payload.frames), self._max_input_bytes, self._max_image_pixels
            )
            self._raise_if_cancelled(goal_handle, ticket)
            self._publish_run_feedback(goal_handle, "inference")
            inference_started = time.perf_counter()
            raw_response = self._generate(
                images,
                prompts.system_prompt,
                prompts.user_prompt,
                self._max_new_tokens,
                token,
            )
            self._raise_if_cancelled(goal_handle, ticket)
            self._publish_raw_response(raw_response)
            result.success = True
            result.response_text = raw_response
            result.error_message = ""
            goal_handle.succeed()
            self._publish_run_feedback(goal_handle, "complete")
            self.get_logger().info(
                f"VLM response ({time.perf_counter() - inference_started:.2f} s): "
                f"'{raw_response[:500]}'"
            )
            return result
        except GenerationCancelled:
            return self._cancel_or_preempt_run(goal_handle, result, ticket)
        except Exception as exc:
            if ticket is not None and token.is_set():
                return self._cancel_or_preempt_run(goal_handle, result, ticket)
            self.get_logger().error(f"VLM request failed: {exc}")
            goal_handle.abort()
            result.success = False
            result.response_text = ""
            result.error_message = str(exc)
            return result
        finally:
            for image in images:
                image.close()
            if ticket is not None:
                self._broker.complete(ticket)
            if token_registered:
                self._cancel_tokens.unregister(goal_handle, token)

    def _execute_environment(self, goal_handle):
        result = AnalyzeEnvironment.Result()
        images = ()
        token = threading.Event()
        ticket = None
        token_registered = False
        try:
            self._cancel_tokens.register(goal_handle, token)
            token_registered = True
            request = goal_handle.request
            ticket = self._broker.submit(
                self._goal_key(goal_handle), ENVIRONMENT_PRIORITY, token
            )
            self._publish_environment_feedback(goal_handle, "queued")
            if not self._wait_for_turn(goal_handle, ticket):
                return self._finish_environment_without_inference(
                    goal_handle, result, ticket
                )

            prompts = self._environment_prompt_builder.build(
                request.observation_id, request.detections
            )
            self._publish_environment_feedback(goal_handle, "decoding_image")
            images = decode_compressed_images(
                [request.image], self._max_input_bytes, self._max_image_pixels
            )
            self._raise_if_cancelled(goal_handle, ticket)
            self._publish_environment_feedback(goal_handle, "inference")
            raw_response = self._generate(
                images,
                prompts.system_prompt,
                prompts.user_prompt,
                self._environment_max_new_tokens,
                token,
            )
            self._raise_if_cancelled(goal_handle, ticket)
            self._publish_raw_response(raw_response)
            self._publish_environment_feedback(goal_handle, "validating")
            try:
                analysis = parse_environment_response(
                    raw_response,
                    (item.detection_id for item in request.detections),
                )
            except EnvironmentSchemaError as first_error:
                self._publish_environment_feedback(goal_handle, "retrying")
                repair_prompts = self._environment_prompt_builder.build_repair(
                    request.observation_id,
                    request.detections,
                    raw_response,
                    str(first_error),
                )
                self._raise_if_cancelled(goal_handle, ticket)
                repaired_response = self._generate(
                    images,
                    repair_prompts.system_prompt,
                    repair_prompts.user_prompt,
                    self._environment_max_new_tokens,
                    token,
                )
                self._raise_if_cancelled(goal_handle, ticket)
                self._publish_raw_response(repaired_response)
                self._publish_environment_feedback(goal_handle, "validating")
                try:
                    analysis = parse_environment_response(
                        repaired_response,
                        (item.detection_id for item in request.detections),
                    )
                except EnvironmentSchemaError as second_error:
                    result.success = False
                    result.scene = ""
                    result.objects = []
                    result.raw_response = repaired_response
                    result.error_message = (
                        "schema invalid after one repair retry: "
                        f"{second_error}"
                    )
                    goal_handle.succeed()
                    self._publish_environment_feedback(goal_handle, "complete")
                    return result
                raw_response = repaired_response
            result.success = True
            result.scene = analysis.scene
            result.objects = [
                self._semantic_message(item) for item in analysis.objects
            ]
            result.raw_response = raw_response
            result.error_message = ""
            goal_handle.succeed()
            self._publish_environment_feedback(goal_handle, "complete")
            return result
        except GenerationCancelled:
            return self._cancel_or_preempt_environment(goal_handle, result, ticket)
        except Exception as exc:
            if ticket is not None and token.is_set():
                return self._cancel_or_preempt_environment(goal_handle, result, ticket)
            self.get_logger().error(f"Environment VLM request failed: {exc}")
            goal_handle.abort()
            result.success = False
            result.scene = ""
            result.objects = []
            result.raw_response = ""
            result.error_message = str(exc)
            return result
        finally:
            for image in images:
                image.close()
            if ticket is not None:
                self._broker.complete(ticket)
            if token_registered:
                self._cancel_tokens.unregister(goal_handle, token)

    def _wait_for_turn(self, goal_handle, ticket: InferenceTicket) -> bool:
        while not ticket.ready_event.wait(0.1):
            if goal_handle.is_cancel_requested:
                self._broker.cancel(ticket, "cancelled by action client")
        return (
            ticket.state == TicketState.ACTIVE
            and not goal_handle.is_cancel_requested
        )

    def _raise_if_cancelled(self, goal_handle, ticket: InferenceTicket) -> None:
        if goal_handle.is_cancel_requested:
            self._broker.cancel(ticket, "cancelled by action client")
        if ticket.cancel_event.is_set():
            raise GenerationCancelled(ticket.reason or "generation cancelled")

    def _generate(self, images, system_prompt, user_prompt, token_limit, token):
        response = self._backend.generate(
            GenerationRequest(
                images=images,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=token_limit,
                cancel_event=token,
            )
        ).strip()
        if not response:
            raise RuntimeError("VLM backend returned an empty response")
        return response

    def _finish_run_without_inference(self, goal_handle, result, ticket):
        if goal_handle.is_cancel_requested or ticket.state == TicketState.CANCELLED:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        result.success = False
        result.response_text = ""
        result.error_message = ticket.reason or "VLM request was not scheduled"
        return result

    def _finish_environment_without_inference(self, goal_handle, result, ticket):
        if goal_handle.is_cancel_requested or ticket.state == TicketState.CANCELLED:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        result.success = False
        result.scene = ""
        result.objects = []
        result.raw_response = ""
        result.error_message = ticket.reason or "VLM request was not scheduled"
        return result

    def _cancel_or_preempt_run(self, goal_handle, result, ticket):
        client_cancel = goal_handle.is_cancel_requested
        if client_cancel:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        result.success = False
        result.response_text = ""
        result.error_message = (
            "VLM request cancelled by client"
            if client_cancel
            else (ticket.reason or "VLM request preempted")
        )
        return result

    def _cancel_or_preempt_environment(self, goal_handle, result, ticket):
        client_cancel = goal_handle.is_cancel_requested
        if client_cancel:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        result.success = False
        result.scene = ""
        result.objects = []
        result.raw_response = ""
        result.error_message = (
            "Environment request cancelled by client"
            if client_cancel
            else (ticket.reason or "Environment request preempted")
        )
        return result

    def _publish_raw_response(self, response: str) -> None:
        self._raw_response_pub.publish(String(data=response))

    @staticmethod
    def _run_priority(event_type: str) -> int:
        if event_type in {"voice", "voice_motion"}:
            return VOICE_PRIORITY
        return MOTION_PRIORITY

    @staticmethod
    def _goal_key(goal_handle) -> str:
        goal_id = getattr(goal_handle, "goal_id", None)
        uuid = getattr(goal_id, "uuid", None)
        return str(id(goal_handle)) if uuid is None else bytes(uuid).hex()

    @staticmethod
    def _publish_run_feedback(goal_handle, status: str) -> None:
        feedback = RunVlm.Feedback()
        feedback.status = status
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _publish_environment_feedback(goal_handle, status: str) -> None:
        feedback = AnalyzeEnvironment.Feedback()
        feedback.status = status
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _semantic_message(value):
        from robot_interfaces.msg import SemanticObject

        message = SemanticObject()
        message.detection_id = value.detection_id
        message.label = value.label
        message.description = value.description
        message.attribute_keys = [key for key, _ in value.attributes]
        message.attribute_values = [item for _, item in value.attributes]
        message.relationships = list(value.relationships)
        message.useful = value.useful
        message.confidence = value.confidence
        return message

    def _integer_parameter(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    def _string_parameter(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _boolean_parameter(self, name: str) -> bool:
        return self.get_parameter(name).get_parameter_value().bool_value

    def destroy_node(self) -> None:
        if hasattr(self, "_run_action_server"):
            self._run_action_server.destroy()
        if hasattr(self, "_environment_action_server"):
            self._environment_action_server.destroy()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node: Optional[VlmNode] = None
    executor: Optional[MultiThreadedExecutor] = None
    try:
        node = VlmNode()
        executor = MultiThreadedExecutor(num_threads=4)
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
