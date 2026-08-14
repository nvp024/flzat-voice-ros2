from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CompressedImage

from PIL import Image
from robot_interfaces.action import RunVlm
from robot_interfaces.msg import MultimodalEvent


class VlmTestClient(Node):
    """One-shot action client for a local image and English prompt."""

    def __init__(self) -> None:
        super().__init__("vlm_test_client")
        self.client = ActionClient(self, RunVlm, "/vlm/run")


def _parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a static image and prompt to the standalone VLM node."
    )
    parser.add_argument("--image", required=True, help="Local image path")
    parser.add_argument("--prompt", required=True, help="English VLM prompt")
    parser.add_argument(
        "--server-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for /vlm/run",
    )
    parser.add_argument(
        "--result-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for model inference",
    )
    return parser.parse_args(arguments)


def _compressed_image(path: Path, node: Node) -> CompressedImage:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        output = BytesIO()
        rgb.save(output, format="JPEG", quality=90)
        width, height = rgb.size
    message = CompressedImage()
    message.header.stamp = node.get_clock().now().to_msg()
    message.header.frame_id = "static_test_image"
    message.format = "rgb8; jpeg compressed rgb8"
    message.data = output.getvalue()
    node.get_logger().info(
        f"Prepared image: {path} ({width}x{height}, {len(message.data)} bytes)."
    )
    return message


def _goal(image: CompressedImage, prompt: str) -> RunVlm.Goal:
    payload = MultimodalEvent()
    payload.header = image.header
    payload.event_type = "standalone_test"
    payload.transcript = prompt.strip()
    payload.trigger_reason = "static_image_test"
    payload.frames = [image]
    goal = RunVlm.Goal()
    goal.input = payload
    return goal


def _feedback(message) -> None:
    print(f"VLM status: {message.feedback.status}")


def _run_request(node: VlmTestClient, options: argparse.Namespace) -> int:
    image = _compressed_image(Path(options.image).expanduser(), node)
    if not node.client.wait_for_server(timeout_sec=options.server_timeout):
        node.get_logger().error("/vlm/run is not available.")
        return 1

    send_future = node.client.send_goal_async(
        _goal(image, options.prompt),
        feedback_callback=_feedback,
    )
    rclpy.spin_until_future_complete(
        node,
        send_future,
        timeout_sec=options.server_timeout,
    )
    if not send_future.done():
        node.get_logger().error("Timed out while sending the VLM goal.")
        return 1
    goal_handle = send_future.result()
    if goal_handle is None or not goal_handle.accepted:
        node.get_logger().error("VLM goal was rejected; the server may be busy.")
        return 1

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(
        node,
        result_future,
        timeout_sec=options.result_timeout,
    )
    if not result_future.done():
        node.get_logger().error("Timed out waiting for VLM inference.")
        goal_handle.cancel_goal_async()
        return 1
    response = result_future.result()
    result = response.result
    if response.status != GoalStatus.STATUS_SUCCEEDED or not result.success:
        node.get_logger().error(
            f"VLM request failed: {result.error_message or response.status}"
        )
        return 1
    print("\n===== VLM RESPONSE =====")
    print(result.response_text)
    return 0


def main(args=None) -> None:
    raw_arguments = sys.argv if args is None else [sys.argv[0], *args]
    cli_arguments = remove_ros_args(args=raw_arguments)[1:]
    options = _parse_arguments(cli_arguments)
    if options.server_timeout <= 0.0 or options.result_timeout <= 0.0:
        raise ValueError("Timeouts must be positive")

    rclpy.init(args=raw_arguments, signal_handler_options=SignalHandlerOptions.NO)
    node: Optional[VlmTestClient] = None
    exit_code = 1
    try:
        node = VlmTestClient()
        exit_code = _run_request(node, options)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("VLM test client interrupted.")
    except Exception as exc:
        if node is not None:
            node.get_logger().error(str(exc))
        else:
            print(f"VLM test client failed: {exc}", file=sys.stderr)
    finally:
        if node is not None:
            node.client.destroy()
            node.destroy_node()
        rclpy.try_shutdown()
    if args is None:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
