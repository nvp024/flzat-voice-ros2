"""Terminal dashboard for logs from the companion robot pipeline."""

import argparse
import sys
from collections import deque
from datetime import datetime
from typing import Optional

import rclpy
from rcl_interfaces.msg import Log
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args

from rich.console import Console, Group
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text


PANELS = {
    "vad_node": {
        "title": "VAD",
        "color": "cyan",
    },
    "stt_node": {
        "title": "STT",
        "color": "green",
    },
    "multimodal_manager": {
        "title": "MULTIMODAL",
        "color": "yellow",
    },
    "vlm_node": {
        "title": "VLM",
        "color": "magenta",
    },
    "tts_node": {
        "title": "TTS",
        "color": "blue",
    },
    "camera_node": {
        "title": "CAMERA / MOTION",
        "color": "bright_red",
    },
}

# Repetitive health messages hidden by default during interaction tracking.
DEFAULT_IGNORED = {
    "camera_node": (
        "Vision health:",
    ),
}

IMPORTANT_KEYWORDS = (
    "Voice detected",
    "Published segment",
    "Transcript:",
    "usable final transcript",
    "Multimodal input:",
    "Dispatching VLM",
    "VLM input:",
    "VLM response",
    "Sending speech",
    "TTS goal accepted",
    "TTS executing",
    "TTS goal completed",
    "TTS active",
    "TTS inactive",
    "Motion started",
    "Motion event",
    "held for voice fusion",
    "suppressed:",
)


def parse_args(arguments: list[str]) -> argparse.Namespace:
    """Parse viewer options after ROS-specific arguments are removed."""

    parser = argparse.ArgumentParser(
        description="Live ROS 2 multimodal /rosout viewer"
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Show the ROS timestamp before each log line.",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=7,
        help="Recent logical lines retained per node (default: 7).",
    )
    parser.add_argument(
        "--show-health",
        action="store_true",
        help="Include repetitive camera Vision health logs.",
    )
    options = parser.parse_args(arguments)
    if options.lines < 1:
        parser.error("--lines must be at least 1")
    return options


class RosMultimodalLogViewer(Node):
    """Collect selected ROS logs in one bounded buffer per pipeline node."""

    def __init__(self, args: argparse.Namespace) -> None:
        # Avoid publishing the viewer's own messages back into /rosout.
        super().__init__("multimodal_log_viewer", enable_rosout=False)

        self.args = args
        self.buffers = {
            logger: deque(maxlen=max(1, args.lines))
            for logger in PANELS
        }

        # VOLATILE displays only messages received after the viewer starts.
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=200,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(Log, "/rosout", self.on_log, qos)

    @staticmethod
    def resolve_logger(logger_name: str) -> Optional[str]:
        """Map an exact or namespaced ROS logger to a dashboard panel."""

        # Prefer an exact logger-name match.
        if logger_name in PANELS:
            return logger_name

        # Also support node loggers carrying a ROS namespace prefix.
        for key in PANELS:
            if logger_name.endswith(key):
                return key

        return None

    def should_ignore(self, logger: str, message: str) -> bool:
        if self.args.show_health:
            return False

        for token in DEFAULT_IGNORED.get(logger, ()):
            if token in message:
                return True

        return False

    def format_timestamp(self, msg: Log) -> str:
        if not self.args.timestamp:
            return ""

        try:
            ts = msg.stamp.sec + msg.stamp.nanosec / 1_000_000_000
            return datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3] + "  "
        except Exception:
            return ""

    def on_log(self, msg: Log) -> None:
        logger = self.resolve_logger(msg.name)
        if logger is None:
            return

        message = msg.msg.strip()
        if not message or self.should_ignore(logger, message):
            return

        timestamp = self.format_timestamp(msg)
        self.buffers[logger].append(
            {
                "text": timestamp + message,
                "level": msg.level,
                "important": any(k in message for k in IMPORTANT_KEYWORDS),
            }
        )

    def make_panel(self, logger: str) -> Panel:
        config = PANELS[logger]
        items = self.buffers[logger]

        rendered_lines = []

        if not items:
            rendered_lines.append(
                Text("Waiting for log…", style="dim")
            )
        else:
            for item in items:
                if item["level"] >= Log.ERROR:
                    style = "bold red"
                    prefix = "✖ "
                elif item["level"] >= Log.WARN:
                    style = "bold yellow"
                    prefix = "⚠ "
                elif item["important"]:
                    style = f"bold {config['color']}"
                    prefix = "● "
                else:
                    style = config["color"]
                    prefix = "  "

                line = Text(prefix + item["text"], style=style)
                line.no_wrap = False
                rendered_lines.append(line)

        return Panel(
            Group(*rendered_lines),
            title=f"[bold {config['color']}]{config['title']}[/]",
            border_style=config["color"],
            padding=(0, 1),
        )

    def build_layout(self) -> Layout:
        layout = Layout()

        layout.split_column(
            Layout(name="top"),
            Layout(name="middle"),
            Layout(name="bottom"),
        )

        layout["top"].split_row(
            Layout(name="vad"),
            Layout(name="stt"),
        )
        layout["middle"].split_row(
            Layout(name="multimodal"),
            Layout(name="vlm"),
        )
        layout["bottom"].split_row(
            Layout(name="tts"),
            Layout(name="camera"),
        )

        layout["vad"].update(self.make_panel("vad_node"))
        layout["stt"].update(self.make_panel("stt_node"))
        layout["multimodal"].update(self.make_panel("multimodal_manager"))
        layout["vlm"].update(self.make_panel("vlm_node"))
        layout["tts"].update(self.make_panel("tts_node"))
        layout["camera"].update(self.make_panel("camera_node"))

        return layout


def main(args=None) -> None:
    """Run the dashboard until ROS shuts down or the user presses Ctrl-C."""

    raw_arguments = sys.argv if args is None else [sys.argv[0], *args]
    viewer_arguments = remove_ros_args(args=raw_arguments)[1:]
    options = parse_args(viewer_arguments)

    rclpy.init(
        args=raw_arguments,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    viewer: Optional[RosMultimodalLogViewer] = None

    console = Console()

    try:
        viewer = RosMultimodalLogViewer(options)
        with Live(
            viewer.build_layout(),
            console=console,
            screen=True,
            refresh_per_second=10,
        ) as live:
            while rclpy.ok():
                rclpy.spin_once(viewer, timeout_sec=0.05)
                live.update(viewer.build_layout())
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
