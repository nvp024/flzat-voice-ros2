import argparse

import pytest

from trigger_engine.multimodal_log_viewer import (
    RosMultimodalLogViewer,
    parse_args,
)


def _viewer(show_health: bool = False) -> RosMultimodalLogViewer:
    viewer = RosMultimodalLogViewer.__new__(RosMultimodalLogViewer)
    viewer.args = argparse.Namespace(show_health=show_health)
    return viewer


def test_parse_args_accepts_display_options() -> None:
    options = parse_args(["--timestamp", "--lines", "12", "--show-health"])

    assert options.timestamp is True
    assert options.lines == 12
    assert options.show_health is True


def test_parse_args_rejects_empty_buffers() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--lines", "0"])


def test_namespaced_loggers_resolve_to_pipeline_panels() -> None:
    assert RosMultimodalLogViewer.resolve_logger("vad_node") == "vad_node"
    assert (
        RosMultimodalLogViewer.resolve_logger("robot.front.vlm_node")
        == "vlm_node"
    )
    assert RosMultimodalLogViewer.resolve_logger("unrelated_node") is None


def test_camera_health_filter_can_be_overridden() -> None:
    assert _viewer().should_ignore(
        "camera_node",
        "Vision health: capture=15.0 FPS",
    )
    assert not _viewer(show_health=True).should_ignore(
        "camera_node",
        "Vision health: capture=15.0 FPS",
    )
