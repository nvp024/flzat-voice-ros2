from pathlib import Path


SRC_ROOT = Path(__file__).parents[2]


def test_ground_objects_action_is_separate_and_traceable() -> None:
    interfaces = SRC_ROOT / "robot_interfaces"
    cmake = (interfaces / "CMakeLists.txt").read_text(encoding="utf-8")
    action = (interfaces / "action" / "GroundObjects.action").read_text(
        encoding="utf-8"
    )
    node = (
        SRC_ROOT / "vlm_pipeline" / "vlm_pipeline" / "vlm_node.py"
    ).read_text(encoding="utf-8")

    assert '"action/GroundObjects.action"' in cmake
    assert "string observation_id" in action
    assert "sensor_msgs/CompressedImage image" in action
    assert "preferred_labels" not in action
    assert "allow_unlisted" not in action
    assert "robot_interfaces/ObjectDetection2D[] detections" in action
    assert "string model_revision" in action
    assert "string prompt_version" in action
    assert '"/vlm/ground_objects"' in node
    assert "GROUNDING_PRIORITY = 5" in node
    assert "MOTION_PRIORITY = 10" in node
    assert "VOICE_PRIORITY = 30" in node


def test_grounding_has_no_repair_retry_and_uses_strict_parser() -> None:
    node = (
        SRC_ROOT / "vlm_pipeline" / "vlm_pipeline" / "vlm_node.py"
    ).read_text(encoding="utf-8")
    execution = node.split("def _execute_grounding", 1)[1].split(
        "def _wait_for_turn", 1
    )[0]

    assert execution.count("parse_grounding_response(") == 1
    assert "build_repair" not in execution
    assert '"validating"' in execution
