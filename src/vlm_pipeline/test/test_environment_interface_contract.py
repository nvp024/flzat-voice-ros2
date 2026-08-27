from pathlib import Path


SRC_ROOT = Path(__file__).parents[2]


def test_environment_interfaces_are_generated():
    interfaces = SRC_ROOT / "robot_interfaces"
    cmake = (interfaces / "CMakeLists.txt").read_text(encoding="utf-8")
    action = (interfaces / "action" / "AnalyzeEnvironment.action").read_text(
        encoding="utf-8"
    )

    assert '"msg/ObjectDetection2D.msg"' in cmake
    assert '"msg/SemanticObject.msg"' in cmake
    assert '"action/AnalyzeEnvironment.action"' in cmake
    assert "sensor_msgs/CompressedImage image" in action
    assert "robot_interfaces/ObjectDetection2D[] detections" in action
    assert "string raw_response" in action


def test_reusable_speech_launch_has_no_loopback_node():
    launch = (
        SRC_ROOT / "audio_pipeline" / "launch" / "speech_services.launch.py"
    ).read_text(encoding="utf-8")

    assert 'executable="vad_node"' in launch
    assert 'executable="stt_node"' in launch
    assert 'executable="tts_node"' in launch
    assert "audio_loopback_node" not in launch


def test_environment_action_performs_validation_and_exactly_one_repair_path():
    node = (
        SRC_ROOT / "vlm_pipeline" / "vlm_pipeline" / "vlm_node.py"
    ).read_text(encoding="utf-8")
    schema = (
        SRC_ROOT / "vlm_pipeline" / "vlm_pipeline" / "environment_schema.py"
    ).read_text(encoding="utf-8")
    repair = (
        SRC_ROOT
        / "vlm_pipeline"
        / "prompts"
        / "environment_memory_v1"
        / "repair.txt"
    ).read_text(encoding="utf-8")

    environment_execute = node.split("def _execute_environment", 1)[1].split(
        "def _wait_for_turn", 1
    )[0]
    assert environment_execute.count("build_repair(") == 1
    assert environment_execute.count("parse_environment_response(") == 2
    assert '"validating"' in environment_execute
    assert '"retrying"' in environment_execute
    assert "supplied by the detector" in schema
    assert "Do not add" in repair
