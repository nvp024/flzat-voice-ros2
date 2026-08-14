from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    camera_arguments = {
        "camera_index": LaunchConfiguration("camera_index"),
        "camera_width": LaunchConfiguration("camera_width"),
        "camera_height": LaunchConfiguration("camera_height"),
        "camera_fps": LaunchConfiguration("camera_fps"),
        "min_motion_ratio": LaunchConfiguration("min_motion_ratio"),
        "pixel_threshold": LaunchConfiguration("pixel_threshold"),
        "debug_fps": LaunchConfiguration("debug_fps"),
        "motion_start_frames": LaunchConfiguration("motion_start_frames"),
        "motion_end_s": LaunchConfiguration("motion_end_s"),
        "settle_s": LaunchConfiguration("settle_s"),
        "save_event_images": LaunchConfiguration("save_event_images"),
        "event_image_dir": LaunchConfiguration("event_image_dir"),
    }
    vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                get_package_share_directory("vision_pipeline"),
                "/launch/vision_pipeline.launch.py",
            ]
        ),
        launch_arguments=camera_arguments.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument("camera_index", default_value="0"),
        DeclareLaunchArgument("camera_width", default_value="640"),
        DeclareLaunchArgument("camera_height", default_value="480"),
        DeclareLaunchArgument("camera_fps", default_value="15.0"),
        DeclareLaunchArgument("min_motion_ratio", default_value="0.02"),
        DeclareLaunchArgument("pixel_threshold", default_value="25"),
        DeclareLaunchArgument("debug_fps", default_value="5.0"),
        DeclareLaunchArgument("motion_start_frames", default_value="3"),
        DeclareLaunchArgument("motion_end_s", default_value="0.5"),
        DeclareLaunchArgument("settle_s", default_value="0.2"),
        DeclareLaunchArgument("save_event_images", default_value="false"),
        DeclareLaunchArgument("event_image_dir", default_value="/tmp/vision_events"),
        DeclareLaunchArgument("vad_silence_ms", default_value="400"),
        DeclareLaunchArgument("whisper_language", default_value="en"),
        DeclareLaunchArgument("motion_hold_s", default_value="0.7"),
        DeclareLaunchArgument("voice_fusion_wait_s", default_value="0.8"),
        DeclareLaunchArgument("overlap_tolerance_s", default_value="0.25"),
        vision_launch,
        Node(
            package="audio_pipeline",
            executable="vad_node",
            name="vad_node",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "silence_ms": ParameterValue(
                    LaunchConfiguration("vad_silence_ms"), value_type=int
                ),
            }],
        ),
        Node(
            package="audio_pipeline",
            executable="stt_node",
            name="stt_node",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "language": ParameterValue(
                    LaunchConfiguration("whisper_language"), value_type=str
                ),
            }],
        ),
        Node(
            package="trigger_engine",
            executable="multimodal_manager",
            name="multimodal_manager",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "mode": "mock",
                "motion_hold_s": ParameterValue(
                    LaunchConfiguration("motion_hold_s"), value_type=float
                ),
                "voice_fusion_wait_s": ParameterValue(
                    LaunchConfiguration("voice_fusion_wait_s"), value_type=float
                ),
                "overlap_tolerance_s": ParameterValue(
                    LaunchConfiguration("overlap_tolerance_s"), value_type=float
                ),
            }],
        ),
    ])
