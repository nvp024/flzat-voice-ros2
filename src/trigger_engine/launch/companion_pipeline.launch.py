from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory("vision_pipeline"),
            "/launch/vision_pipeline.launch.py",
        ]),
        launch_arguments={
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
        }.items(),
    )
    vlm_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory("vlm_pipeline"),
            "/launch/vlm_server.launch.py",
        ]),
        launch_arguments={
            "backend": LaunchConfiguration("backend"),
            "model_id": LaunchConfiguration("model_id"),
            "device": LaunchConfiguration("device"),
            "dtype": LaunchConfiguration("dtype"),
            "quantization": LaunchConfiguration("quantization"),
            "max_new_tokens": LaunchConfiguration("max_new_tokens"),
            "trust_remote_code": LaunchConfiguration("trust_remote_code"),
            "local_files_only": LaunchConfiguration("local_files_only"),
            "prompt_profile": LaunchConfiguration("prompt_profile"),
            "prompt_directory": LaunchConfiguration("prompt_directory"),
            "do_image_splitting": LaunchConfiguration("do_image_splitting"),
        }.items(),
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
        DeclareLaunchArgument("voice_visual_after_s", default_value="0.75"),
        DeclareLaunchArgument("motion_vlm_cooldown_s", default_value="5.0"),
        DeclareLaunchArgument("held_response_ttl_s", default_value="10.0"),
        DeclareLaunchArgument("pending_motion_ttl_s", default_value="3.0"),
        DeclareLaunchArgument("pending_voice_ttl_s", default_value="10.0"),
        DeclareLaunchArgument("active_vlm_timeout_s", default_value="60.0"),
        DeclareLaunchArgument("vlm_cancel_grace_s", default_value="3.0"),
        DeclareLaunchArgument("backend", default_value="smolvlm2"),
        DeclareLaunchArgument(
            "model_id",
            default_value="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        ),
        DeclareLaunchArgument("device", default_value="auto"),
        DeclareLaunchArgument("dtype", default_value="auto"),
        DeclareLaunchArgument("quantization", default_value="none"),
        DeclareLaunchArgument("max_new_tokens", default_value="48"),
        DeclareLaunchArgument("trust_remote_code", default_value="false"),
        DeclareLaunchArgument("local_files_only", default_value="false"),
        DeclareLaunchArgument("prompt_profile", default_value="companion_robot_v1"),
        DeclareLaunchArgument("prompt_directory", default_value=""),
        DeclareLaunchArgument("do_image_splitting", default_value="false"),
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
            package="audio_pipeline",
            executable="tts_node",
            name="tts_node",
            output="screen",
            emulate_tty=True,
        ),
        vlm_launch,
        Node(
            package="trigger_engine",
            executable="multimodal_manager",
            name="multimodal_manager",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "mode": "vlm",
                "motion_hold_s": ParameterValue(
                    LaunchConfiguration("motion_hold_s"), value_type=float
                ),
                "voice_fusion_wait_s": ParameterValue(
                    LaunchConfiguration("voice_fusion_wait_s"), value_type=float
                ),
                "overlap_tolerance_s": ParameterValue(
                    LaunchConfiguration("overlap_tolerance_s"), value_type=float
                ),
                "voice_visual_after_s": ParameterValue(
                    LaunchConfiguration("voice_visual_after_s"), value_type=float
                ),
                "motion_vlm_cooldown_s": ParameterValue(
                    LaunchConfiguration("motion_vlm_cooldown_s"), value_type=float
                ),
                "held_response_ttl_s": ParameterValue(
                    LaunchConfiguration("held_response_ttl_s"), value_type=float
                ),
                "pending_motion_ttl_s": ParameterValue(
                    LaunchConfiguration("pending_motion_ttl_s"), value_type=float
                ),
                "pending_voice_ttl_s": ParameterValue(
                    LaunchConfiguration("pending_voice_ttl_s"), value_type=float
                ),
                "active_vlm_timeout_s": ParameterValue(
                    LaunchConfiguration("active_vlm_timeout_s"), value_type=float
                ),
                "vlm_cancel_grace_s": ParameterValue(
                    LaunchConfiguration("vlm_cancel_grace_s"), value_type=float
                ),
            }],
        ),
    ])
