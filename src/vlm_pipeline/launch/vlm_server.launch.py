from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
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
        Node(
            package="vlm_pipeline",
            executable="vlm_node",
            name="vlm_node",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "backend": ParameterValue(
                    LaunchConfiguration("backend"), value_type=str
                ),
                "model_id": ParameterValue(
                    LaunchConfiguration("model_id"), value_type=str
                ),
                "device": ParameterValue(
                    LaunchConfiguration("device"), value_type=str
                ),
                "dtype": ParameterValue(
                    LaunchConfiguration("dtype"), value_type=str
                ),
                "quantization": ParameterValue(
                    LaunchConfiguration("quantization"), value_type=str
                ),
                "max_new_tokens": ParameterValue(
                    LaunchConfiguration("max_new_tokens"), value_type=int
                ),
                "trust_remote_code": ParameterValue(
                    LaunchConfiguration("trust_remote_code"), value_type=bool
                ),
                "local_files_only": ParameterValue(
                    LaunchConfiguration("local_files_only"), value_type=bool
                ),
                "prompt_profile": ParameterValue(
                    LaunchConfiguration("prompt_profile"), value_type=str
                ),
                "prompt_directory": ParameterValue(
                    LaunchConfiguration("prompt_directory"), value_type=str
                ),
                "do_image_splitting": ParameterValue(
                    LaunchConfiguration("do_image_splitting"), value_type=bool
                ),
            }],
        ),
    ])
