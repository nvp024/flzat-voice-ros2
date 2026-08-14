from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("camera_index", default_value="0"),
        DeclareLaunchArgument("camera_width", default_value="640"),
        DeclareLaunchArgument("camera_height", default_value="480"),
        DeclareLaunchArgument("camera_fps", default_value="15.0"),
        DeclareLaunchArgument("min_motion_ratio", default_value="0.02"),
        DeclareLaunchArgument("pixel_threshold", default_value="25"),
        DeclareLaunchArgument("debug_fps", default_value="5.0"),
        DeclareLaunchArgument("motion_start_frames", default_value="3"),
        DeclareLaunchArgument("motion_end_s", default_value="0.7"),
        DeclareLaunchArgument("settle_s", default_value="0.3"),
        DeclareLaunchArgument("save_event_images", default_value="false"),
        DeclareLaunchArgument(
            "event_image_dir", default_value="/tmp/vision_events"
        ),
        Node(
            package="vision_pipeline",
            executable="camera_node",
            name="camera_node",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "camera_index": ParameterValue(
                    LaunchConfiguration("camera_index"), value_type=int
                ),
                "camera_width": ParameterValue(
                    LaunchConfiguration("camera_width"), value_type=int
                ),
                "camera_height": ParameterValue(
                    LaunchConfiguration("camera_height"), value_type=int
                ),
                "camera_fps": ParameterValue(
                    LaunchConfiguration("camera_fps"), value_type=float
                ),
                "min_motion_ratio": ParameterValue(
                    LaunchConfiguration("min_motion_ratio"), value_type=float
                ),
                "pixel_threshold": ParameterValue(
                    LaunchConfiguration("pixel_threshold"), value_type=int
                ),
                "debug_fps": ParameterValue(
                    LaunchConfiguration("debug_fps"), value_type=float
                ),
                "motion_start_frames": ParameterValue(
                    LaunchConfiguration("motion_start_frames"), value_type=int
                ),
                "motion_end_s": ParameterValue(
                    LaunchConfiguration("motion_end_s"), value_type=float
                ),
                "settle_s": ParameterValue(
                    LaunchConfiguration("settle_s"), value_type=float
                ),
                "save_event_images": ParameterValue(
                    LaunchConfiguration("save_event_images"), value_type=bool
                ),
                "event_image_dir": ParameterValue(
                    LaunchConfiguration("event_image_dir"), value_type=str
                ),
            }],
        ),
    ])
