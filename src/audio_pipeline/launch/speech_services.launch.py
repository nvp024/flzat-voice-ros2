"""Reusable speech services without the VAD -> STT -> TTS loopback client."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "vad_silence_ms",
                default_value="500",
                description="Silence required to finish an utterance.",
            ),
            DeclareLaunchArgument(
                "whisper_language",
                default_value="en",
                description="Whisper language code; empty enables auto detection.",
            ),
            Node(
                package="audio_pipeline",
                executable="vad_node",
                name="vad_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "silence_ms": ParameterValue(
                            LaunchConfiguration("vad_silence_ms"),
                            value_type=int,
                        )
                    }
                ],
            ),
            Node(
                package="audio_pipeline",
                executable="stt_node",
                name="stt_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "language": ParameterValue(
                            LaunchConfiguration("whisper_language"),
                            value_type=str,
                        )
                    }
                ],
            ),
            Node(
                package="audio_pipeline",
                executable="tts_node",
                name="tts_node",
                output="screen",
                emulate_tty=True,
            ),
        ]
    )
