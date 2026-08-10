from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # 1. VAD — listens to microphone, publishes /audio_events
        Node(
            package="audio_pipeline",
            executable="vad_node",
            name="vad_node",
            output="screen",
            emulate_tty=True,
        ),

        # 2. STT — action server /stt_action (Whisper)
        Node(
            package="audio_pipeline",
            executable="stt_node",
            name="stt_node",
            output="screen",
            emulate_tty=True,
        ),

        # 3. TTS — action server /tts_action (pyttsx3)
        Node(
            package="audio_pipeline",
            executable="tts_node",
            name="tts_node",
            output="screen",
            emulate_tty=True,
        ),

        # 4. Orchestrator — subscribes /audio_events, calls STT → TTS
        Node(
            package="trigger_engine",
            executable="audio_visual_trigger",
            name="audio_visual_trigger",
            output="screen",
            emulate_tty=True,
        ),

    ])
