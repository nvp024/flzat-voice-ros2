# FLZAT Companion Robot ROS 2

This workspace contains a V0.1 real-time companion robot pipeline combining
microphone audio, camera images, a vision-language model (VLM), and speech
output. Version 1.1 is being added incrementally; Parts 1 and 2 implement the
usable-final-STT barrier, bounded scheduling, and cooperative VLM cancellation.

## Pipeline

```text
Microphone → VAD → STT ───────────────┐
                                      ├→ Multimodal Manager → VLM → TTS → Speaker
Camera → Motion Detection → Frame Buffer ┘
```

The manager creates `voice`, `motion`, or `voice_motion` events using timestamps
to synchronize the transcript with one relevant camera frame. Camera capture
continues independently while VLM and TTS requests are processed.

## Packages

- `audio_pipeline`: microphone VAD, Whisper STT, and TTS.
- `vision_pipeline`: camera capture, motion detection, and frame ring buffer.
- `vlm_pipeline`: replaceable VLM action server and prompt profiles.
- `trigger_engine`: audio/visual fusion and asynchronous VLM/TTS coordination.
- `robot_interfaces`: custom ROS messages, services, and actions.

## Build

```bash
cd /home/phucnv/Documents/JTR/High_system/flzat_robot_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Activate the Python environment containing PyTorch, Transformers, Whisper, and
the audio dependencies before building or launching.

## Run the complete companion pipeline

```bash
ros2 launch trigger_engine companion_pipeline.launch.py
```

The default VLM is:

```text
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

Override it without changing source code:

```bash
ros2 launch trigger_engine companion_pipeline.launch.py \
  model_id:=HuggingFaceTB/SmolVLM2-256M-Video-Instruct
```

## Run individual pipelines

Audio loopback test:

```bash
ros2 launch audio_pipeline voice_pipeline.launch.py
```

Vision test:

```bash
ros2 launch vision_pipeline vision_pipeline.launch.py
```

## Monitor

Open the ROS log GUI:

```bash
ros2 run rqt_console rqt_console
```

View camera or selected multimodal frames:

```bash
ros2 run rqt_image_view rqt_image_view
```

Select `/vision/debug` or `/multimodal/mock_frame` in the image viewer.

Inspect VLM output:

```bash
ros2 topic echo /vlm/raw_response std_msgs/msg/String \
  --qos-durability transient_local

ros2 topic echo /multimodal/vlm_response std_msgs/msg/String
```

## Documentation

- `docs/PIPELINE_STATUS.md`: Version 0.1 status and architecture.
- `docs/COMPANION_PIPELINE_TESTING.md`: full pipeline testing.
- `docs/MULTIMODAL_TESTING.md`: camera and voice fusion testing.
- `docs/VLM_TESTING.md`: standalone VLM testing.
- `docs/PLANNING_v1.1.md`: incremental Version 1.1 design and acceptance plan.
- `docs/V1.1_PART1_TESTING.md`: Part 1 behavior and test procedure.
- `docs/V1.1_PART2_TESTING.md`: VLM cancellation and preemption testing.
