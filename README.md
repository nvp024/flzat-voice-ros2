# FLZAT Companion Robot ROS 2

This workspace contains a V0.1 real-time companion robot pipeline combining
microphone audio, camera images, a vision-language model (VLM), and speech
output. Version 1.1 is being added incrementally; Parts 1–3 implement the
usable-final-STT barrier, cooperative VLM cancellation, and relevant visual
frame refresh.

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
- `vlm_pipeline`: replaceable shared VLM server, prompt profiles, and global
  voice/environment/motion inference priority broker.
- `trigger_engine`: audio/visual fusion and asynchronous VLM/TTS coordination.
- `robot_interfaces`: custom ROS messages, services, and actions.

## Build

```bash
cd <integrate-root>/flzat-voice-ros2
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

Use the Qwen2-VL 2B backend:

```bash
ros2 launch trigger_engine companion_pipeline.launch.py \
  backend:=qwen2_vl \
  model_id:=Qwen/Qwen2-VL-2B-Instruct \
  do_image_splitting:=false \
  quantization:=none
```

Qwen2-VL is substantially larger than the default 500M SmolVLM2 checkpoint.
Use CUDA when available, or expect slower inference and greater memory use on
CPU.

## Run individual pipelines

Audio loopback test:

```bash
ros2 launch audio_pipeline voice_pipeline.launch.py
```

Vision test:

```bash
ros2 launch vision_pipeline vision_pipeline.launch.py
```

Reusable speech services without the audio loopback client:

```bash
ros2 launch audio_pipeline speech_services.launch.py
```

The shared VLM server exposes both `/vlm/run` and
`/vlm/analyze_environment`. Voice requests have priority over background
environment requests, which have priority over motion-only requests. The
server runs one inference and retains at most one latest pending request.

The environment action and prompt transport are Phase 2 foundations for the
separate environment-memory workspace. Strict JSON validation, repair retry,
and conversion into `SemanticObject` results belong to environment-memory
Phase 6; until then, successful environment inference is available in the
action's `raw_response` field.

## Monitor

Open the pipeline log dashboard in a separate sourced terminal:

```bash
ros2 run trigger_engine multimodal_log_viewer --timestamp
```

The dashboard separates VAD, STT, camera/motion, multimodal manager, VLM, and
TTS logs. It retains seven recent lines per node and hides repetitive camera
health reports by default. Use `--lines 10` to retain more lines or
`--show-health` to include camera health reports.

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
- `docs/V1.1_PART3_TESTING.md`: visual refresh and frame-selection testing.
