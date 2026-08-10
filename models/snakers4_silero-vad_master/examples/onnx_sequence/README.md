# Offline ONNX Sequence Inference

This optional example reduces ONNX Runtime call overhead when processing long offline recordings. It exports a graph that evaluates consecutive Silero VAD frames in one call while preserving the temporal LSTM state.

The sequence axis represents consecutive frames from one recording. It is not a batch of independent recordings, and this example does not change the package loader, streaming API, or shipped models.

## Setup

Run the commands from the repository root. The exporter requires PyTorch and ONNX; the runner requires NumPy and ONNX Runtime:

```bash
python -m pip install torch onnx onnxruntime numpy
```

## Export

Generate the model locally from the packaged TorchScript weights:

```bash
python examples/onnx_sequence/export.py \
  --sample-rate 16000 \
  --output silero_vad_16k_sequence.onnx
```

Use `--sample-rate 8000` to export the 8 kHz architecture. Generated ONNX files are not part of the package and should not be committed.

The exporter reconstructs the public v5/v6 architecture, validates expected weight names and shapes, uses a dynamic sequence axis, and runs the ONNX model checker. The resulting model accepts:

```text
input: [sequence_length, frame_samples + context_samples]
h:     [1, 1, 128]
c:     [1, 1, 128]
```

It returns one speech probability per frame and the final hidden and cell states. The states and the previous frame's audio context must be carried into the next block.

## Validate and benchmark

`run.py` implements a bounded runner, compares it with the stock frame-at-a-time ONNX model, and exits with an error if probabilities or final recurrent state differ by more than `1e-6`. It also reports whether the results are array-exact:

```bash
python examples/onnx_sequence/run.py \
  silero_vad_16k_sequence.onnx \
  --sample-rate 16000 \
  --seconds 60.01 \
  --trials 5
```

You can validate an uncompressed mono PCM16 WAV with `--wav path/to/audio.wav`. The default 512-frame block covers 16.384 seconds at either supported sample rate and keeps model inputs bounded for recordings of any length.

This example returns frame probabilities. Applications that need speech timestamps should feed those probabilities into their offline timestamp logic; the core `get_speech_timestamps` API remains unchanged.

## Reference results

The submitted `run.py` was measured on one pinned AMD Ryzen 5 5600X core with ONNX Runtime 1.27.0 using `CPUExecutionProvider` and one runtime thread. Timings include framing and probability inference; they exclude audio decoding, model export, session creation, and timestamp post-processing. The long input was this [69m20.679s recording](https://www.youtube.com/watch?v=Cdle09QLPLI), decoded once to 16 kHz mono PCM16 before timing.

| Input | Stock frame ONNX | Sequence ONNX | Speedup | Output |
| --- | ---: | ---: | ---: | --- |
| 69m20.679s, 16 kHz | 11.8733 s | 4.8382 s | 2.45x | exact |
| Repository 60s fixture, 16 kHz | 0.1651 s | 0.0705 s | 2.34x | exact |
| Repository fixture resampled to 8 kHz | 0.1363 s | 0.0425 s | 3.21x | exact |

Each result is the median of five alternating warmed trials. Timing varies by CPU and ONNX Runtime version; no timing threshold is used as a correctness check.

The temporal-sequence approach was proposed in [discussion #408](https://github.com/snakers4/silero-vad/discussions/408) and later used by [faster-whisper](https://github.com/SYSTRAN/faster-whisper/pull/936). Bounded blocks follow the long-input handling added in [faster-whisper #1198](https://github.com/SYSTRAN/faster-whisper/pull/1198), and the v6 architecture follows [faster-whisper #1390](https://github.com/SYSTRAN/faster-whisper/pull/1390).
