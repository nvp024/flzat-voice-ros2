#!/usr/bin/env python3
"""Validate and benchmark an offline Silero VAD sequence ONNX model."""

import argparse
import statistics
import time
import wave
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import onnxruntime as ort


REPOSITORY = Path(__file__).resolve().parents[2]
RATE_CONFIG = {
    8000: (256, 32),
    16000: (512, 64),
}
STATE_SHAPE = (2, 1, 128)


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def frame_blocks(
    audio: np.ndarray,
    frame_samples: int,
    context_samples: int,
    max_frames: int,
) -> Iterator[np.ndarray]:
    frame_count = (audio.size + frame_samples - 1) // frame_samples
    previous_context = np.zeros(context_samples, dtype=np.float32)
    for first_frame in range(0, frame_count, max_frames):
        block_frames = min(max_frames, frame_count - first_frame)
        first_sample = first_frame * frame_samples
        samples = audio[
            first_sample : min(audio.size, (first_frame + block_frames) * frame_samples)
        ]
        frames = np.zeros((block_frames, frame_samples), dtype=np.float32)
        frames.reshape(-1)[: samples.size] = samples

        contexts = np.empty((block_frames, context_samples), dtype=np.float32)
        contexts[0] = previous_context
        if block_frames > 1:
            contexts[1:] = frames[:-1, -context_samples:]
        previous_context = frames[-1, -context_samples:].copy()
        yield np.concatenate((contexts, frames), axis=1)


def run_stock(
    model: ort.InferenceSession,
    audio: np.ndarray,
    sample_rate: int,
    initial_state: np.ndarray,
    max_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    frame_samples, context_samples = RATE_CONFIG[sample_rate]
    state = initial_state.copy()
    probabilities = []
    rate = np.asarray(sample_rate, dtype=np.int64)
    for block in frame_blocks(audio, frame_samples, context_samples, max_frames):
        for model_input in block:
            probability, state = model.run(
                ["output", "stateN"],
                {
                    "input": model_input.reshape(1, -1),
                    "state": state,
                    "sr": rate,
                },
            )
            probabilities.append(float(probability[0, 0]))
    return np.asarray(probabilities, dtype=np.float32), state


def run_sequence(
    model: ort.InferenceSession,
    audio: np.ndarray,
    sample_rate: int,
    initial_state: np.ndarray,
    max_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    frame_samples, context_samples = RATE_CONFIG[sample_rate]
    hidden = initial_state[0:1].copy()
    cell = initial_state[1:2].copy()
    probabilities = []
    for block in frame_blocks(audio, frame_samples, context_samples, max_frames):
        values, hidden, cell = model.run(
            ["speech_probs", "hn", "cn"],
            {"input": block, "h": hidden, "c": cell},
        )
        probabilities.append(values)
    return np.concatenate(probabilities), np.concatenate((hidden, cell), axis=0)


def read_wav(path: Path, sample_rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        if (
            source.getframerate() != sample_rate
            or source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getcomptype() != "NONE"
        ):
            raise ValueError(
                "%s must be uncompressed %d Hz mono PCM16" % (path, sample_rate)
            )
        audio = np.frombuffer(
            source.readframes(source.getnframes()),
            dtype="<i2",
        )
    return audio.astype(np.float32) / 32768.0


def parse_args() -> argparse.Namespace:
    data = REPOSITORY / "src" / "silero_vad" / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_model", type=Path)
    parser.add_argument(
        "--stock-model",
        type=Path,
        default=data / "silero_vad.onnx",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=sorted(RATE_CONFIG),
        default=16000,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--wav", type=Path, help="uncompressed mono PCM16 WAV")
    source.add_argument(
        "--seconds",
        type=float,
        default=60.01,
        help="duration of deterministic synthetic input (default: 60.01)",
    )
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--trials", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_frames < 1 or args.trials < 1:
        raise SystemExit("--max-frames and --trials must be positive")
    if args.wav:
        audio = read_wav(args.wav.resolve(strict=True), args.sample_rate)
    else:
        if args.seconds <= 0:
            raise SystemExit("--seconds must be positive")
        random = np.random.default_rng(17 + args.sample_rate)
        audio = (
            random.standard_normal(round(args.seconds * args.sample_rate)) * 0.03
        ).astype(np.float32)
    if audio.size == 0:
        raise SystemExit("input audio is empty")

    stock = session(args.stock_model.resolve(strict=True))
    sequence = session(args.sequence_model.resolve(strict=True))
    random = np.random.default_rng(29 + args.sample_rate)
    initial_state = (random.standard_normal(STATE_SHAPE) * 0.01).astype(np.float32)

    frame_samples = RATE_CONFIG[args.sample_rate][0]
    warmup = audio[: min(audio.size, 16 * frame_samples)]
    run_stock(stock, warmup, args.sample_rate, initial_state, args.max_frames)
    run_sequence(sequence, warmup, args.sample_rate, initial_state, args.max_frames)

    runners = {
        "stock": (run_stock, stock),
        "sequence": (run_sequence, sequence),
    }
    times = {name: [] for name in runners}
    outputs = {}
    for trial in range(args.trials):
        order = ("stock", "sequence") if trial % 2 == 0 else ("sequence", "stock")
        for name in order:
            function, model = runners[name]
            started = time.perf_counter()
            outputs[name] = function(
                model,
                audio,
                args.sample_rate,
                initial_state,
                args.max_frames,
            )
            times[name].append(time.perf_counter() - started)

    expected_probabilities, expected_state = outputs["stock"]
    actual_probabilities, actual_state = outputs["sequence"]
    comparisons = {}
    for label, actual_values, expected_values in (
        ("probability", actual_probabilities, expected_probabilities),
        ("state", actual_state, expected_state),
    ):
        if actual_values.shape != expected_values.shape:
            raise SystemExit(
                "%s shape mismatch: %s != %s"
                % (label, actual_values.shape, expected_values.shape)
            )
        difference = float(np.max(np.abs(actual_values - expected_values)))
        comparisons[label] = (
            difference,
            np.array_equal(actual_values, expected_values),
        )
        if not np.allclose(actual_values, expected_values, rtol=0, atol=1e-6):
            raise SystemExit(
                "%s mismatch: max abs difference %.9g" % (label, difference)
            )

    stock_median = statistics.median(times["stock"])
    sequence_median = statistics.median(times["sequence"])
    print(
        "Validated %d frames at %d Hz: max abs diff %.3g, bit-exact=%s"
        % (
            actual_probabilities.size,
            args.sample_rate,
            max(item[0] for item in comparisons.values()),
            all(item[1] for item in comparisons.values()),
        )
    )
    print("Stock ONNX median:    %.4f s" % stock_median)
    print("Sequence ONNX median: %.4f s" % sequence_median)
    print("Speedup:              %.2fx" % (stock_median / sequence_median))


if __name__ == "__main__":
    main()
