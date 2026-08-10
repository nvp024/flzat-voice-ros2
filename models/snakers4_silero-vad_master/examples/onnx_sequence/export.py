#!/usr/bin/env python3
"""Export a state-preserving Silero VAD model for offline sequence inference."""

import argparse
import inspect
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


REPOSITORY = Path(__file__).resolve().parents[2]
RATE_CONFIG = {
    8000: ("_model_8k", 256, 32),
    16000: ("_model", 512, 64),
}
STATE_SIZE = 128


class STFT(nn.Module):
    def __init__(self, sample_rate: int):
        super().__init__()
        filter_length = int(sample_rate / 62.5)
        self.filter_length = filter_length
        self.hop_length = filter_length // 2
        self.padding = nn.ReflectionPad1d(filter_length // 2)
        self.register_buffer(
            "forward_basis_buffer",
            torch.zeros(filter_length + 2, 1, filter_length),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        audio = self.padding(audio).unsqueeze(1)
        transform = F.conv1d(
            audio,
            self.forward_basis_buffer,
            stride=self.hop_length,
        )
        cutoff = self.filter_length // 2 + 1
        real = transform[:, :cutoff, 1:]
        imaginary = transform[:, cutoff:, 1:]
        return torch.sqrt(real.pow(2) + imaginary.pow(2))


class SequenceVad(nn.Module):
    def __init__(self, sample_rate: int):
        super().__init__()
        input_channels = int(sample_rate / 125) + 1
        self.stft = STFT(sample_rate)
        self.encoder = nn.ModuleList(
            [
                nn.Conv1d(input_channels, 128, 3, padding=1),
                nn.Conv1d(128, 64, 3, stride=2, padding=1),
                nn.Conv1d(64, 64, 3, stride=2, padding=1),
                nn.Conv1d(64, 128, 3, padding=1),
            ]
        )
        self.recurrent = nn.LSTM(STATE_SIZE, STATE_SIZE)
        self.output = nn.Conv1d(STATE_SIZE, 1, 1)

    def forward(
        self,
        audio: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.stft(audio)
        for layer in self.encoder:
            encoded = F.relu(layer(encoded))

        # Rows are consecutive time steps, not independent batch elements.
        decoded, (hidden, cell) = self.recurrent(
            encoded.permute(0, 2, 1),
            (hidden, cell),
        )
        probabilities = torch.sigmoid(self.output(F.relu(decoded).permute(1, 2, 0)))
        return probabilities.reshape(-1), hidden, cell


def weight_mapping(prefix: str) -> Dict[str, str]:
    mapping = {
        "stft.forward_basis_buffer": prefix + ".stft.forward_basis_buffer",
        "recurrent.weight_ih_l0": prefix + ".decoder.rnn.weight_ih",
        "recurrent.weight_hh_l0": prefix + ".decoder.rnn.weight_hh",
        "recurrent.bias_ih_l0": prefix + ".decoder.rnn.bias_ih",
        "recurrent.bias_hh_l0": prefix + ".decoder.rnn.bias_hh",
        "output.weight": prefix + ".decoder.decoder.2.weight",
        "output.bias": prefix + ".decoder.decoder.2.bias",
    }
    for index in range(4):
        for parameter in ("weight", "bias"):
            mapping["encoder.%d.%s" % (index, parameter)] = (
                "%s.encoder.%d.reparam_conv.%s" % (prefix, index, parameter)
            )
    return mapping


def load_weights(model: nn.Module, source: torch.jit.ScriptModule, prefix: str) -> None:
    source_state = source.state_dict()
    target_state = model.state_dict()
    mapping = weight_mapping(prefix)
    branch_names = {name for name in source_state if name.startswith(prefix + ".")}
    if branch_names != set(mapping.values()):
        raise RuntimeError(
            "Source architecture differs from this example: missing=%s unexpected=%s"
            % (
                sorted(set(mapping.values()) - branch_names),
                sorted(branch_names - set(mapping.values())),
            )
        )
    remapped = {}
    for target_name, source_name in mapping.items():
        value = source_state[source_name]
        if value.shape != target_state[target_name].shape:
            raise RuntimeError(
                "Unexpected shape for %s: expected %s, got %s"
                % (
                    source_name,
                    tuple(target_state[target_name].shape),
                    tuple(value.shape),
                )
            )
        remapped[target_name] = value
    model.load_state_dict(remapped, strict=True)


def export_model(
    source_path: Path,
    output_path: Path,
    sample_rate: int,
    opset: int,
) -> None:
    import onnx

    prefix, frame_samples, context_samples = RATE_CONFIG[sample_rate]
    source = torch.jit.load(str(source_path), map_location="cpu")
    source.eval()
    model = SequenceVad(sample_rate)
    load_weights(model, source, prefix)
    model.eval()

    frames = torch.zeros(10, frame_samples + context_samples)
    state = torch.zeros(1, 1, STATE_SIZE)
    options = {
        "input_names": ["input", "h", "c"],
        "output_names": ["speech_probs", "hn", "cn"],
        "dynamic_axes": {
            "input": {0: "sequence_length"},
            "speech_probs": {0: "sequence_length"},
        },
        "opset_version": opset,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (frames, state, state.clone()),
            str(output_path),
            **options,
        )

    exported = onnx.load(str(output_path))
    onnx.checker.check_model(exported)


def parse_args() -> argparse.Namespace:
    data = REPOSITORY / "src" / "silero_vad" / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=sorted(RATE_CONFIG),
        default=16000,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=data / "silero_vad.jit",
        help="source TorchScript model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output path (default: silero_vad_<rate>k_sequence.onnx)",
    )
    parser.add_argument("--opset", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve(strict=True)
    output = args.output or Path(
        "silero_vad_%dk_sequence.onnx" % (args.sample_rate // 1000)
    )
    export_model(source, output.resolve(), args.sample_rate, args.opset)
    print("Exported %s" % output.resolve())


if __name__ == "__main__":
    main()
