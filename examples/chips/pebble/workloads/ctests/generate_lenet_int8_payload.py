#!/usr/bin/env python3
"""Generate the fixed LeNet/Pebble INT8 payload and its Python golden result."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image


MAGIC = 0x4C4E5438  # "LNT8"
VERSION = 1
TILE = 16


@dataclass(frozen=True)
class Scales:
    input: np.float32
    weight: np.float32
    output: np.float32


SCALES = {
    "conv1": Scales(
        np.float32(127.0), np.float32(217.50604248046875), np.float32(27.21164321899414)
    ),
    "conv2": Scales(
        np.float32(31.608097076416016),
        np.float32(71.5003433227539),
        np.float32(4.385692119598389),
    ),
    "fc1": Scales(
        np.float32(9.430673599243164),
        np.float32(76.45767211914062),
        np.float32(1.6892915964126587),
    ),
    "fc2": Scales(
        np.float32(2.5877368450164795),
        np.float32(106.72378540039062),
        np.float32(0.6678189635276794),
    ),
    "fc3": Scales(
        np.float32(1.084639549255371),
        np.float32(85.73625946044922),
        np.float32(0.5431923866271973),
    ),
}


def fp32_bits(value: np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", float(np.float32(value))))[0]


def quantize(values: np.ndarray, multiplier: np.float32) -> np.ndarray:
    scaled = np.multiply(values.astype(np.float32), multiplier, dtype=np.float32)
    return np.clip(np.rint(scaled), -128, 127).astype(np.int8)


def requantize(values: np.ndarray, multiplier: np.float32) -> np.ndarray:
    scaled = np.multiply(
        values.astype(np.int32).astype(np.float32), multiplier, dtype=np.float32
    )
    return np.clip(np.rint(scaled), -128, 127).astype(np.int8)


def layer_numeric(
    scales: Scales, bias: np.ndarray
) -> tuple[np.float32, np.float32, np.ndarray]:
    input_step = np.divide(np.float32(1.0), scales.input, dtype=np.float32)
    weight_step = np.divide(np.float32(1.0), scales.weight, dtype=np.float32)
    common_step = np.multiply(input_step, weight_step, dtype=np.float32)
    requant_multiplier = np.multiply(common_step, scales.output, dtype=np.float32)
    output_step = np.divide(np.float32(1.0), scales.output, dtype=np.float32)
    bias_q = np.rint(
        np.divide(bias.astype(np.float32), common_step, dtype=np.float32)
    ).astype(np.int32)
    return requant_multiplier, output_step, bias_q


def rescale(
    values: np.ndarray, previous_output: np.float32, next_input: np.float32
) -> np.ndarray:
    step = np.divide(np.float32(1.0), previous_output, dtype=np.float32)
    real = np.multiply(values.astype(np.float32), step, dtype=np.float32)
    return quantize(real, next_input)


def pad_weight_matrix(
    weight: np.ndarray, kind: str
) -> tuple[np.ndarray, int, int, int]:
    if kind == "conv":
        out_channels, in_channels, kh, kw = weight.shape
        k = in_channels * kh * kw
        n = out_channels
        matrix = weight.transpose(1, 2, 3, 0).reshape(k, n)
    else:
        n, k = weight.shape
        matrix = weight.transpose(1, 0)
    n_padded = ((n + TILE - 1) // TILE) * TILE
    padded = np.zeros((k, n_padded), dtype=np.float32)
    padded[:, :n] = matrix.astype(np.float32)
    return padded, k, n, n_padded


def im2col_nhwc(values: np.ndarray, kernel: int) -> np.ndarray:
    height, width, channels = values.shape
    output_h = height - kernel + 1
    output_w = width - kernel + 1
    cols = np.empty((output_h * output_w, channels * kernel * kernel), dtype=np.int8)
    row = 0
    for oh in range(output_h):
        for ow in range(output_w):
            cols[row] = (
                values[oh : oh + kernel, ow : ow + kernel, :]
                .transpose(2, 0, 1)
                .reshape(-1)
            )
            row += 1
    return cols


def maxpool2(values: np.ndarray) -> np.ndarray:
    height, width, channels = values.shape
    return values.reshape(height // 2, 2, width // 2, 2, channels).max(axis=(1, 3))


def run_layer(
    a: np.ndarray,
    weight_padded: np.ndarray,
    bias_q: np.ndarray,
    requant: np.float32,
    n: int,
) -> np.ndarray:
    accumulator = a.astype(np.int32) @ weight_padded[:, :n].astype(np.int32)
    accumulator += bias_q.reshape(1, n)
    return requantize(accumulator, requant)


def tensor_stats(values: np.ndarray) -> dict[str, object]:
    flat = values.reshape(-1).astype(np.int64)
    return {
        "count": int(flat.size),
        "sum": int(flat.sum()),
        "min": int(flat.min()),
        "max": int(flat.max()),
        "first": [int(value) for value in flat[:8]],
    }


def load_model(checkpoint: Path):
    sys.path.insert(0, str(checkpoint.parent))
    return torch.load(checkpoint, map_location="cpu", weights_only=False).eval()


def load_input(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    if image.size != (28, 28):
        raise ValueError(f"expected a 28x28 LeNet input, got {image.size}")
    values = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    return np.subtract(
        np.multiply(values, np.float32(2.0), dtype=np.float32),
        np.float32(1.0),
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    image = load_input(args.image)
    with torch.inference_mode():
        fp32_logits = (
            model(torch.from_numpy(image).reshape(1, 1, 28, 28))
            .numpy()
            .reshape(-1)
            .astype(np.float32)
        )

    layer_names = ("conv1", "conv2", "fc1", "fc2", "fc3")
    layer_kinds = ("conv", "conv", "linear", "linear", "linear")
    layers = []
    for name, kind in zip(layer_names, layer_kinds):
        module = getattr(model, name)
        padded, k, n, n_padded = pad_weight_matrix(module.weight.detach().numpy(), kind)
        scales = SCALES[name]
        requant, output_step, bias_q = layer_numeric(
            scales, module.bias.detach().numpy()
        )
        qweight = quantize(padded, scales.weight)
        layers.append(
            {
                "name": name,
                "weight": padded,
                "qweight": qweight,
                "bias_q": bias_q,
                "k": k,
                "n": n,
                "n_padded": n_padded,
                "scales": scales,
                "requant": requant,
                "output_step": output_step,
            }
        )

    stages: dict[str, object] = {}
    q = quantize(image.reshape(-1), SCALES["conv1"].input).reshape(28, 28, 1)
    stages["input_q"] = tensor_stats(q)
    cols = im2col_nhwc(q, 5)
    stages["conv1_im2col_q"] = tensor_stats(cols)
    conv1_acc = cols.astype(np.int32) @ layers[0]["qweight"][:, :6].astype(np.int32)
    conv1_acc += layers[0]["bias_q"].reshape(1, 6)
    stages["conv1_acc"] = tensor_stats(conv1_acc)
    q = requantize(conv1_acc, layers[0]["requant"]).reshape(24, 24, 6)
    stages["conv1_q"] = tensor_stats(q)
    q = maxpool2(np.maximum(q, np.int8(0)))
    stages["pool1_q"] = tensor_stats(q)

    q = rescale(q, SCALES["conv1"].output, SCALES["conv2"].input)
    stages["conv2_input_q"] = tensor_stats(q)
    cols = im2col_nhwc(q, 5)
    q = run_layer(
        cols, layers[1]["qweight"], layers[1]["bias_q"], layers[1]["requant"], 16
    ).reshape(8, 8, 16)
    stages["conv2_q"] = tensor_stats(q)
    q = maxpool2(np.maximum(q, np.int8(0)))
    stages["pool2_q"] = tensor_stats(q)

    q = rescale(q, SCALES["conv2"].output, SCALES["fc1"].input)
    stages["fc1_input_nhwc_q"] = tensor_stats(q)
    q = q.transpose(2, 0, 1).reshape(1, 256)
    stages["fc1_input_q"] = tensor_stats(q)
    q = run_layer(
        q, layers[2]["qweight"], layers[2]["bias_q"], layers[2]["requant"], 120
    )
    stages["fc1_q"] = tensor_stats(q)
    q = np.maximum(q, np.int8(0))

    q = rescale(q, SCALES["fc1"].output, SCALES["fc2"].input)
    stages["fc2_input_q"] = tensor_stats(q)
    q = run_layer(
        q, layers[3]["qweight"], layers[3]["bias_q"], layers[3]["requant"], 84
    )
    stages["fc2_q"] = tensor_stats(q)
    q = np.maximum(q, np.int8(0))

    q = rescale(q, SCALES["fc2"].output, SCALES["fc3"].input)
    stages["fc3_input_q"] = tensor_stats(q)
    q = run_layer(
        q, layers[4]["qweight"], layers[4]["bias_q"], layers[4]["requant"], 10
    ).reshape(-1)
    stages["fc3_q"] = tensor_stats(q)
    quant_logits = np.multiply(
        q.astype(np.float32), layers[4]["output_step"], dtype=np.float32
    )
    quant_class = int(np.argmax(quant_logits))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(
            struct.pack("<5I", MAGIC, VERSION, image.size, len(layers), quant_class)
        )
        stream.write(q.astype(np.int8).tobytes())
        stream.write(bytes(TILE - q.size))
        logits_padded = np.zeros(TILE, dtype=np.float32)
        logits_padded[: quant_logits.size] = quant_logits
        stream.write(logits_padded.astype("<f4").tobytes())
        stream.write(image.reshape(-1).astype("<f4").tobytes())
        for layer in layers:
            scales = layer["scales"]
            stream.write(
                struct.pack(
                    "<10I",
                    layer["k"],
                    layer["n"],
                    layer["n_padded"],
                    layer["weight"].size,
                    fp32_bits(scales.input),
                    fp32_bits(scales.weight),
                    fp32_bits(scales.output),
                    fp32_bits(layer["requant"]),
                    fp32_bits(layer["output_step"]),
                    layer["bias_q"].size,
                )
            )
            stream.write(layer["weight"].astype("<f4").tobytes())
            stream.write(layer["bias_q"].astype("<i4").tobytes())

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "image": str(args.image.resolve()),
        "fp32_class": int(np.argmax(fp32_logits)),
        "fp32_logits": [float(value) for value in fp32_logits],
        "quant_class": quant_class,
        "quant_q": [int(value) for value in q],
        "quant_logits": [float(value) for value in quant_logits],
        "payload_bytes": args.output.stat().st_size,
        "scale_mode": "per-tensor",
        "rounding": "RNE",
        "zero_point": 0,
        "stages": stages,
        "layers": {
            layer["name"]: {
                "weight_q": tensor_stats(layer["qweight"]),
                "bias_q": tensor_stats(layer["bias_q"]),
                "requant_multiplier_bits": fp32_bits(layer["requant"]),
            }
            for layer in layers
        },
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
