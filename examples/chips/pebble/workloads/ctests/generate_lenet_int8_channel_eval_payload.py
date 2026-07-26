#!/usr/bin/env python3
"""Generate a per-channel LeNet accuracy payload for Pebble BEMU.

Scales are taken from a quant-eval calibration manifest.  Convolution
activations use logical-channel scales, Linear activations use one tensor
scale, and weights/outputs use output-channel scales.  The golden model
follows the hardware accumulator contract: raw products are reduced per
input/weight scale pair, each partial is RNE-aligned to the largest product
step for that output channel, and the result is requantized with an
output-channel multiplier.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from generate_lenet_int8_eval_payload import load_mnist
from generate_lenet_int8_payload import (
    MAGIC,
    TILE,
    im2col_nhwc,
    load_model,
    maxpool2,
    pad_weight_matrix,
    tensor_stats,
)


VERSION = 4
LAYER_NAMES = ("conv1", "conv2", "fc1", "fc2", "fc3")
LAYER_KINDS = ("conv", "conv", "linear", "linear", "linear")


def fp32_bits(values: np.ndarray) -> list[int]:
    return np.asarray(values, dtype="<f4").view("<u4").astype(np.uint32).tolist()


def scale_values(module: dict[str, object], role: str) -> tuple[np.ndarray, np.ndarray]:
    entry = module[role]
    multiplier = np.asarray(entry["quant_multiplier"], dtype=np.float32)
    encoded = [int(value) for value in entry["quant_multiplier_bits"]]
    if fp32_bits(multiplier) != encoded:
        raise ValueError(f"{role} multiplier bits do not match the manifest")
    step = np.divide(np.float32(1.0), multiplier, dtype=np.float32)
    return multiplier, step


def quantize_channel(
    values: np.ndarray, multiplier: np.ndarray, *, axis: int = -1
) -> np.ndarray:
    shape = [1] * values.ndim
    shape[axis] = multiplier.size
    scaled = np.multiply(
        values.astype(np.float32), multiplier.reshape(shape), dtype=np.float32
    )
    return np.clip(np.rint(scaled), -128, 127).astype(np.int8)


def prepare_layers(model, manifest: dict[str, object]) -> list[dict[str, object]]:
    if manifest.get("mode") != "channel":
        raise ValueError("the calibration manifest is not per-channel")
    layers: list[dict[str, object]] = []
    for name, kind in zip(LAYER_NAMES, LAYER_KINDS):
        module = getattr(model, name)
        calibrated = manifest["modules"][name]
        weight, k, n, n_padded = pad_weight_matrix(module.weight.detach().numpy(), kind)
        input_multiplier, input_step = scale_values(calibrated, "input")
        weight_multiplier, weight_step = scale_values(calibrated, "weight")
        output_multiplier, output_step = scale_values(calibrated, "output")
        expected_input_groups = module.in_channels if kind == "conv" else 1
        if input_multiplier.size != expected_input_groups:
            raise ValueError(f"{name}: unexpected input channel count")
        if weight_multiplier.size != n or output_multiplier.size != n:
            raise ValueError(f"{name}: unexpected output channel count")

        qweight = np.zeros((k, n_padded), dtype=np.int8)
        qweight[:, :n] = quantize_channel(weight[:, :n], weight_multiplier, axis=1)
        product_step = np.multiply(
            input_step[:, None], weight_step[None, :], dtype=np.float32
        )
        common_step = np.max(product_step, axis=0).astype(np.float32)
        alignment = np.divide(product_step, common_step[None, :], dtype=np.float32)
        requant = np.multiply(common_step, output_multiplier, dtype=np.float32)
        bias_q = np.rint(
            np.divide(
                module.bias.detach().numpy().astype(np.float32),
                common_step,
                dtype=np.float32,
            )
        ).astype(np.int32)
        layers.append(
            {
                "name": name,
                "kind": kind,
                "k": k,
                "n": n,
                "n_padded": n_padded,
                "weight": weight,
                "qweight": qweight,
                "input_multiplier": input_multiplier,
                "weight_multiplier": weight_multiplier,
                "output_multiplier": output_multiplier,
                "input_step": input_step,
                "output_step": output_step,
                "alignment": alignment,
                "requant": requant,
                "bias_q": bias_q,
            }
        )
    return layers


def align_partial(partial: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    if np.any(partial < np.iinfo(np.int32).min) or np.any(
        partial > np.iinfo(np.int32).max
    ):
        raise OverflowError("per-channel partial exceeded signed INT32")
    scaled = np.multiply(
        partial.astype(np.int32).astype(np.float32), ratio, dtype=np.float32
    )
    return np.rint(scaled).astype(np.int64)


def requantize(accumulator: np.ndarray, layer: dict[str, object]) -> np.ndarray:
    accumulator = accumulator + layer["bias_q"].reshape(1, -1)
    safe = np.clip(accumulator, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(
        np.int32
    )
    scaled = np.multiply(
        safe.astype(np.float32), layer["requant"].reshape(1, -1), dtype=np.float32
    )
    return np.clip(np.rint(scaled), -128, 127).astype(np.int8)


def run_conv(columns: np.ndarray, layer: dict[str, object]) -> np.ndarray:
    input_groups = layer["input_multiplier"].size
    kernel_area = layer["k"] // input_groups
    accumulator = np.zeros((columns.shape[0], layer["n"]), dtype=np.int64)
    for channel in range(input_groups):
        begin = channel * kernel_area
        end = begin + kernel_area
        partial = columns[:, begin:end].astype(np.int64) @ layer["qweight"][
            begin:end, : layer["n"]
        ].astype(np.int64)
        accumulator += align_partial(
            partial, layer["alignment"][channel].reshape(1, -1)
        )
    return requantize(accumulator, layer)


def run_linear(values: np.ndarray, layer: dict[str, object]) -> np.ndarray:
    partial = values.astype(np.int64) @ layer["qweight"][
        : layer["k"], : layer["n"]
    ].astype(np.int64)
    aligned = align_partial(partial, layer["alignment"])
    return requantize(aligned, layer)


def rescale_channel(
    values: np.ndarray,
    previous_step: np.ndarray,
    next_multiplier: np.ndarray,
    *,
    axis: int = -1,
) -> np.ndarray:
    shape = [1] * values.ndim
    shape[axis] = previous_step.size
    real = np.multiply(
        values.astype(np.float32), previous_step.reshape(shape), dtype=np.float32
    )
    return quantize_channel(real, next_multiplier, axis=axis)


def run_quant(image: np.ndarray, layers: list[dict[str, object]]) -> np.ndarray:
    q = quantize_channel(
        image.reshape(28, 28, 1), layers[0]["input_multiplier"], axis=2
    )
    q = run_conv(im2col_nhwc(q, 5), layers[0]).reshape(24, 24, 6)
    q = maxpool2(np.maximum(q, np.int8(0)))
    q = rescale_channel(
        q, layers[0]["output_step"], layers[1]["input_multiplier"], axis=2
    )

    q = run_conv(im2col_nhwc(q, 5), layers[1]).reshape(8, 8, 16)
    q = maxpool2(np.maximum(q, np.int8(0)))
    real = np.multiply(
        q.astype(np.float32),
        layers[1]["output_step"].reshape(1, 1, -1),
        dtype=np.float32,
    )
    real = real.transpose(2, 0, 1).reshape(1, 256)
    q = quantize_channel(real, layers[2]["input_multiplier"], axis=1)

    q = run_linear(q, layers[2])
    q = rescale_channel(
        np.maximum(q, np.int8(0)),
        layers[2]["output_step"],
        layers[3]["input_multiplier"],
        axis=1,
    )
    q = run_linear(q, layers[3])
    q = rescale_channel(
        np.maximum(q, np.int8(0)),
        layers[3]["output_step"],
        layers[4]["input_multiplier"],
        axis=1,
    )
    return run_linear(q, layers[4]).reshape(-1)


def write_layer(stream, layer: dict[str, object]) -> None:
    alignment = np.asarray(layer["alignment"], dtype=np.float32)
    stream.write(
        struct.pack(
            "<9I",
            layer["k"],
            layer["n"],
            layer["n_padded"],
            layer["weight"].size,
            layer["input_multiplier"].size,
            layer["weight_multiplier"].size,
            layer["output_multiplier"].size,
            alignment.size,
            layer["bias_q"].size,
        )
    )
    for values, dtype in (
        (layer["weight"], "<f4"),
        (layer["input_multiplier"], "<f4"),
        (layer["weight_multiplier"], "<f4"),
        (layer["output_multiplier"], "<f4"),
        (layer["input_step"], "<f4"),
        (layer["output_step"], "<f4"),
        (alignment, "<f4"),
        (layer["requant"], "<f4"),
        (layer["bias_q"], "<i4"),
    ):
        stream.write(np.asarray(values, dtype=dtype).tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mnist-root", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.offset < 0:
        parser.error("--offset must be non-negative")

    model = load_model(args.checkpoint)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    images, labels = load_mnist(args.mnist_root, args.offset, args.limit)
    layers = prepare_layers(model, manifest)

    fp32_classes = []
    with torch.inference_mode():
        for start in range(0, len(images), args.batch_size):
            batch = torch.from_numpy(images[start : start + args.batch_size, None])
            fp32_classes.append(model(batch).argmax(dim=1).numpy().astype(np.uint8))
    fp32_classes_array = np.concatenate(fp32_classes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quant_classes = np.empty(len(images), dtype=np.uint8)
    with args.output.open("wb") as stream:
        stream.write(struct.pack("<5I", MAGIC, VERSION, 784, len(layers), len(images)))
        for layer in layers:
            write_layer(stream, layer)
        for index, (image, label, fp32_class) in enumerate(
            zip(images, labels, fp32_classes_array)
        ):
            q = run_quant(image, layers)
            logits = np.multiply(q.astype(np.float32), layers[-1]["output_step"])
            quant_class = int(np.argmax(logits))
            quant_classes[index] = quant_class
            q_padded = np.zeros(TILE, dtype=np.int8)
            q_padded[: q.size] = q
            stream.write(struct.pack("<3I", int(label), int(fp32_class), quant_class))
            stream.write(q_padded.tobytes())
            stream.write(image.reshape(-1).astype("<f4").tobytes())
            if (index + 1) % 1000 == 0:
                print(f"generated {index + 1}/{len(images)} samples", file=sys.stderr)

    fp32_correct = int(np.count_nonzero(fp32_classes_array == labels))
    quant_correct = int(np.count_nonzero(quant_classes == labels))
    agreement = int(np.count_nonzero(quant_classes == fp32_classes_array))
    count = len(images)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "offset": args.offset,
        "samples": count,
        "fp32_correct": fp32_correct,
        "fp32_top1": fp32_correct / count,
        "python_int8_correct": quant_correct,
        "python_int8_top1": quant_correct / count,
        "top1_loss_percentage_points": (fp32_correct - quant_correct) * 100.0 / count,
        "prediction_agreement": agreement / count,
        "payload_bytes": args.output.stat().st_size,
        "scale_mode": "per-channel",
        "activation_scale": "conv-per-channel-linear-per-tensor",
        "weight_scale": "per-output-channel",
        "accumulator": "max-product-step-rne-alignment",
        "rounding": "RNE",
        "zero_point": 0,
        "layers": {
            layer["name"]: {
                "input_channels": int(layer["input_multiplier"].size),
                "weight_channels": int(layer["weight_multiplier"].size),
                "output_channels": int(layer["output_multiplier"].size),
                "weight_q": tensor_stats(layer["qweight"]),
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
