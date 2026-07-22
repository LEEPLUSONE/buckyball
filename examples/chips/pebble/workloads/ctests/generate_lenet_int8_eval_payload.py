#!/usr/bin/env python3
"""Generate a batched MNIST payload for LeNet accuracy evaluation on Pebble BEMU."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from generate_lenet_int8_payload import (
    MAGIC,
    SCALES,
    TILE,
    fp32_bits,
    im2col_nhwc,
    layer_numeric,
    load_model,
    maxpool2,
    pad_weight_matrix,
    quantize,
    rescale,
    run_layer,
    tensor_stats,
)


VERSION = 2
LAYER_NAMES = ("conv1", "conv2", "fc1", "fc2", "fc3")
LAYER_KINDS = ("conv", "conv", "linear", "linear", "linear")


def find_raw_files(root: Path) -> tuple[Path, Path]:
    for directory in (root, root / "raw", root / "MNIST" / "raw"):
        images = directory / "t10k-images-idx3-ubyte"
        labels = directory / "t10k-labels-idx1-ubyte"
        if images.is_file() and labels.is_file():
            return images, labels
    raise FileNotFoundError(f"cannot find raw MNIST test files below {root}")


def load_mnist(
    root: Path, offset: int, limit: int | None
) -> tuple[np.ndarray, np.ndarray]:
    image_path, label_path = find_raw_files(root)
    image_data = image_path.read_bytes()
    label_data = label_path.read_bytes()
    magic, image_count, rows, columns = struct.unpack_from(">4I", image_data)
    label_magic, label_count = struct.unpack_from(">2I", label_data)
    if magic != 2051 or label_magic != 2049 or rows != 28 or columns != 28:
        raise ValueError("invalid MNIST IDX headers")
    if image_count != label_count:
        raise ValueError("MNIST image/label count mismatch")
    if offset >= image_count:
        raise ValueError(f"offset {offset} exceeds MNIST size {image_count}")
    remaining = image_count - offset
    count = remaining if limit is None else min(limit, remaining)
    pixels = np.frombuffer(
        image_data, dtype=np.uint8, offset=16 + offset * 784, count=count * 784
    )
    labels = np.frombuffer(
        label_data, dtype=np.uint8, offset=8 + offset, count=count
    ).copy()
    images = pixels.reshape(count, 28, 28).astype(np.float32)
    images = images / np.float32(255.0) * np.float32(2.0) - np.float32(1.0)
    return images, labels


def prepare_layers(model) -> list[dict[str, object]]:
    layers: list[dict[str, object]] = []
    for name, kind in zip(LAYER_NAMES, LAYER_KINDS):
        module = getattr(model, name)
        weight, k, n, n_padded = pad_weight_matrix(module.weight.detach().numpy(), kind)
        scales = SCALES[name]
        requant, output_step, bias_q = layer_numeric(
            scales, module.bias.detach().numpy()
        )
        layers.append(
            {
                "name": name,
                "weight": weight,
                "qweight": quantize(weight, scales.weight),
                "bias_q": bias_q,
                "k": k,
                "n": n,
                "n_padded": n_padded,
                "scales": scales,
                "requant": requant,
                "output_step": output_step,
            }
        )
    return layers


def run_quant(image: np.ndarray, layers: list[dict[str, object]]) -> np.ndarray:
    q = quantize(image.reshape(-1), SCALES["conv1"].input).reshape(28, 28, 1)
    q = run_layer(
        im2col_nhwc(q, 5),
        layers[0]["qweight"],
        layers[0]["bias_q"],
        layers[0]["requant"],
        6,
    ).reshape(24, 24, 6)
    q = maxpool2(np.maximum(q, np.int8(0)))
    q = rescale(q, SCALES["conv1"].output, SCALES["conv2"].input)
    q = run_layer(
        im2col_nhwc(q, 5),
        layers[1]["qweight"],
        layers[1]["bias_q"],
        layers[1]["requant"],
        16,
    ).reshape(8, 8, 16)
    q = maxpool2(np.maximum(q, np.int8(0)))
    q = rescale(q, SCALES["conv2"].output, SCALES["fc1"].input)
    q = q.transpose(2, 0, 1).reshape(1, 256)
    q = run_layer(
        q,
        layers[2]["qweight"],
        layers[2]["bias_q"],
        layers[2]["requant"],
        120,
    )
    q = rescale(np.maximum(q, np.int8(0)), SCALES["fc1"].output, SCALES["fc2"].input)
    q = run_layer(
        q,
        layers[3]["qweight"],
        layers[3]["bias_q"],
        layers[3]["requant"],
        84,
    )
    q = rescale(np.maximum(q, np.int8(0)), SCALES["fc2"].output, SCALES["fc3"].input)
    return run_layer(
        q,
        layers[4]["qweight"],
        layers[4]["bias_q"],
        layers[4]["requant"],
        10,
    ).reshape(-1)


def write_layer(stream, layer: dict[str, object]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    images, labels = load_mnist(args.mnist_root, args.offset, args.limit)
    layers = prepare_layers(model)

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
            quant_class = int(np.argmax(q))
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
        "mnist_root": str(args.mnist_root.resolve()),
        "offset": args.offset,
        "samples": count,
        "fp32_correct": fp32_correct,
        "fp32_top1": fp32_correct / count,
        "python_int8_correct": quant_correct,
        "python_int8_top1": quant_correct / count,
        "top1_loss_percentage_points": (fp32_correct - quant_correct) * 100.0 / count,
        "prediction_agreement": agreement / count,
        "payload_bytes": args.output.stat().st_size,
        "scale_mode": "per-tensor",
        "rounding": "RNE",
        "zero_point": 0,
        "layers": {
            layer["name"]: {
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
