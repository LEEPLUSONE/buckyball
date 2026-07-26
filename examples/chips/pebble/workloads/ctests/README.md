# Full-INT8 LeNet on Pebble BEMU

This workload runs all five LeNet compute layers through Pebble BEMU:

- `FP2INT` quantizes the input and every layer's weights at runtime.
- `MATRIX` executes tiled INT8 convolution and linear MACs.
- `INT2FP` in INT8-output mode requantizes every INT32 layer result.
- `INT2FP` followed by `FP2INT` converts activations between independently
  calibrated layer boundaries.

ReLU, max-pooling, im2col, tile packing, and bias addition are scalar control
code. The current Pebble compiler does not yet lower a complete imported LeNet
graph, so the C workload is also an executable reference for that missing
lowering.

Generate the ignored runtime payload from the repository checkpoint and image:

```bash
python3 generate_lenet_int8_payload.py \
  --checkpoint /path/to/LeNet/lenet-model.pth \
  --image /path/to/LeNet/images/8.bmp \
  --output lenet_int8_payload.bin \
  --summary lenet_int8_payload.json
```

From this directory, compile the RISC-V Linux ELF after activating the
Buckyball development environment:

```bash
repo=$(git rev-parse --show-toplevel)
"$RISCV/bin/riscv64-unknown-linux-gnu-gcc" \
  -O2 -g -static -march=rv64gc -fno-builtin-printf \
  -I"$repo/bb-tests/workloads/lib" \
  -I"$repo/examples/chips/toy/workloads/common" \
  lenet_int8_bemu.c \
  "$repo/examples/chips/toy/workloads/common/buckyball.c" \
  -o lenet_int8_bemu-linux
```

Run from this directory because the current BEMU CLI does not forward guest
arguments and the program therefore opens `lenet_int8_payload.bin` by default:

```bash
spike_lib=$(dirname "$(find "$repo/bebop/target/debug/build" \
  -path '*/spike_install/lib/libriscv.so' -print -quit)")
LD_LIBRARY_PATH="$spike_lib" \
  "$repo/bebop/target/debug/bebop" run bemu --pk \
  --elf ./lenet_int8_bemu-linux --log-dir /tmp/lenet-int8-bemu
```

The requant DMA staging area is deliberately a 64-byte-aligned static buffer.
Large heap-backed staging buffers can cross guest pages that are not represented
by BEMU's current fast linear address mapping.

The program succeeds only if the final INT8 logits, dequantized FP32 logits,
and classification match the Python golden embedded in the payload.

## Full MNIST accuracy evaluation

Generate either the full test split or a non-overlapping shard. Each payload
stores the shared weights once and the Python FP32 class plus exact INT8 logits
for every sample:

```bash
python3 generate_lenet_int8_eval_payload.py \
  --checkpoint /path/to/LeNet/lenet-model.pth \
  --mnist-root /path/to/LeNet/data/MNIST \
  --offset 0 --limit 1000 \
  --output lenet_int8_eval_payload.bin \
  --summary lenet_int8_eval_payload.json
```

Build `lenet_int8_bemu_eval.c` with the same compiler flags and include paths
shown above, then run `lenet_int8_bemu_eval-linux`. The guest quantizes weights
only once, performs scalar MATRIX/requant checks on the first sample, and
reports FP32 Top-1, Pebble INT8 Top-1, accuracy loss, prediction agreement, and
exact Python/BEMU INT8-logit agreement over the payload.

Top-1 tie-breaking is deterministic: the lowest class index wins, matching
`argmax` and the guest's strict-greater comparison.

## Per-channel activation and weight evaluation

`lenet_int8_channel_bemu_eval.c` implements the quant-eval per-channel
hardware contract in Pebble BEMU. Convolution inputs use one scale per logical
NCHW channel, Linear inputs use one scale for the whole activation tensor,
weights use one scale per output channel, and layer outputs/requantization use
one scale per output channel.

The guest loads up to 256 FP32 values into a 1 KiB MMIO scale table and binds
that table to the conversion instruction's source bank. The physical mapping
is explicit:

- convolution activation row: one spatial position, lanes 0..15 are channel
  block `c0..c0+15`;
- weight row: one reduction index, lanes 0..15 are output-channel block
  `oc0..oc0+15`;
- requant row: one output position, lanes 0..15 are output-channel block
  `oc0..oc0+15`.

Each per-channel block passes `table_offset = c0 * sizeof(float)` (or `oc0 *
sizeof(float)`) to the conversion instruction. Linear inputs instead use the
scalar FP2INT instruction and do not require a lane-to-channel lookup. Linear
weights and outputs still exercise non-zero table offsets because their output
channel counts exceed 16. For convolution, the guest aligns partial sums from
different input/output scale pairs through Pebble `INT2FP` plus `FP2INT` with
binary32 RNE before accumulation and final output-channel requantization.

Generate a payload from the calibration manifest:

```bash
python3 generate_lenet_int8_channel_eval_payload.py \
  --checkpoint /path/to/LeNet/lenet-model.pth \
  --manifest /path/to/channel/calibration_manifest.json \
  --mnist-root /path/to/LeNet/data/MNIST \
  --offset 0 --limit 10 \
  --output lenet_int8_channel_eval_payload.bin \
  --summary lenet_int8_channel_eval_payload.json
```

Build and run `lenet_int8_channel_bemu_eval.c` with the same compiler/BEMU
commands used above. The program fails on the first sample whose final INT8
logits differ from the Python hardware-contract golden. Classification is
performed after per-channel dequantization; comparing raw INT8 logits would be
incorrect because the ten output channels have different steps.
