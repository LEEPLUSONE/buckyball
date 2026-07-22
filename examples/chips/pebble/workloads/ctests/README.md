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
