# Out-of-tree T1 support

The migration targets upstream Modular commit
`da0db31d422e19ad5952d8d106b6da1d6899cf0a` (2026-10-23).
No T1 code is added to upstream `Mojo/`, `max/`, or runtime directories.

Mojo is an external, non-flake source input named `mojo-upstream`, fetched
from `modular/modular` and pinned by revision and NAR hash in `flake.lock`.
The compiler, standard library, HAL headers, MAX compatibility overlay and
release metadata all use this same input. Local copies of `Mojo/`, `SDK/`,
`max/` and the upstream Bazel workspace are not build inputs.

This integration can live in a standalone repository containing only
`flake.nix`, `flake.lock`, `nix/` and `T1/`; no Mojo fork or rebasing is
required. Existing upstream files in this checkout are left untouched.
To update Mojo, change the revision in `inputs.mojo-upstream.url`, run
`nix flake lock`, and validate `nix build .#mojo .#t1`. Update the dependency
cache hash if upstream's dependency set changes. The T1 compiler plugin
and HAL must be rebuilt and tested against that same revision.

## Package boundary

```sh
nix build .#mojo --out-link result-mojo
nix build .#t1 --out-link result-t1
```

`mojo` is the upstream compiler, standard library and tools, with no T1
registration, runtime replacement or T1 environment variables. `t1`
depends on that toolchain and adds:

- `libT1Compiler.so`: official compiler plugin entry points for target
  traits and object emission, loaded with `MODULAR_COMPILER_PLUGINS`.
- `libT1Driver.so`: official HAL C ABI, loaded with
  `MODULAR_DRIVER_PLUGINS` rather than AsyncRT symbol replacement.
- Mojo target/hooks and a MAX source overlay staged **inside the T1
  package**, using upstream's HAL DeviceContext compatibility mapping.
- The RV32 launch trampoline and a `mojo-t1` launcher.

The overlay does not edit installed upstream packages or the checkout.
The compiler plugin must be rebuilt with the same upstream revision as the
compiler: its C++ interface is not a stable cross-version ABI. The HAL
interface is also experimental and pinned to this revision.

Before the development-output refactor, both commands above passed with the corrected Bazel dependency-cache hash,
without overriding the compiler plugin with a cached store path. The rebuilt
T1 package passes the four-hart quantized VME RTL smoke test; embedded device
ELFs for inference, RVV inference, thread exchange, RMSNorm and both VME
tests match the packages used for the measurements below.

The compiler plugin now depends on `mojo-dev` (also exposed as `mojo.dev`),
which exports headers and link libraries from the upstream SDK build.
Only the two T1 C++ translation units are compiled in the plugin package;
upstream C++ dependencies are no longer rebuilt in a second Bazel workspace.
The HAL driver uses the same development output's HAL headers. Adding this
output requires a one-time SDK rebuild; T1-only changes do not rebuild Mojo.

The development-output build passes `nix build .#mojo-dev .#mojo .#t1`,
the Mojo execution check and the development-export regression test. T1's
install checks compile the examples with the new compiler plugin. These
packaging checks do not constitute a new RTL performance run.

## Run

```sh
export T1RT_EMULATOR=/path/to/pokedex
export T1RT_EMULATOR_KIND=pokedex
export T1RT_MEMDUMP=1
result-t1/bin/mojo-t1 run --target-accelerator=t1 T1/examples/saxpy.mojo
```

Use the launcher to ensure both plugins and the overlay are selected.
Do not set `MODULAR_MOJO_MAX_SHARED_LIBS` to the old `libT1RT.so`.
See [runtime/README.md](runtime/README.md) for emulator prerequisites,
standalone HAL tests, timing semantics and launch limitations.

The T1 target currently requests host-layout argument encoding because
upstream's generic pointer encoder requires equal host/device sizes.
The driver and trampoline implement the existing low-32-bit argument
convention explicitly; upstream plugins alone do not remove that RV32
ABI constraint. T1 still needs its own ELF linker backend, driver and
simulator integration.

## Migration verification

The upstream toolchain, compiler plugin and standalone HAL package have
built with Nix. The overlay package has also compiled SAXPY and TinyLlama
through `nix build`, reusing the unchanged compiler-plugin store output.
The compiled Mojo SAXPY passes through the HAL driver on Pokedex.
Before the threading changes, the single-thread stories15M baseline produced
token 9038 (`Once`) for the empty prompt. The threaded TinyLlama and the
thread-exchange test compile through Nix; the exchange test passes 64
barriers with one, two, three and four active harts on RTL.
Historical documents in `docs/` must not be treated as validated HAL examples.

The `thread-tilelang` T1 checkout provides a four-hart configuration with
1024-bit per-thread VLEN. The extension now implements independent launch
stacks, official thread-index and block-barrier hooks, and partitioned
transformer matvec, attention and elementwise work. It supports one block
of one to four harts; larger grids and dynamic shared-memory allocations
are rejected. Set `T1RT_THREAD_COUNT` to the actual emulator hart count,
not merely the desired launch width. The launch count cannot exceed it.
The model runner passes the 1–4 hart exchange tests. After fixing attention
helpers to strip-mine at VLEN 1024, both single- and four-hart RVV stories15M
inference produce `Once upon a time`, matching the C reference. The complete
TinyLlama 1.1B model is undergoing comparison. Full-model RTL inference
remains a required acceptance gate and has not yet passed.
Pokedex's `--thread-count` currently applies to RTL differential checking,
not its standalone `run` command, so that command alone is insufficient
to validate multithreaded inference.

## Software VME path

`t1llama` now defaults to `T1_VME=1`: its projection, feed-forward and
classifier matvecs use Zvt `vtmmu` through `mojo/vme.mojo`, rather than
merely running RVV on hardware that contains a VME. T1 commit `643b1e1c`
supports unsigned 8-bit activations, signed 8-bit weights and 32-bit
integer accumulation. Host weight loading packs group-32 symmetric
quantized weights and per-output scales/sums. The device quantizes each
activation group, corrects the unsigned zero point and rescales to FP32.
Attention and other non-matvec operations still use RVV/FP32.

The four-hart `t1-vme` integer test passes on RTL, including signed weights
and actual `vtmmu` issue events on all four harts. `t1-vme-quantized` also
passes on RTL, checking group scales, zero-point correction and partial tiles.
The matvec now double-buffers integer result blocks: after submitting the
next VME block, independent RVV registers v24–v27 correct and rescale the
previous block before waiting for the new result. Matrix configuration is
restored before tile stores because changing `vtype` clears that configuration.
The quantized RTL test observes simultaneous matrix-MAC and vector-arithmetic
writeback; inspect a retained trace with:

```sh
python3 T1/tools/verify-vme-overlap.py /path/to/rtl-event.jsonl
```

This is evidence of writeback overlap, not a measurement of execution-window
overlap or a full-model throughput claim. VME is shared between harts: the
intended optimization overlaps one hart's matrix work with another hart's
vector work, rather than requiring same-hart overlap.
The overlap report separates per-hart overlap from simultaneous writes by
different harts, including matrix-to-vector hart pairs (`matrix->vector`).
Pair counts can overlap; the cross-hart cycle count counts each cycle only
once. The checker requires cross-hart overlap and matrix activity on all
four harts, but does not establish arbitration fairness from a bounded trace.
A bounded replay of the full TinyLlama 1.1B RTL image on
the corrected hardware observed VME execution on all four harts and 28
matrix/vector simultaneous-writeback cycles, all across different harts.
No same-hart writeback overlap was observed in that sample; this is not a
failure criterion for the shared-VME optimization. Its benefit still needs
end-to-end cycle measurements, together with shared-unit utilization and
arbitration-wait measurements.

`tools/capture-rtl-tail.py --wall-timeout 30` can bound a diagnostic replay
without interrupting the original inference process. A wall-time limit
returns status 124 and records `wall_timeout: true` in `capture.json`; this
is an intentionally incomplete trace, not an inference pass or RTL failure.
The current activation quantizer remains scalar, and thread-specialized
FFN/SwiGLU producer/consumer scheduling is not yet implemented.
Full-model VME numerical and RTL acceptance is still in progress.
The package includes `t1llama-rvv`, built with `-D T1_VME=0`, as the
four-hart FP32 RVV baseline. Compare its reported hardware cycles with
`t1llama` using the same model, prompt, position count and RTL simulator;
host simulation time is not a hardware-performance measurement. VME uses
quantized matvecs while this baseline uses FP32, so their numerical results
must be checked against their respective references rather than assumed
bit-identical.

For numerical comparison, `nix/t1-quantized-reference.nix` accepts a
`referenceSource` pointing to a llama2.c checkout. It builds its CPU runner
with `tools/quantized-matmul.c`, an independent signed-dot-product reference
using the same group-32 quantization (without VME's unsigned offset).
The four-hart VME ISA-model run and this reference both generate
`Once upon a time` for the four-position stories15M regression.
The full TinyLlama 1.1B four-hart VME ISA-model run also matches the
quantized reference's transcript over 12 positions (ignoring the runner's
final output newline). This remains separate from RTL acceptance.

## RTL acceptance

### First-layer VME performance evaluation

The TinyLlama 1.1B first transformer layer at position 0 completes on the
corrected four-hart RTL (VLEN 1024, DLEN 256, TE 32), using the same input
and simulator for both paths:

| Path | Kernel cycles | Total simulator cycles |
| --- | ---: | ---: |
| FP32 RVV | 7,832,651 | 7,832,964 |
| Group-32 quantized VME | 24,912,793 | 24,913,106 |

Both complete memory readbacks match their respective four-hart ISA-model
reference runs byte for byte, including all 2048 layer-output floats.
The current VME path takes **3.181 times as many kernel cycles** (0.314x
speedup), so the observed cross-hart overlap does not yield a net speedup.
These are kernel-entry-to-return MMIO cycle measurements, not host runtime;
they include device activation quantization, computation and synchronization,
but exclude host weight packing and host-to-device staging before launch.
This compares quantized VME with FP32 RVV, not equal-precision implementations.

The VME submission loop alone accounts for 1,720,320 instructions per hart
by static loop counting: 10,752 group-32 output tiles, eight K=4 iterations
per tile, and 20 instructions per iteration. This identifies substantial
operand-loading, broadcast and loop overhead, but does not attribute an
exact fraction of the measured cycles to it. No subsequent layer is run
for this performance evaluation; it is not full-inference acceptance.

Recheck completed, preserved run directories and their ISA-model memory
readbacks without rerunning simulation:

```sh
python3 T1/tools/compare-layer-rtl.py \
  --rvv /path/to/rvv-run --vme /path/to/vme-run \
  --rvv-reference /path/to/rvv-isa.bin --vme-reference /path/to/vme-isa.bin
```

The tool rejects unsuccessful runs, mismatched readbacks, different reported
RTL configurations and missing or ambiguous kernel intervals. The caller
must still ensure both images represent the same model, layer, position and
input, and that the simulator builds are identical.

Build the `thread-tilelang` checkout's four-Rocket configuration with Nix:

```sh
git -C "$T1_SOURCE" apply --unidiff-zero --check "$PWD/T1/runtime/threaded-rtl-memdump.patch"
git -C "$T1_SOURCE" apply --unidiff-zero "$PWD/T1/runtime/threaded-rtl-memdump.patch"
git -C "$T1_SOURCE" apply --unidiff-zero --check "$PWD/T1/runtime/threaded-rtl-correctness.patch"
git -C "$T1_SOURCE" apply --unidiff-zero "$PWD/T1/runtime/threaded-rtl-correctness.patch"
nix build "$T1_SOURCE#t1.vme32t-threaded.t1rocketemu.verilator-emu" \
  --cores 4 --max-jobs 1 --out-link result-t1-rtl
```

The simulator must implement `+t1_memory_dump_path` and
`+t1_memory_dump_range`; the current integration adds this to the T1
checkout's `difftest/dpi_t1rocketemu` finalization path. This is separate
from the upstream Mojo source, which remains unmodified.
The patch targets T1 commit `643b1e1c`; do not reapply it to a checkout
that already contains these changes.

`runtime/threaded-rtl-correctness.patch` contains hardware fixes for SRT
integer/float mode transitions, Rocket scalar-return arbitration, and the
Lane Stage0/load-return arbiter. The latter must reselect live requests:
load returns can move between slots under backpressure, so retaining only
the selected producer while OR-ing both producers' valid signals can admit
invalid load payloads with a newer instruction's destination and tag.
The directed tests in `tests/rtl/srt_mode.sv` and
`tests/rtl/scalar_contention.sv` reproduce failures in the baseline generated
RTL and pass with the corresponding generated-RTL substitutions, built by
`nix/t1-srt-regression.nix` and `nix/t1-scalar-regression.nix`.
The scalar test injects a MulDiv response to exercise simultaneous returns;
it is not an instruction-stream or full-system acceptance test.
Both directed tests also pass against RTL regenerated from the patched
Scala sources (`applyCandidate = false; expectFixed = true`). The Lane
arbiter substitution alone fixes the four-hart LMUL8 RMSNorm stall; numerical
checks pass with one through four harts. `verify-threaded.py` includes this
2048-element regression before inference. With all three fixes regenerated
and built together, RTL passes the 1–4 hart exchange/RMSNorm tests and both
VME matrix tests, including quantization and partial tiles. Inference
validation and full TinyLlama RTL acceptance remain pending.

The acceptance script runs exchange tests with 1, 2, 3 and 4 active harts,
then compares single- and four-hart model tokens and the C reference's
generated text. Use a converted TinyLlama 1.1B model for final acceptance;
stories15M is only the faster regression model. For the VME path, pass
`--reference-kind quantized` and the independently built quantized C runner
as `--reference`; this additionally runs both VME matrix tests. For the
original FP32 baseline, use a package compiled with `T1_VME=0` and the
unmodified C reference. Neither comparison alone quantifies loss relative
to FP32; model quality evaluation remains separate.

```sh
python3 T1/tools/verify-threaded.py \
  --package result-t1 \
  --emulator result-t1-rtl/bin/t1-vme32t-threaded-t1rocketemu-verilated-simulator \
  --kind rtl --model /path/to/tinyllama1.1b.bin \
  --tokenizer /path/to/tokenizer.bin --reference /path/to/llama2-c-run \
  --prompt 'The capital of France is' --steps 12 \
  --output /path/to/new-validation-directory
```

This writes separate stdout/stderr logs and creates `report.json` only
after all comparisons pass. A Pokedex report is not an RTL result.
