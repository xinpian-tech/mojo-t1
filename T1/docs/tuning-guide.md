# Evaluating T1 kernel performance

## Scope

The current integration supports one block with one to four harts. The
threaded configuration described in [the T1 README](../README.md) has
1024-bit per-thread VLEN and a VME shared across harts. Do not transfer
throughput, latency or VRF-port limits from the older blastoise setup
without measuring the actual configuration.

The T1 README records both a historical TinyLlama single-token RVV/VME
comparison and a Stories15M four-token prefill comparison. The former
compares FP32 RVV with group-32 quantized VME, not equal-precision kernels.
The latter compares equal-precision GEMV and GEMM. Neither is full-inference
acceptance, and results from different models/token counts are not ratios
of the same workload.

## Comparison procedure

1. Pin the Mojo, T1 software and simulator revisions, including RTL patches.
   Record the hardware configuration, hart count and simulator memory model.
2. Keep the model, layer, token position, input and launch geometry fixed.
   Run one layer for this comparison, not the complete model.
3. Check each implementation's complete memory readback against its own
   matching ISA reference before comparing performance. Quantized VME and
   FP32 RVV need not produce identical bytes to each other.
4. Keep `sim_result.json`, `mmio-event.jsonl`, `memdump.bin`, the kernel ELF
   and the reference readback. Record the commands and input provenance.
5. Compare completed runs without repeating simulation:

```sh
python3 T1/tools/compare-layer-rtl.py \
  --rvv /path/to/rvv-run --vme /path/to/vme-run \
  --rvv-reference /path/to/rvv-isa.bin --vme-reference /path/to/vme-isa.bin
```

The comparison tool checks successful runs, matching reported hardware
configuration, exact reference readbacks and one kernel timing interval.
It cannot establish that the model, input or simulator revision matches;
the caller must verify those separately.

## VME programming model and current hardware

The instruction definitions in T1's `custom-instructions/rv_zvt` pin
[Zvt v0.4 at commit 8a44aff](https://github.com/riscv/riscv-isa-manual/blob/8a44aff661f5a27b93cb805a6494ab1d6f1c6de8/src/unpriv/zvt.adoc).
Use that version when interpreting this implementation, rather than assuming
that all instructions in a newer draft are implemented.

- The operation is `C[M,N] += A[K,M].T * B[K,N]`. For INT8 to INT32,
  KMAX is four, and successive K slices are two vector registers apart.
- `msetmtype` resets `vl` and derives LMUL from the tile/vector dimensions.
  For VLEN=1024, TE=32, SEW=8 and TWIDEN=4, LMUL is one. `vset*` clears
  `mtype`; changing vector configuration requires restoring matrix state.
- `msettn` changes the active vector length without discarding `tm` or `tk`.
  The GEMM kernel uses it to load exactly M activation elements and N weight
  elements, without padding, overreading, or switching LMUL.
- INT32 tile state consists of four independently addressable tiles per hart.
  Sharing execution hardware does not combine separate harts' operands.

The following implementation constraints were checked in T1 commit
`643b1e1c`, with the existing runtime correctness patches:

| Implementation source | Consequence for software |
| --- | --- |
| `t1zaozi/src/T1Projection.scala` | Direct LSU-to-VME forwarding is disabled when threaded; single-thread forwarding tests are not a four-hart performance model. |
| `t1zaozi/src/T1ThreadScheduler.scala` | A hart has at most one tile access in flight; independent vector operations remain subject to register hazards and shared capacity. |
| `t1zaozi/src/vme/VmeCore.scala` | The configured 2-by-2 PE array scans 4-by-4 output elements. M=1 uses only a quarter of the arithmetic positions. |
| `t1zaozi/src/vme/VmeTileStorage.scala` | Accumulation uses protected SRAM read-modify-write operations. Bank writes take priority over reads; peak arithmetic width is not sustained throughput. |
| `t1zaozi/src/vme/VmeSharedMemory.scala` | Tile transfers and the vector LSU share an AXI outlet; concurrent execution does not create additional memory bandwidth. |

### Llama dataflow

Prefill processes up to four already-known consecutive prompt tokens per
layer launch. Each hart owns output columns across all token rows, so a
weight tile is reused across those rows. Activations are quantized separately
per token and per group of 32, then packed as `[K,token]`. Each output row uses
its own activation scale and the existing per-output weight scale and sum.
Quantization groups cannot be merged into one integer accumulation when their
scales differ.

Q/K/V and FFN projections use this GEMM path. RoPE uses each token's actual
position; attention reads only positions up to that token, even though the
whole batch's K/V projections have already been written. The classifier runs
only on the final token of a batch. Single-token decode retains GEMV: unknown
future tokens cannot be used to fill matrix rows.

For a full M=4, N=32, K=4 operation, useful input bytes are 16+128=144,
versus 4*(4+128)=528 for four independent GEMVs. This is an operand-reuse
calculation, not measured AXI traffic or a 3.67x end-to-end speedup; it excludes
quantization, scales, sums, accumulator traffic, output updates and tails.

`T1_PREFILL_BATCH=1` retains token-at-a-time host scheduling.
`T1_PREFILL_GEMM=0` retains batched scheduling but projects each token with
GEMV, providing an equal-precision, equal-token-count performance reference.
The optional final CLI argument `--first-layer-only` synchronizes and exits
after the first layer launch, before the classifier or any subsequent layer.

`T1_PREFILL_INTERLEAVE=1` enables an experimental double-buffered GEMM
epilogue: submit the next matrix tile, process the previous result with
v24–v27, then finish the submitted tile. It is disabled by default: the
5-by-140, K=64 quantized test improved from 53,385 to 53,198 kernel cycles
(0.35%), including on the RAW-repaired simulator, with identical readbacks,
but no full-layer benefit has yet been
established. Its trace recorded 13 cross-hart simultaneous matrix/vector
writeback cycles and none within one hart. That is not evidence of useful
same-hart execution overlap.

## Interpreting measurements

### Confirmed scalar-to-vector correctness failure

The four-hart synthetic single-token layer differs between the previously
corrected RTL simulator and the VME ISA model. This is a correctness failure,
not a performance result. Stopping before SwiGLU produces identical complete
memory dumps, including W1/W3 outputs. Replaying those captured operands in
the isolated `t1-swiglu` executable also passes; the failure depends on the
instruction sequence in the full layer.

The failing trace identifies a stale scalar floating-point operand:

- Hart 0 commits `fmv.w.x fa5, t1` at cycle 80728, writing `0x3fb8aa3b`
  (approximately 1.442695).
- After one intervening `vmv.v.v`, `vfmadd.vf v11, fa5, v10` at PC
  `0x80102150` commits at cycle 80730 and enters T1 at cycle 80862.
- The first element of `v11` is `0xbe9a06c3` and `v10` is `0x3f000000`.
  Its result should be approximately 0.065990, but the physical write at
  cycle 80887 is `0x41d8b11d` (27.086481). This matches using the previous
  `fa5=0xc2b0c0a5` (approximately -88.376259), not the newly written value.

The inspected Rocket implementation gates its EX/MEM/WB floating-point RAW
checks with `idDecodeOutput(d.fp)` in `rocketvzaozi/src/Rocket.scala`.
`Decoder.scala` sets that field for scalar FP instruction sets, while vector
FP consumers use `vectorReadFRs1`. This is the suspected missing hazard
coverage; the exact pipeline repair remains an RTL-owner task. No RTL is
modified by this software work. The independent
`scalar_vector_hazard.mojo` reproducer uses no VME instructions: the ISA
model returns 6.5 for every element, whereas RTL returns 0.5 for element
zero. It also contains an adjacent `fmv.w.x` / `vfmv.v.f` dependency, so its
failure alone does not distinguish that dependency from the later FMA
dependency; the full-layer trace above isolates the latter. Do not discard
the failing reference or treat a passing batched path as proof that this
hazard is fixed.

The hardware repair is tracked in
[T1 PR #211](https://github.com/xinpian-tech/T1/pull/211), commit `90641f92`.
Its clean four-hart RTL regression passes 30 producer/consumer/distance
combinations in 22,022 cycles, covering scalar moves, loads, FP add/divide,
vector-to-scalar returns and zero/one/two-instruction gaps. The old RTL fails
the same regression at the FMA dependency. With the existing independent
simulator fixes retained, the original failing single-token ELF also passes:
105,315 total cycles, with its complete 157,088-byte memory dump identical
to the ISA reference. No software workaround is required for that failure.

The final software package also passes `t1-prefill --tokens 1`, `2`, `3`
and `4` on that repaired simulator. Each invocation compares sequential and
batched residuals and the full KV cache; its final complete 157,088-byte
readback matches the corresponding ISA run. The integer GEMM and all three
quantized variants (GEMV, blocking GEMM, interleaved GEMM) likewise pass
their host assertions and complete ISA readback comparisons. The real-model
single-layer result and immutable package/simulator paths are in the T1
README. The validated hardware repair is `90641f92`; later PR commits need
their own validation before being substituted for that simulator.

### Measurement scope

- HAL `DeviceContext.execution_time` measures host nanoseconds, including
  simulator startup and synchronization. It is not an RTL cycle count.
- Use the kernel-entry/return MMIO interval for RTL kernel cycles. Report
  total simulation cycles separately. Pokedex counts retired instructions,
  which are not interchangeable with RTL cycles.
- Cross-hart vector/matrix overlap is expected with shared VME resources.
  An overlap count does not establish a speedup or full matrix utilization.
  Lack of same-hart overlap alone is not evidence of a scalar race.
- Account for operand loads, broadcasts, quantization, tile shape, result
  stores, fences, synchronization and shared-unit contention. More matrix
  throughput does not guarantee lower end-to-end kernel latency.
- Static instruction counts and bounded trace samples suggest hypotheses;
  they do not assign an exact fraction of total execution cycles to a cause.
- Change one factor at a time and repeat correctness checks. If a slowdown
  suggests an RTL bug, separate evidence of incorrect behavior or a stall
  from expected serialization, and reduce it to a reproducible case.

See [the runtime README](../runtime/README.md) for simulator and trace
settings. The old fixed-cycle estimates and blanket claims that scalar
overhead is hidden are not assumptions of the current analysis.
