# Evaluating T1 kernel performance

## Scope

The current integration supports one block with one to four harts. The
threaded configuration described in [the T1 README](../README.md) has
1024-bit per-thread VLEN and a VME shared across harts. Do not transfer
throughput, latency or VRF-port limits from the older blastoise setup
without measuring the actual configuration.

The T1 README records a historical TinyLlama single-token RVV/VME
comparison and Stories15M four-token prefill comparisons. The original VME
GEMM improves over equal-precision VME GEMV by 1.762x, but takes 45.2% more
cycles than the existing FP32 RVV path before shared quantization. Shared RVV
packing and Q/K/V plus gate/up reuse reduce VME to 1,191,922 cycles: 37.3%
below the original VME GEMM and 9.0% below the existing RVV baseline.
RVV versus VME compares different
precisions; GEMV versus GEMM uses the same group-32 quantization. None is
full-inference acceptance, and results from different models/token counts
are not ratios of the same workload.

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

The production prefill path cooperatively quantizes each input group once:
harts partition `(group, token)` tasks, use RVV max reduction/division/rounding
and strided byte stores, then synchronize before consuming the shared pack.
Q/K/V reuse one pack and gate/up reuse another. Attention output and the
post-SwiGLU input each receive their own pack. The four passes process 6,528
elements for four-token Stories15M, versus 39,936 in the original per-hart,
per-projection scalar packer. These are element counts, not cycle ratios.
The appended workspace occupies `max(dim, hidden) * 9 // 8` FP32 words:
four byte-packed rows plus four FP32 scales per 32-element group. Division,
ties-away-from-zero rounding, RTZ conversion and clipping remain unchanged;
no reciprocal approximation or quantization precision change is introduced.
The generic `_matmul` and single-token GEMV retain the scalar packer as
independent references; shared packing currently optimizes prefill only.

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

### Original VME slowdown: scalar-packing baseline

A complete, streamed trace of the same measured Stories15M first-layer ELFs
reproduces the original 1,901,030 VME and 1,309,351 RVV kernel cycles. Both
traced memory dumps still match their respective ISA references. No kernel
instrumentation, additional layers, or software arithmetic changes are used.

The strongest software bottleneck is redundant scalar activation quantization.
Each hart owns different output columns, but independently scans and quantizes
all four input token rows into a private stack buffer. Across the seven
projections, the input reduction dimensions sum to 2,496; quantizing once
per projection requires 9,984 token activation elements. The trace instead executes 39,936
element divisions plus 1,248 scale divisions: the same input work is repeated
on all four harts. Each element also incurs scalar loads, abs/max, sign and
clamp branches, integer conversion and byte packing. RVV has no activation
quantization pass.

The next structural cost is the group-32 epilogue. Each group has independent
activation/weight scales, so the current kernel finishes and stores four
INT32 tile rows, reloads them, corrects the unsigned zero point, converts to
FP32, applies scales and reads/updates the FP32 output. Across four harts,
there are 4,608 tile-row stores and 4,608 vector epilogue iterations. The
existing FP32 RVV kernel instead keeps its output accumulator in vector
registers across the full reduction. Simply merging differently scaled
integer groups would change the computation and is not a valid shortcut.

| Dynamic instruction metric, all four harts | VME GEMM | Pure RVV |
| --- | ---: | ---: |
| Retired instructions | 1,553,808 | 212,156 |
| Scalar instructions, excluding vector/matrix config | 1,367,952 | 119,188 |
| Vector/matrix configuration instructions | 35,444 | 1,416 |
| Vector loads, including non-projection work | 95,194 | 42,152 |
| Scalar FP divisions, including non-projection work | 41,276 | 92 |

VME issues 9,216 `vtmmu` operations. Each K=4 step has four short activation
loads and four weight loads, giving 73,728 operand-load instructions before
epilogue and attention traffic. Four activation elements occupy only four
bytes per load. Threaded LSU-to-VME forwarding is disabled, so operands pass
through the vector register file. The weight-byte savings therefore do not
imply fewer load instructions or less dispatch/VRF overhead. These are
instruction counts, not measured AXI bandwidth or evidence that external
memory bandwidth is saturated.

As a localization aid, assign each interval between successive scalar-core
retirements to the preceding instruction's PC region. Hart 0's 1,901,303-cycle
observed timeline (including trampoline overhead) breaks down as follows:

| PC region | Retirement-interval cycles | Share |
| --- | ---: | ---: |
| Scalar activation quantization | 816,516 | 42.9% |
| Operand loads and matrix submission | 273,553 | 14.4% |
| Tile stores and fence | 34,939 | 1.8% |
| FP32 group epilogue and its waits | 465,585 | 24.5% |
| Other work, loop setup and synchronization | 310,710 | 16.3% |

All harts spend 816,516–816,540 such cycles in the quantization regions.
This is not an execution-unit utilization breakdown: queued matrix work can
finish while scalar execution is already in an epilogue region. Likewise,
the mean `vtmmu` commit-to-issue interval is 517.2 cycles and issue-to-result
interval is 41.8 cycles, but these intervals overlap across instructions and
harts; summing them would overcount elapsed time. Do not call the epilogue
share pure conversion time or the submission share pure matrix compute time.

This diagnosis motivated the shared RVV quantization and cross-projection
reuse now implemented in the prefill path. The counts and PC ranges in this
section describe the original scalar-packing ELF, not the optimized kernel.
Reducing group-result traffic and pipelining operand loads remain follow-up
work. The separately tested epilogue interleaving gain was only 0.35%; that
experiment did not address the redundant scalar packing.

To collect bounded-size summaries without keeping gigabytes of raw events:

```sh
python3 T1/tools/profile-rtl.py \
  --emulator /path/to/simulator --image /path/to/image.elf \
  --output /path/to/new-profile --dump-range 0x82000000:0x847499c0
```

Use the dump range for the actual image, not this example unconditionally.
`instructions.csv` records hart, PC, instruction count and following-retire
gaps; `profile.json` records event counts and T1 queue/result intervals.
Check the complete readback and MMIO cycles against the original run.
For the measured VME ELF, the seven half-open quantization PC ranges are
`801004ea:801005ac`, `8010099c:80100a60`, `80100e32:80100efa`,
`801019e0:80101aa8`, `8010216e:80102236`, `801025ee:801026b6`,
and `80102d94:80102e5c` (hex). The epilogue ranges are
`801006a8:80100708`, `80100b58:80100bba`, `80100ffa:80101064`,
`80101ba0:80101c0a`, `8010233e:801023aa`, `801027be:8010282a`,
and `80102f64:80102fd0`. Re-derive ranges from disassembly after code changes.

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
