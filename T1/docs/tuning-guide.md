# Tuning kernels for T1: method, rules of thumb, and traps

`performance-analysis.md` is the measurement record — every number, in the
order it was found, including the corrections.  This document is the
distillation: how to find a bottleneck on this machine, what to check in which
order, the rules that transfer to a new kernel, and the mistakes that produced
confidently-wrong conclusions along the way.

Everything here was measured on **blastoise** (VLEN 2048, DLEN 256) with the
cycle-accurate `t1rocketemu` RTL simulation, driving hand-written RVV kernels
for llama2 inference (`T1/examples/t1llama.mojo`).  Where a rule was measured
in one shape only, that is stated — this machine has punished generalization
more than once.

## 1. Know the machine's four numbers

Optimization on T1 is mostly deciding *which* of four ceilings a loop sits
under.  For blastoise, from `t1zaozi/params/blastoise/*.json`,
`designs/blastoise.toml` and `t1zaozi/src/VRF.scala`:

| Resource | Nominal | Measured best | Derivation |
|---|---|---|---|
| Execution | **8 f32/cycle** | 7.7-7.8 (96-98%) | `dLen 256` = `laneNumber 4` × `datapathWidth 64`; `datapathWidth = laneScale × eLen` |
| Memory | **32 B/cycle** | 28.1 (88%) | AXI data width 256 bit |
| VRF ports | **32 f32-accesses/cycle** *(at stock `portFactor` 4)* | — | `laneNumber × rfBankNum × ramWidth` = 4 × 4 × 64 bit = 1024 bit/cy = 32 f32; `rfBankNum = portFactor` (`--vrfBankSize`), `ramWidth = datapathWidth` |
| Vector instructions in flight | **4 executing + 1 latched** | 5 | `chainingSize` slots plus `requestReg`; on stock RTL a hardcoded 2-bit tag caps it at this regardless of `chainingSize` (PR #183) |

The VRF row is the one that surprises people, and it is the only one that
moves with configuration.  Per element, a load costs 1 write,
`vfadd.vv`/`vfmacc.vf` cost 2 reads + 1 write, `vfadd.vf` costs 1 read + 1
write.  At full rate an arithmetic instruction needs 24 accesses/cycle and a
load needs the other 8: **load + MAC at peak is exactly 32 of 32 at stock
`portFactor`, zero headroom.**  That means the pair cannot overlap for free
even in principle — the nominal floor is 32 cycles/pair — but it does not by
itself explain the measured 56 cycles, which is only 57% of that floor; the
rest is bank conflicts and arbitration, which is why doubling the banks
recovers most of it (§6).  At `portFactor` 8 the budget is 64 accesses/cycle
and VRF ports stop being the binding currency.

Two more machine constants worth memorizing:

- The **reduction unit** is serialized in hardware — `vfredusum.vs` is
  decode-`special` (`isSpecial.scala`), so it takes the last slot and only when
  that slot is idle.  Empirically its *occupancy* fits `126 + 1.25 × vl`
  cycles, with the next reduce accepted at roughly half that (measured II ~104
  cy at vl=64).  Budget a stream of reduces with the II, a single one with the
  occupancy.  Either way: keep it out of inner loops.
- The **scalar core** is in-order and single-issue, **no address is cacheable**
  (`--cacheable` in `designs/blastoise.toml` is an all-ones 32-bit bitpat, so
  it matches only `0xFFFF_FFFF` while the SRAM lives at `0x8000_0000`), and
  only **one uncached access** may be in flight
  (`maxUncachedInFlight == 1`, `HellaCache.scala`).  An *exposed* scalar load
  or store therefore costs ~43 cycles of drained vector unit.

## 2. Method: how to measure without fooling yourself

**Never instrument.**  Inserting timing instructions changes the schedule you
are trying to measure.  Everything here comes from the simulator's retirement
trace (`T1Issue`/`VrfWrite`/`T1Retire`/`T1Release` with cycle stamps) and the
ISA model's PC histogram — both entirely non-invasive.

**Always measure marginals, and always in the saturated regime.**  The single
most expensive mistake of this project (§7.1).  A launch has a fixed ~150-cycle
shadow, and short instruction streams hide under it completely:

```
m4 vfadd.vv instructions:   2    4    8   16   32    64
cycles:                    150  152  152  152  160  1168
marginal cy/instruction:     -  1.0  0.0  0.0  0.5  31.5
```

Take `(cycles(N₂) − cycles(N₁)) / (N₂ − N₁)` with **both** points past the
knee, and verify with a third.  The knee is wherever `N × (per-instruction
cost)` clears the ~150-cycle shadow, so locate it per shape: for m4
`vfadd.vv` at vl=256 it is ~48 instructions; for cheaper shapes it is later.

**Use both simulators for what each is good at.**  The `pokedex` ISA model
retires ~1-3M instructions/s against the RTL's ~10-20k cycles/s
(`T1/runtime/README.md`): use it for correctness (token-exact generation) on
every change, the RTL only for cycles.  A change is not done until it is
token-exact on both.

**The RTL is deterministic.**  Cycle counts repeat bit-identically across runs
and across layers, so a 0.5% difference is signal, not noise.  That is what
makes single-run A/B comparisons legitimate here.

**Every conclusion needs a control that would have failed if it were wrong.**
Run the same binary on the baseline simulator; run the same simulator on a
binary with the suspected feature removed; vary one operand class at a time.
Every trap in §7 was caught by a control and would have been prevented by
running it first.

**Compute the roofline before optimizing.**  Work out the compute, memory and
VRF-access floors for the shape at hand.  The useful question is never "is it
fast?" but "which ceiling is it under, and at what percentage?".

## 3. The bottleneck ladder

Ordered by the cost measured on a *real kernel*, not on a microbenchmark:

1. **Reductions in an inner loop.**  `vfredusum` is serialized and
   non-pipelined.  Restructuring the matvec around an offline-transposed
   weight matrix removes them entirely — the largest single win of the project.
2. **Scalar memory operations exposed inside a vector stream.**  ~43 cycles
   each when exposed (§4).  Batching them out of the dependent path: **1.34×**
   on the layer kernel.
3. **VRF port pressure in load+arith mixes.**  A configuration change worth
   **1.19-1.26×** on real kernels (§6); in software, prefer instruction forms
   with fewer VRF accesses per element.
4. **Short or partially-used vectors.**  Streaming ops cost what their `vl`
   says, but init/fold/reduce steps written at VLMAX pay full width for
   however few elements are useful — one of the three reasons raising VLEN is
   worse for this kernel family.
5. **Writing v0** — a *latent* hazard rather than a measured kernel cost:
   eliminating it from these kernels was cycle-neutral (the v0 writes hid
   behind the load stream), but a dependent accumulate chain on v0 measures
   2.5-4× the same chain on an ordinary register, so a schedule change or a
   faster memory system can expose it at any time.

## 4. Rules of thumb

**Registers**

- **Never write `v0`.**  It is the RVV mask register, and on stock RTL T1 ORs
  `vd == v0` into its special-instruction term, pinning such instructions to
  the last of the four slots and admitting them only when that slot is idle
  (T1 issue #180; PR #181 removes the term).  A dependent chain accumulating
  into v0 measures 2.5-4× the same chain on v8/v24 (short-stream micro — read
  the ratio, not the absolutes).
- **`vmv.v.i` is separately expensive, for a different reason.**  It is not
  selected by the v0 path; it is listed with the `vmerge` family in
  `isSpecialslot.scala`.  In an identical load-interleaved stream a
  `vle32` + `vmv.v.i` pair cost **2.3×** a `vle32` + `vfadd.vv` pair.  Keep it
  out of inner loops — hoist accumulator zeroing to the strip level.
- **Dependent arithmetic chains are not inherently expensive** on ordinary
  registers: the same chain measures like an independent stream.  Do not burn
  registers on multiple accumulators to "break dependencies" — on this machine
  that was a workaround for the v0 penalty in disguise.
- **Respect LMUL alignment.**  With LMUL=8 only v0/v8/v16/v24 are legal group
  numbers; assemblers accept `v28` and you will be measuring undefined
  behaviour (§7.3).

**Instruction selection**

- **Count VRF accesses per element — that is the currency.**  `vfmacc.vf`
  (read vd, read vs2, write vd) and `vfadd.vv` (2 reads, 1 write) have
  identical traffic and measure identically to the cycle (56.0 cy/pair with a
  load, at e32/m4 vl=256, stock config); `vfadd.vf` drops one read and costs
  exactly 7 cycles less.
- **Streaming cost scales with the `vl` you set, not with VLMAX** — a
  strip-mined partially-filled group is not wasted work (m8 at vl=288 costs
  36.9 cy, at vl=512 costs 65.5; both ~7.8 elem/cycle).  This holds only for
  instructions you actually strip-mine: init, fold and reduce steps written at
  VLMAX pay full-width occupancy regardless.
- **LMUL is a register-budget decision where no fold/reduce ladder survives**
  (m4 and m8 land within 1% in the reduction-free transposed matvec), and an
  efficiency decision where one does: the reduction-based kernel measured
  896,996 / 751,127 / 758,733 layer cycles at LMUL 1 / 2 / 4.

**Scalars**

- **An exposed scalar load or store costs ~43 cycles** — integer or floating
  point, load or store, result consumed or dead.  It is *not* the data
  dependency: a register producer feeding a vector instruction one cycle later
  (`fmv.w.x`, `fadd.s`) is free.
- **Placement is everything.**  Between a load and its dependent MAC the stall
  is fully exposed (61.4 → 104.6 cy/column); hoisted to a batch boundary it
  runs under the shadow of already-dispatched vector work (71.4 on stock, 58.1
  on the tuned config), and inside a batch only the first access pays ~43
  while the rest pay ~8.7.  Eight is enough on stock (72.8 at four, 71.4 at
  eight); on the tuned config 12 and 16 are flat with each other (57.4/57.1).
  Never spread them among the MACs: that measured 98.6.
- **Correctness, not just speed: never place an `flw` immediately before the
  `vfmacc.vf` that consumes it.**  On stock RTL the precise FP interlocks are
  gated by `d.fp`, which is 0 for rv_v instructions, so the vector instruction
  can capture a stale `fs1` (T1 issue #178; PR #179 fixes it).  Batching
  satisfies this by construction — every scalar ends up ≥ `batch` instructions
  from its use.
- **Scalar bookkeeping is free while long vector phases are in flight.**
  Address increments, pointer chains and loop branches hid completely in the
  reduce-heavy kernel (96-100% of scalar retires under vector execution).  In
  finely interleaved kernels the shadow shrinks — the same metric fell to 57%
  on the software-pipelined variant — so do not assume it, measure it.

**Memory**

- **Reuse buys nothing.**  The data window is uncached, so re-reading one
  address costs exactly what streaming costs (61.4 both).  Do not distort a
  layout chasing locality.  Access *form* may still matter: only unit-stride
  was measured; strided, segmented and indexed loads were not.
- **Instruction-level parallelism above 4 is unusable** — that is the number
  of T1 slots; a fifth instruction can sit latched in `requestReg` but is not
  executing.

## 5. Software patterns that won

| Pattern | Effect | Cost / precondition |
|---|---|---|
| Offline weight transpose → reduction-free matvec (`vfmacc.vf` column accumulation) | removes `vfredusum` from the inner loop entirely | weights must be transposed once during staging |
| Batched scalar prefetch into f-registers | layer 341,281 → 254,311 (**1.34×**) | `n` divisible by the batch; ≤ 8 f-registers; also the #178 correctness workaround |
| v0-free register plan | cycle-neutral today | removes a latent 2.5-4× hazard |
| Uniform init + `tu` strip-mined loop | correct for any `n`, including `n` < one chunk | ~9% slower than a `vfmul`-first-chunk variant (817,844 vs 751,127 at VLEN 2048, LMUL 2); add a comptime fast path when the caller guarantees `n ≥ chunk` |
| Comptime asm generation parameterized by SEW / LMUL / vl / batch | every shape sweepable via `-D` without touching assembly | generator must enforce LMUL-aligned register groups |
| Allocating streamed read-only buffers first | memory-dump readback costs no simulated cycles | requires the `T1RT_MEMDUMP` simulator patch |

## 6. Hardware knobs, measured (DLEN 256 fixed)

These are **RTL elaboration parameters**, not runtime flags: set them in
`designs/blastoise.toml` (see `T1/runtime/blastoise-tuning.patch`) and rebuild
the simulator.  Area and timing costs were **not** evaluated — only throughput.

| Knob | Verdict |
|---|---|
| `vrfBankSize` (= `portFactor`) 4 → 8 alone | **1.19× layer / 1.25× head**; mixed pair 56.0 → 43.2 cy |
| `vrfRamType` `p0rw` → `p0rp1w` alone | comparable (1.21× layer); better than banks for `vfmacc.vf` specifically |
| both together | **1.26× layer / 1.45× head**; mixed pair 43.1 cy = 86% of the achievable-load roofline |
| `vrfBankSize` 8 → 16 | nothing — VRF banking saturates at 8 |
| `chainingSize` 4 → 8 | nothing: the in-flight tag caps the window at 5 regardless (PR #183), and arithmetic is already at the datapath roofline |
| `laneScale` | 2 (4 × 64b) is the optimum: narrower lanes tax the reduction, wider lanes tax the serialized per-lane write path (pair period 550 → 607 cy, tracking `vmv.v.i` init). Streaming itself is insensitive — an earlier "wider lanes lose in streaming" reading was a retirement-shadow artifact |
| VLEN ↑ | worse for GEMV rows shorter than the chunk (reduction-based kernel, −81% at VLEN 4096: reduce stub and fold ladder both scale with VLEN). Not measured on the reduction-free kernel |

Recommended operating point for LLM-style kernels at DLEN 256:
`--vrfBankSize=8 --vrfRamType=p0rp1w`, subject to an area/timing review.

## 7. Traps, and the controls that catch them

Each produced a confidently-stated wrong conclusion before a control caught
it.  They are listed because the *shape* of each mistake recurs.

### 7.1 Unsaturated microbenchmarks

Streams of 16-48 instructions sat inside the launch shadow, so measured IIs
were understated by up to 2× and, worse, ordered wrongly relative to each
other.  This produced an incorrect "loads and arithmetic never overlap"
conclusion and a performance claim in an upstream PR that had to be withdrawn.
**Control:** sweep the stream length and locate the knee before trusting any
marginal.

### 7.2 A microbenchmark that used v0 as its accumulator

The dependent-chain benchmark accumulated into v0, so it measured the v0
dispatch penalty and was read as "dependent arithmetic never chains".
**Control:** re-run the same shape with the operand class varied — the
identical chain on v8 and v24.

### 7.3 Illegal register groups

`vfadd.vv v8, v24, v28` at LMUL=8 assembles but is architecturally ill-formed
(groups must be LMUL-aligned), so the numbers described nothing.
**Control:** re-derive operand legality whenever the shape changes; better,
generate assembly from a generator that cannot emit misaligned groups.

### 7.4 A harness that ignored its own argument

The run script hardcoded the baseline simulator for `SIM=rtl`, so two
"configuration variants" were silently measured on stock hardware — and
produced cycle counts identical to the baseline, which was the tell.
**Control:** identical results across supposedly different builds are a bug
report about the harness, not a finding.

### 7.5 Attributing a cost to the nearest dependency

The ~43-cycle scalar cost looked like the RAW through `f[rs1]` feeding
`vfmacc.vf`.  It is not: an integer load into a dead register, and even a
store, cost exactly the same, while a register producer feeding the same
vector instruction is free.
**Control:** vary dependence and operation class independently — dependent vs.
dead, load vs. store, integer vs. float.

### 7.6 Citing a working tree that already contains your own fix

Statements about "what the hardware does" must be checked against the upstream
commit, not a local tree carrying your patches: a review of this very document
reported that `vd == v0` is *not* special-cased and that the instruction tag
*does* scale with `chainingSize` — both true only of the patched tree.
**Control:** cite behaviour as of a named upstream revision, and say which
issue or PR changes it.

The common lesson: a measurement that agrees with the hypothesis is worth much
less than one that would have disagreed if the hypothesis were false.

## 8. Where the kernels stand, and what is left

stories15M, RTL cycles, token-exact against the ISA model.  Per-token figures
are derived (6 layer launches + 1 head launch), not separately measured:

| Configuration | layer kernel | head kernel | per token (derived) |
|---|---|---|---|
| generic Mojo SIMD (start) | 3,343,155 | — (never measured) | — |
| first hand-written asm (v2) | 1,465,269 | 15,081,105 | — |
| current kernels, stock hardware | **254,311** | **2,105,093** | 3,630,959 |
| same kernels, `vrfBankSize=8 p0rp1w` | **201,368** | **1,454,575** | 2,662,783 |

Against the memory roofline on `vrfBankSize=8 p0rp1w` (weights must stream
through once; a pure load stream achieves 28.1 B/cycle = 88% of AXI):

| Kernel | achieved | of AXI peak | of achievable load rate |
|---|---|---|---|
| layer | 19.8 B/cy | 62% | 70% |
| head | 25.3 B/cy | 79% | **90%** |

The head kernel is essentially at the memory roofline.  The layer's remaining
30%, decomposed at the kernel's own shape (cycles per column): pure load 41.0
→ load+MAC 47.3 (+6.3, vector-side startup) → +batched scalars 58.1 (+10.8,
the scalar tax) → real kernel ~65 (+~7, short and mixed `vl` across output
strips).  Software has exhausted the scalar third (deeper batching is flat,
interleaving is worse); the strip-shape third is still open; the vector-side
third and the scalar third both want hardware:

- a cacheable device window for the D$ — which first needs coherence with the
  vector unit's stores, or an explicit fence;
- more than one uncached access in flight;
- accumulator forwarding that keeps a MAC chain's `vd` out of the VRF (T1
  issue #182), which would take the GEMV inner loop from four VRF accesses per
  element to two.

## 9. Upstream findings from this work

| Thread | What it is |
|---|---|
| xinpian-tech/T1 #175, PR #176 | `chainingSize > 4` does not elaborate — slot token reports unwired |
| #178, PR #179 | stale `fs1` on OPFVF after `flw`: the precise FP interlocks are gated by `d.fp`, which is 0 for rv_v (correctness bug) |
| #180, PR #181 | any `vd = v0` instruction is pinned to the last sequencer slot, serializing dependent consumers |
| #182 | load/exec overlap limited by VRF port provisioning; includes the measured `portFactor` sweep and the accumulator-forwarding proposal |
| PR #183 | the in-flight instruction tag is a hardcoded 2 bits, so `chainingSize > 4` buys nothing (performance claim withdrawn; mechanism stands) |
