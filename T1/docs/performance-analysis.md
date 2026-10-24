# T1 microarchitectural performance analysis of the llama kernels

Non-invasive analysis of the hand-written RVV llama kernels
(`T1/examples/t1llama.mojo`) on the cycle-accurate `t1rocketemu` RTL
simulation of the **blastoise** design.  All data comes from the
simulator's per-instruction retirement trace (`rtl-event.jsonl`:
`T1Issue`/`T1Retire`/`T1Release` events with cycle timestamps for every
vector instruction, plus scalar retires) — no instrumentation
instructions were inserted into the kernels.

Workload: stories15M (dim 288, hidden 768), one transformer layer
launch (2,976 matvec rows) and the classifier head launch
(32,000 × 288).  The same kernels run TinyLlama-1.1B; only `n` and the
row counts change.

## 1. The machine (blastoise)

From `t1zaozi/params/blastoise/*.json`:

| Parameter | Value | Consequence |
|---|---|---|
| VLEN | 2048 | m1 = 64 f32 lanes-worth of state |
| Lanes × datapath | 4 × 64 bit | DLEN = 256 bit = 8 f32/cycle execute |
| `chainingSize` | 4 | at most 4 vector instructions in flight per lane |
| VRF | banked SRAM, `p0rw` (single port), `portFactor` 4, read latency 2 | loads writing the VRF contend with VFU reads |
| LSU | 3 MSHRs/bank, AXI 256-bit data, `toVRFWriteQueueSize` 96 | 32 B/cycle peak memory bandwidth, deep outstanding |
| Scalar core | Rocket (in-order), decoupled vector issue queue | scalar instructions run under vector execution |

## 2. Headline results (RTL, cycle-accurate)

| Kernel version | layer kernel | head kernel |
|---|---|---|
| v1 generic Mojo SIMD | 3,343,155 | — |
| v2 first asm (m4, vfredusum, no tails) | 1,465,269 | 15,081,105 |
| v5 row-loop-in-asm, 6-row groups, packed stores | 1,088,598 | 11,147,993 |
| v6 two-level `vfadd` fold before short reduce | **824,567** | **8,310,745** |

Total: **4.05×** (layer) / **1.81×** (head, v2 baseline) over the
generic-SIMD kernels, entirely from
restructuring against measured microarchitectural behavior.  Cycle
counts are bit-identical across layers and across runs — the RTL is
deterministic, which makes every one of these numbers exactly
reproducible.

## 3. Chaining works — and hides all scalar overhead

From the v5 layer-launch trace (20,550 vector instructions):

- Unit-stride `vle32` (m4, vl=256) issue back-to-back: the issue-to-issue
  gap median is **1 cycle**; up to 5 vector instructions are
  simultaneously in flight, and ≥2 are in flight during **98.5%** of
  kernel cycles.
- The dependent `vfmul`/`vfmacc` issues ~33 cycles after the `vle32`
  producing its operand and executes while the load is still streaming
  beats — Load→Exec chaining exactly as advertised by the Cray-style
  design.
- **96.6% (layer) / 100.0% (head) of scalar retires happen while vector
  instructions are in flight.**  The address increments, pointer chains
  and loop control that dominate the scalar instruction count (§ audit
  in the previous commit) cost **zero cycles**: Rocket runs them in the
  shadow of vector execution.

The last point corrects the natural reading of the instruction-mix
metric: pushing dynamic RVV share from 47.8% to 71.5% mattered because
it *removed serialized work* (scalar tails, reduction trees, a
byte-wise memset), not because scalar instructions were expensive — the
surviving ones are already free.

## 4. Anatomy of one 6-row matvec group (v5, n=288)

Timeline reconstructed from issue/retire events (one group, 2,090
cycles):

```
phase        cycles   what the trace shows
loads+MACs   0-585    7 x vle32 pairs cadence ~66 cy; vfmul chained ~33 cy
                      behind each load; 2 loads overlapped throughout
reduce      585-1947  6 x vfredusum (vl=256): occupancy 446 cy each,
                      accepted every ~224 cy -> phase 1,362 cy
pack/store 1725-2133  5 x vslideup + vse32, chained under the reduce tail
next group  2090-     first vle32 of the next group issues 1 cy after vse32
```

Aggregated over the launch: the reduce phase costs **62.1%** of all
kernel cycles; loads are in flight during only 30.1% of them.

### 4.1 The reduction unit is the bottleneck

Measured `vfredusum` behavior:

| vl (e32) | occupancy | acceptance interval |
|---|---|---|
| 256 (m4) | 446 cy | ~224 cy |
| 64 (m1, after fold) | 206 cy | ~103 cy |

Linear fit: **`vfredusum ≈ 126 + 1.25 × vl` cycles, non-pipelined**
(next reduce accepted at roughly half the previous one's occupancy).
Reductions execute near element-serial and do not benefit from the four
lanes.  Consequences:

- Six back-to-back full-width reduces serialize into a 1,362-cycle
  phase during which the (independent!) next group's loads cannot
  issue: with `chainingSize = 4`, the outstanding reduces occupy the
  instruction slots and stall the in-order issue front end.
- This also retroactively explains v1: the generic `reduce_add`
  lowering (log2 slide/add tree + `vfredosum`) paid this fixed cost
  *and* a tree of dependent slides per row.

### 4.2 v6: fold with the lanes, reduce only the stub

The lanes add at 8 f32/cycle and chain; the reduction unit does ~0.8
elem/cycle and does not.  So v6 folds each m4 accumulator 256→128→64
with two lane-parallel `vfadd.vv` (measured occupancy 67 cy, fully
chained) and reduces only 64 elements:

- fold+reduce phase per group: 1,362 → **830 cycles**
- layer kernel: 1,088,598 → **824,567 cycles (-24%)**

The reduce phase is still 50% of the kernel — `vfredusum`'s ~126-cycle
fixed cost now dominates its own execution.

### 4.3 Load phase: 48% of peak bandwidth, VRF-port bound

During the load phase the cadence is one m4 load (1 KiB = 32 AXI
beats) per ~66 cycles ≈ 15.5 B/cycle against the 32 B/cycle AXI peak.
The LSU (3 MSHRs, 96-deep VRF write queue) keeps two loads in flight,
but every load must *write* the VRF while the chained `vfmul`/`vfmacc`
*reads* two operand groups from it — with single-port (`p0rw`) banked
VRF RAM at `portFactor` 4, the write and read streams contend for bank
ports.  The ~2× gap between achieved and peak bandwidth is consistent
with load writes losing roughly half the port slots to execution reads
(hypothesis; confirming it needs VRF-port waveforms rather than the
retirement trace).

## 5. Where the next factor is (software)

In v6, per group: loads ~585 cy (irreducible at current VRF: 252 beats
minimum = 43% of that), fold+reduce ~830 cy, of which only ~180 cy is
lane work.  The two phases barely overlap because the in-order front
end cannot issue the next group's loads past the outstanding reduces.

1. **Software pipelining across groups** — start group *i+1*'s loads
   before group *i*'s fold/reduce section.  Requires freeing two vector
   register groups for double buffering (5-row groups instead of 6) and
   staying within the 4-slot budget: expected to hide most of the load
   phase under the reduce phase, bounding the group at ~max(585, 830) ≈
   830 cy → ~25% further.
2. **Amortize the reduce fixed cost** — reduce two rows per
   `vfredusum` by first `vslideup`-merging two 64-elem stubs into one
   128-elem register (1 fold more, half the reduces): saves ~6 × 63 cy
   fixed per group at the cost of 3 slides + later scalar split; ~10%.
3. **Attention at long context**: `_dot`'s vl=hs=64 reduce pays the
   same 126-cycle fixed cost per cached position; at pos ≫ 64 this
   dominates attention.  Batch the dot products (one m4/m8 load of
   several K rows, MAC against broadcast q, fold, one reduce per 4-8
   positions) before long-context runs.

## 6. Hardware co-design notes

The same numbers, read as design feedback for blastoise-class configs:

- **A pipelined (or logarithmic) reduction unit is the single highest
  value change for GEMV/attention workloads**: `vfredusum` at
  126 + 1.25/elem non-pipelined makes reductions 50-62% of a
  bandwidth-starved kernel.  A tree reducer at DLEN width would cut the
  variable cost to vl/8 and pipelining would remove the serialization;
  both together turn the reduce phase into noise (<5%).
- **`chainingSize` 4 is the second limiter**: long-occupancy
  instructions (reduces today, possibly gathers tomorrow) fill all four
  slots and stall issue of independent memory traffic.  6-8 slots, or
  an issue policy that reserves one slot for loads, would let the load
  phase run under the reduce phase with no software pipelining at all.
- **VRF port pressure caps streaming at ~50%** of AXI bandwidth in
  load+MAC phases.  `portFactor` or a two-port VRF option trades area
  for the other 2× — worthwhile only after the reduction unit, since
  today the load phase is not the critical path.
- The scalar core is never the problem: a minimal in-order Rocket
  fully hides the bookkeeping of hand-tiled vector loops behind
  chaining.  Spending area on the vector side (reduction, slots, VRF
  ports) beats any scalar-side improvement for these kernels.

## 7. Hardware tuning under DLEN = 256 (measured)

Config-space experiments on `designs/blastoise.toml`
(`T1/runtime/blastoise-tuning.patch`), v6 kernels, stories15M:

| Config (DLEN = 256 fixed) | layer kernel | head kernel | verdict |
|---|---|---|---|
| baseline: `p0rw`, 4 banks, 4x64b lanes | 824,567 | 8,310,745 | |
| **`p0rp1w` (1R1W two-port VRF) + 8 banks** | **784,929 (-4.8%)** | **7,990,760 (-3.8%)** | **adopted** |
| + `laneScale=1` (8x32b lanes) | 851,885 | — | rejected: `vfredusum` occupancy 206 -> 252 (deeper cross-lane combine), loads unchanged |
| `chainingSize=8` | does not elaborate | — | generator hardcodes 4 slot pipelines; filed [xinpian-tech/T1#175](https://github.com/xinpian-tech/T1/issues/175) |

The two-port VRF confirms §4.3 exactly where predicted: m4 load
occupancy 123 → 92 cycles, load cadence 66 → 49 cycles — and the
fold+reduce phase is unchanged (830 → 835), because the reduction unit
does not touch VRF load-write ports.  With the load phase now ~44%
faster and the reduce phase untouched, the reduce share rises further:
the config space under DLEN = 256 is effectively exhausted, and the two
remaining walls — the non-pipelined ~`126 + 1.25/elem` reduction unit
and the hardcoded 4 instruction slots — both require RTL changes
(§6, issue #175).

## 8. Does 8-deep chaining help?  (measured: no — and why)

With the elaboration fix (xinpian-tech/T1#176) `chainingSize=8` builds and
runs.  Two kernel generations, both configs, stories15M:

| Kernel | slots=4 | slots=8 | delta |
|---|---|---|---|
| v6 (fold+reduce after each 6-row group) — layer | 824,567 | 833,917 | +1.1% |
| v6 — head | 8,310,745 | 8,422,698 | +1.3% |
| v7 (software-pipelined pairs, conflict-free registers) — layer | 759,219 | 755,400 | **-0.5%** |

v6 cannot exploit extra slots for a *software* reason: its reduce
destinations (v0-v6) alias the next group's load buffers (w v0-v3,
x v4-v7), so the register dependence — not the slot count — blocks
cross-group overlap.  v7 removes the aliasing (buffers v0/v4, reduce
stubs v24-v26) and interleaves each pair's fold+`vfredusum` into the next
pair's load stream: 8.6% faster than v6, but **identical between 4 and 8
slots**.

The v7@cs8 retirement trace explains it: the vector unit is completely
idle **46.9%** of kernel cycles, at most 5 instructions ever in flight,
and scalar-under-vector overlap drops to 57% (from 96-100% in v6).  The
issue front end — vtype ping-pong between the m4 compute shape and the
m1/m2 fold/reduce shapes, plus the scalar bookkeeping between interleave
points that can no longer hide under long vector phases — is the limiter.
Four slots were never the binding constraint for this kernel family;
the §6 recommendation is revised accordingly: pipelining/shortening the
reduction unit remains the high-value hardware change, extra slots are
not.

## 9. SEW/LMUL/vl-templated kernels and the LMUL sweep

All hand-written helpers are now generated at compile time from
SEW/LMUL parameters (`t1llama.mojo`: `MV_LMUL`, `EW_LMUL`, overridable
with `-D T1_MV_LMUL=<1|2|4>`); vl always follows as VLMAX of the shape
with `vsetvli` strip-mining for remainders, and the matvec derives its
whole register allocation (load buffers, double-buffered accumulator
pairs, reduce stubs, fold ladder depth = log2 LMUL) from the parameter.

Layer kernel, stock config, greedy output identical for all shapes:

| MV_LMUL | chunk (e32) | layer cycles |
|---|---|---|
| 1 | 64 | 896,996 |
| **2** | **128** | **751,127** |
| 4 | 256 | 758,733 (= hand-written v7 within 0.06%) |

LMUL=2 edges out LMUL=4: the shorter reduce stubs and cheaper fold
ladder outweigh the doubled loop-iteration count; LMUL=1 loses to pure
per-iteration overhead.  The generator makes such sweeps a rebuild flag
instead of an assembly rewrite.

## 10. Raising the chime (VLEN/DLEN): measured, and it hurts

With DLEN fixed at 256, VLEN raised 2048 -> 4096 doubles the chime (the
cycles one vector instruction occupies: chunk/8 for e32).  The kernel
generator only needs `-D T1_VLEN=4096`; the RTL config changes
`vlen`/`zvl4096b` in `designs/blastoise.toml`.  Layer kernel, identical
(any-n-safe) generator everywhere, outputs token-exact in all cells:

| VLEN | MV_LMUL | chunk | chime/instr | layer cycles |
|---|---|---|---|---|
| 2048 | 2 | 128 | 16 cy | **817,844** |
| 2048 | 4 | 256 | 32 cy | 1,019,522 |
| 4096 | 2 | 256 | 32 cy | 1,047,535 |
| 4096 | 4 | 512 | 64 cy | 1,480,733 |

Monotonically worse with chime, -81% at the extreme.  Three mechanisms,
all visible in the earlier traces:

1. The m1 reduce stub scales with VLEN (64 -> 128 e32 lanes), so
   `vfredusum` (~126 + 1.25/elem, serial) costs 206 -> 286 cycles per
   row — and reductions are already the critical resource.
2. The fold ladder and accumulator init run at the VLMAX of each level
   regardless of how many lanes are useful; at n=288 a 512-lane chunk
   pays full-width occupancies for 288 lanes of work.
3. The overhead that longer vectors classically amortize — scalar loop
   control per instruction — was already free: chaining and the
   decoupled scalar core hide it at chime 8-16 (§3).

The matched-chunk cells make the reduce effect clean: VLEN 2048/LMUL 4
and VLEN 4096/LMUL 2 execute the same chunk length and differ mainly in
stub length (64 vs 128), costing ~3%.

Corollary for blastoise-class configs: with a serial reduction unit and
short GEMV rows, extra VLEN is pure downside for this workload family —
spend the VRF area on reduction throughput instead.  (Long-vector
workloads without reductions — SAXPY-like streaming — are where higher
chime pays.)

Note: this section uses the any-n-safe generator (accumulator zero-init
+ uniformly strip-mined `tu` loop), required once chunk can exceed the
smallest matvec n.  It costs ~9% versus §9's vfmul-first-chunk variant
at VLEN 2048 (817,844 vs 751,127 at LMUL 2); a comptime fast path for
callers that guarantee n >= chunk is the obvious follow-up.

## 11. The DLEN axis: sublinear, and the reduce unit is why

The complementary sweep: VLEN fixed at 2048, DLEN 128/256/512 (2/4/8
lanes x 64 bit; the AXI data width follows DLEN, so this scales execute
*and* memory bandwidth together).  The kernels are unchanged — DLEN is
architecturally invisible to RVV code.  Layer kernel, uniform generator:

| DLEN | chime (m1) | MV_LMUL=2 | MV_LMUL=4 | SAXPY n=4096 |
|---|---|---|---|---|
| 128 | 16 cy | 1,304,573 | 1,732,957 | 4,294 |
| 256 | 8 cy | 817,844 | 1,019,522 | 3,390 |
| 512 | 4 cy | **693,220** | 714,771 | 2,880 |

Doubling DLEN buys 1.60x at the low end but only **1.18x** from 256 to
512, against 2x the datapath and bus area.  The Amdahl term is the same
serial reduction unit as everywhere else in this document: its
throughput does not scale with lanes (§ hardware-tuning even measured it
*degrading* with more lanes at fixed DLEN), so as the streaming phases
shrink, the reduce share of every row grows.  Note LMUL=4 gains more
from DLEN=512 than LMUL=2 does (its longer chunks actually use the wider
datapath), nearly closing the L2/L4 gap.

Taken together, the two chime sweeps give a clean design statement for
GEMV-class workloads on blastoise-family configs:

- chime up via VLEN: strictly worse (serial reduce work per row scales
  with VLEN);
- chime down via DLEN: better but sublinear (serial reduce share grows);
- every axis measured — VLEN, DLEN, LMUL, lanes, VRF ports, slots —
  points at the same conclusion: **pipeline the reduction unit first;
  every other knob is second-order until then.**

## 12. Lane organization (laneScale): a shallow optimum at 64-bit lanes

Last axis: how the fixed DLEN=256 is sliced into lanes
(`--laneScale`: lane datapath = 32 x laneScale bits).  VLEN 2048,
MV_LMUL=2, uniform generator, stock config otherwise:

| laneScale | lanes | layer cycles | SAXPY n=4096 |
|---|---|---|---|
| 1 | 8 x 32b | 853,203 (+4.3%) | 3,390 |
| **2** | **4 x 64b** | **817,844** | 3,390 |
| 4 | 2 x 128b | 903,469 (+10.5%) | 3,390 |

SAXPY is bit-identical in cycles across all three — pure streaming sees
only DLEN.  The GEMV kernel sees the organization: more, narrower lanes
add cross-lane coordination (the earlier tuned-config experiment
measured `vfredusum` occupancy 206 -> 252 at 8 lanes — consistent with
853k here); fewer, wider lanes cost lane-level parallelism in the fold
and elementwise phases without helping the serial reduction at all.
Stock blastoise's 4 x 64b is the right slicing.

### 12.1 Why both directions lose: the per-class decomposition

Retirement traces for all three slicings (same binary, so instruction
counts are identical -- 36,979 vector instructions per layer launch):

| occupancy p50/p90 (cy) | 8 x 32b | 4 x 64b | 2 x 128b |
|---|---|---|---|
| `vle32` m2 vl=128 | 90 / 169 | 91 / 170 | 91 / **236** |
| `vfmacc` m2 vl=128 | 65 / 76 | 66 / 90 | 66 / **165** |
| `vmv.v.i` m2 | 140 | 139 | **171** |
| `vfredusum` m1 vl=64 | **170** / 254 | 148 / 208 | **140** / 190 |
| `vfadd` m1 (fold) | 60 | 60 | 60 |
| layer total | 853,203 | 817,844 | 903,469 |

laneScale moves cost between two different units:

- **More, narrower lanes tax the reduction.**  `vfredusum` occupancy
  climbs 140 -> 148 -> 170 across 2/4/8 lanes: the reduce combines
  per-lane partials through the mask unit, and both the cross-lane
  combine chain and the per-lane orchestration grow with lane count.
  (In-lane chunk count moves the *other* way --
  `reduceResultChunkCount = datapathWidth/eLen` is 4 at 128-bit lanes
  vs 1 at 32-bit -- but the cross-lane part dominates.)  Streaming is
  actually *best* at 8 lanes (tightest p90s, `vfmacc` cadence 42 vs 49
  cy): per-lane port pressure is lowest at 1 elem/lane/cycle.
- **Fewer, wider lanes tax the serialized inter-pair segment — and the
  load "tail" is a measurement artifact.**  Two follow-up experiments
  falsified the obvious bank-conflict reading of the p90 table: at
  2 x 128b, neither `vrfBankSize` 4 -> 8 nor a `p0rp1w` two-port VRF
  removes the 27.9% of loads at ~230 cycles (the share is bit-identical
  in all three configs; each shaves only ~3.5% of the span, 903,117 ->
  873,893 / 869,720, with load p50 91 -> 83/85 — a small real conflict
  term).  Tracing the slow loads shows they retire a median of **8
  cycles after their program-order predecessor retires**: they are the
  first loads of each pair, issued while the previous pair's
  init/fold/reduce/store chain is still retiring, and their occupancy
  measures **in-order retirement queueing**, not execution stalls.  The
  real regression is in that serialized chain: the steady-state pair
  period grows 550 -> 607 cycles, matching `vmv.v.i` init +32 x2 and
  related write-side growth, while load/MAC p50s are unchanged.  At two
  lanes, every per-lane structure that is not datapath — write queues,
  slot state machines, mask-unit interfaces — is halved while per-lane
  work doubles: the coordination fabric thins out even though the
  arithmetic bandwidth is identical.

So the axis is a genuine tradeoff between cross-lane coordination
(reduction path) and per-lane VRF port bandwidth (streaming path), and
at DLEN = 256 the crossover sits exactly at 4 x 64b.  It also cleanly
explains the hardware-tuning section's earlier observation (reduce
206 -> 252 at 8 lanes on the tuned config) and predicts that a two-port
VRF shifts the optimum toward wider lanes, while a pipelined reduction
unit shifts it toward narrower ones.

## 13. Startup time and dead time, measured per unit

Occupancy at two vector lengths (vl=128/32) linearly separates each
unit's pipeline-fill **startup** (intercept) from marginal throughput
(slope); retire spacing of back-to-back same-unit instructions gives the
**initiation interval**, whose excess over the element work is **dead
time**:

| unit | 8 x 32b | 4 x 64b | 2 x 128b |
|---|---|---|---|
| `vle32` startup / marginal thr | 47 cy / 3.0 e/cy | 47 cy / 2.9 e/cy | 47 cy / 2.9 e/cy |
| `vfmacc` startup / marginal thr | 38 cy / 4.6 e/cy | 37 cy / 4.4 e/cy | 37 cy / 4.4 e/cy |
| `vfredusum` initiation interval (64 e32) | 127 cy | 104 cy | 95 cy |
| -> of which dead time (work = 8 cy) | **119 cy** | **96 cy** | **87 cy** |
| `vmv.v.i` occupancy (128 e32) | 140 | 139 | **171** |

Three findings:

1. **Startup is large (~40-50 cy per instruction) and lane-invariant.**
   The steady-state pair period (550 cy at 4 x 64b) is almost entirely a
   stack of dependent startups/initiation intervals — init -> loads(47)
   -> last MAC(37) -> fold(60+60) -> reduce(104+104) -> slide(61) ->
   store(48) ~ 520 cy — while the throughput work in the same pair is
   ~110 cy.  This, not issue bandwidth, is the root of the 46.9% vector
   idle measured in §8: the chain is data-dependent, so neither more
   slots nor wider issue can compress it.  It also explains why the
   marginal load throughput (~3 e/cy) sits far below DLEN/4B = 8 e/cy:
   short chunks never amortize the fill.
2. **The reduction unit is ~92% dead time** (95-127 cy initiation for
   8 cy of element work), and its dead time *grows with lane count*
   (+32 cy from 2 to 8 lanes = the cross-lane combine/orchestration).
   That is the whole laneScale=1 regression, and it is why pipelining
   this one unit dwarfs every other knob measured in this document.
3. **The 2 x 128b outlier is `vmv.v.i` (+32 cy), a write-path term**:
   with two lanes the accumulator-init and store/slide writes share
   half as many per-lane write queues at twice the per-lane burst
   length.  Banks and RAM ports were falsified as the cause in §12.1;
   the queueing structure, thinned with the lane count, remains.

Design reading: chime, slots, banks and lanes all orbit two fixed
costs — a ~45-cycle per-instruction startup and a ~100-cycle reduce
initiation interval.  Cutting those two numbers (deeper decoupling of
the issue-to-lane path; a pipelined reduction tree) is worth more than
every configuration axis combined.

## 14. Solving startup and dead time — measured and proposed

### 14.1 Software (measured): remove the chain, not the links

If the pair period is a stack of dependent startups (§13), the winning
move is fewer, longer-lived instructions per dependence chain.  The
weights are already re-staged every launch, so the loader now stores
each matrix **transposed** (`-D T1_MV_T=1`, default) and the matvec
becomes reduction-free column accumulation: for each output block, one
`vmv.v.i`, then n chained `vle32` + `vfmacc.vf x[c]` pairs, one
`vse32`.  No per-row init, no fold ladder, no `vfredusum`, no slides,
no scalar result stores — per matvec the startup is paid once and the
dead-time-bound reduction unit is not used at all:

| kernel (RTL, stock config) | layer | head |
|---|---|---|
| best dot-product kernel (§9) | 817,844 | 8,310,745 |
| transposed, reduction-free | **341,281 (2.40x)** | **2,644,595 (3.14x)** |

Cumulative over the generic-SIMD baseline: **9.8x** (layer).  The
kernel is now genuinely streaming-bound; the remaining 2.6x to the
~130k-cycle memory floor is per-column load startup — attackable with
multi-column unrolling (more independent loads in flight) rather than
any reduction machinery.  Reductions survive only in attention/rmsnorm,
where they are O(heads + 2) per layer instead of O(rows).

### 14.2 A silent-corruption RTL bug, found by this kernel

The first transposed run produced wrong tokens on the RTL while pokedex
was exact.  Minimized: back-to-back `flw` + `vfmacc.vf` executes the
second MAC with the *previous* `fs1` value — the Rocket->T1 request
path captures the scalar operand without a RAW interlock against the
in-flight `flw` writeback (n=1 correct, n=2 corrupts, 4 filler
instructions fix it; register renaming does not).  Reported with the
minimal repro as xinpian-tech/T1#178.  The generator works around it by
software-pipelining the scalar loads one column ahead into alternating
f-registers.

### 14.3 Hardware proposals, ranked by measured leverage

1. **Pipelined (tree) reduction unit** — initiation 95-127 cy for 8 cy
   of work today.  Still first for workloads that cannot transpose
   (attention scores at long context, softmax, rmsnorm), and it would
   have made §4-§12 unnecessary.
2. **Fix #178** (correctness, not performance).
3. **Cut the ~47-cycle instruction startup**: a fast path for unmasked
   unit-stride ops that skips the mask-pipeline stages, plus cracking
   `vsetvli`+op pairs at dispatch, would shorten every dependence chain
   T1 ever runs; at the measured chain lengths this is worth ~20% even
   on the transposed kernel.
4. **Deeper per-lane write queues** only if wide lanes are ever chosen
   (§12.1); at 4 x 64b it is not the constraint.

## 15. The execution-cadence ceiling (falsified fixes, and the real one)

With the reduction-free kernel the remaining matvec cost is ~96 cycles
per `vfmacc.vf` against 36 cycles of element work (3.2 e/cy sustained,
40% of DLEN).  Three hypotheses were killed experimentally, each by a
config or kernel variant with cycle-level measurement:

| hypothesis | experiment | layer cycles | verdict |
|---|---|---|---|
| accumulator RAW chain | even/odd column striping into two independent accumulators | 342,954 | no effect |
| VRF port contention | `p0rp1w` two-port + 8 banks | 342,808 | no effect |
| shared float VFU | drop zvbb, `vfuInstantiateParameter=large` (per-slot float units) | 341,356 | no effect |

What remains is the lane's slot compaction machinery, read from the
RTL rather than the docs (`t1zaozi/src/laneStage/Lane.scala`):

1. Slots form a compaction queue toward slot 0; a hole appears exactly
   when slot 0's instruction completes (`slotCanShift(0) :=
   !slotOccupied(0)`, L319) and `slotShiftValid = scanLeftOr(~occupied)`
   marks every position at-or-above a hole (L327).
2. **A slot's element stream is frozen whenever a hole exists below
   it**: for index >= 1, `slotActive` includes
   `!slotShiftValid(index)` and gates the stage0 enqueue
   (`s.io.enqueue.valid := slotActive & ...`, L754-763).  Every
   completion therefore freezes every other in-flight arithmetic
   instruction while the queue shifts.
3. The stage0/1/2/3 pipelines and execution units are instantiated
   **per slot index** (L711/880/914/938/980); shifting copies the
   progress counters (`maskGroupCountVec`, `slotExecuteIndex`) into the
   next slot's state, so after each shift the instruction's element
   stream must refill a different pipeline instance from scratch.
4. An instruction entering at slot 3 pays this freeze+shift+refill up
   to three times before reaching slot 0 — which is why back-to-back
   vfmacc occupancy (~90 cy for 36 cy of work) approximately equals the
   initiation interval, independent of functional-unit count, VRF
   ports, or data dependences.  Loads are immune because the LSU
   executes them and the slot holds only bookkeeping — hence load issue
   gaps of 9 cycles next to MAC gaps of 96 in the same trace.  At m=288 the kernel issues
3,072 MACs x ~96 cy ~ 295k cycles of structural minimum; the measured
341,281 is 87% of that bound (the rest is attention/rmsnorm/rope).

Software has reached this machine's per-instruction cadence limit:
the cumulative kernel journey is 3,343,155 -> 341,281 cycles (9.8x) on
unchanged stock hardware.  The next factor belongs to RTL work, in
order: decouple the slot shift (or allow out-of-order slot retirement),
cut the ~45-cycle instruction startup, pipeline the reduction unit
(still needed for attention at long context).

## 16. Does the MAC chain?  The initiation-interval matrix

Microbenchmarks (pure instruction streams, cycles from launch deltas
between N=16 and N=48 repetitions, stock config) settle what chains and
what does not.  **The absolute values below are understated** — §18.1
shows streams this short do not saturate the machine, so read this
table for the *ratios* between shapes, and §18 for steady-state
numbers:

| stream (e32) | II per instruction | notes |
|---|---|---|
| independent `vfadd.vv` m1 | 5 | deeply pipelined |
| independent `vfadd.vv` m4 | 19 | 4 destination chains |
| independent `vfadd.vv` m8 (vl=512, work 64 cy) | **38** | cross-slot overlap works |
| independent `vfmacc.vf` m8 | **50** | MACs pipeline fine alone |
| **dependent** `vfadd.vv` chain m8, accumulator **v0** | **89-140** | not a chaining limit — the v0 penalty, see §17 |
| **dependent** `vfadd.vv` chain m8, accumulator v8 or v24 | **35** | arith->arith *does* chain element-wise |
| pure `vle32` m8 | 69 | ~2-deep, near beat rate |
| `vle32`+`vfmacc.vf` interleaved 1:1 | 110/pair | the kernel's shape |
| same instructions, 4+4 grouped | 58/pair | order matters in the micro... |
| 2+2 grouped (legal register schedule) | 104/pair | ...but not with real buffer counts |

Answers to "why don't the MACs chain":

1. **They do.**  Dependent load->MAC chaining works (that is how a MAC
   starts ~33 cy into its producer's stream), and independent MAC->MAC
   pipelining works (II 38-50 vs occupancy ~90).
2. Dependent arithmetic chains too — **as long as the destination is
   not v0** (II 35, indistinguishable from independent streams).  The
   "no arith->arith chaining" reading in an earlier revision of this
   table came from a microbenchmark that accumulated into v0; §17
   dissects the real mechanism.  What remains is (b): the lane
   pipelines never overlap a load with an arithmetic instruction, see
   §18.
3. At the layer's dimensions (m=288 -> vl=288, only 36 beats of data
   per load) both instructions of a column are **startup-dominated**
   (~47 cy fill each): two instructions at ~2-deep overlap give the
   invariant ~96 cy/column that every software restructure hit -
   single/dual accumulators, 1:1/2:2/4:4 ordering, m4/m8 blocking all
   land within 1% (341,281 / 342,954 / 344,095 / 342,808).  The same
   kernels at TinyLlama's dimensions (vl=512 blocks) sit much closer to
   beat rate.

This closes the loop with §13: for short-vector GEMV the ~47-cycle
per-instruction startup is not one bottleneck among several - it is the
only remaining one, and it is a hardware number.

## 17. The v0 penalty: never use the mask register as data

`v0` is architecturally an ordinary vector register that RVV *also*
designates as the mask operand.  On T1 that dual role has a large,
easily-tripped cost.  The discriminating experiment — one dependent
accumulate chain, three destination registers, everything else identical:

| dependent chain `vfadd.vv vX, vX, v16` (m8, vl=512) | II per instruction |
|---|---|
| vX = **v0** | **89-140** (~ occupancy: zero element overlap) |
| vX = v8 | **35** |
| vX = v24 | **35** |

Independent streams measure 37-38, so a dependent chain on a normal
register is already at the machine's streaming rate — chaining works.
The mechanism behind the v0 column is in `t1zaozi/src/T1.scala`:
`specialInstruction = decodeResult(Decoder.special) | requestReg.bits.vdIsV0`
(L345-346), and specials are admitted only into the **single last**
sequencer slot, and only when that slot is **idle** (L550, L883-884).
Back-to-back v0 writers therefore serialize at retirement cadence: the
consumer is not even dispatched to the lanes until the producer retires.

Consequences for software, applied to every kernel in
`T1/examples/t1llama.mojo`:

- No hand-written kernel writes `v0`.  Accumulators live in v8/v16,
  streaming buffers in v16/v24, reduction seeds and destinations in
  v24/v25, the dot-product weight buffer in v28.
- `vmv.v.i` is *not* free of this either: it encodes as `vmerge.vim`
  with `vm=1`, and in a load-interleaved stream a `vle32`+`vmv.v.i`
  pair costs **256 cy** against 112 for `vle32`+`vfadd.vv`.  Use it
  only outside inner loops (the transposed matvec zeroes an
  accumulator once per output strip, not per column).
- The one legitimate v0 use — a mask produced by a compare — remains:
  72 `vmflt.vf` out of 76,606 vector-register writes per token
  (0.09%), from compiler-generated code, not from the kernels.

Eliminating v0 from the kernels is **cycle-neutral on the layer**
(341,281 before and after): in the transposed matvec the v0 writes were
already hidden behind the load stream.  It removes a landmine — a
schedule change that shortens the loads, or a machine with more memory
bandwidth, would have made the penalty visible — and it makes the
kernels immune regardless of how the hardware resolves
[xinpian-tech/T1#180](https://github.com/xinpian-tech/T1/issues/180)
(fix proposed in PR #181: drop `vdIsV0` from `specialInstruction`,
which lifts the dependent-v0 chain from II 89.7 to 36.25).

## 18. The real wall: VRF port bandwidth, measured at the roofline

### 18.1 Every initiation interval above was measured too short

The II tables in §16 and the first version of this section used
straight-line streams of 16-48 instructions.  A stream-length sweep
shows that regime is **not saturated** — the whole stream hides under
the fixed ~150-cycle launch overhead:

```
instrs (m4 `vfadd.vv`, 256 elem each):   2    4    8   16   32    64
cycles:                                 150  152  152  152  160  1168
marginal cycles per instruction:          -  1.0  0.0  0.0  0.5  31.5
```

Re-measured in the saturated regime (marginal cycles between 48- and
144-instruction streams, e32/m4, vl=256, blastoise stock):

| stream | steady state | roofline | utilization |
|---|---|---|---|
| `vle32` only | **37.0 cy/load** | 1024 B / 32 B/cy = 32 | 87% of AXI |
| independent `vfadd.vv` | **33.2 cy/instr** | 256 / 8 per cy = 32 | 96% of DLEN |
| `vle32` + `vfadd.vv` | **56.0 cy/pair** | 32 (all three units) | 57% |
| `vle32` + `vfadd.vf` | **49.0 cy/pair** | 24 (VRF) / 32 (others) | 65% |
| `vle32` + `vfmacc.vf` (the GEMV shape) | **56.0 cy/pair** | 32 | 57% |

Each unit alone is healthy: loads reach 87% of AXI, arithmetic 96% of
the datapath.  The mix is at 57% of all three — and 56 sits between
fully serial (37 + 33 = 70) and fully overlapped (37), so about a fifth
of the available overlap is realized, not none.

### 18.2 The cost is VRF port accesses, and the design has no margin

VRF nominal bandwidth is `laneNumber × rfBankNum × ramWidth`
= 4 × 4 × 64 bit = **32 f32-accesses/cycle** (`rfBankNum = portFactor`,
`ramWidth = datapathWidth = laneScale × eLen = 64`, single-ported
`p0rw` banks).  Per element: a load costs 1 write, `vfadd.vv` and
`vfmacc.vf` cost 2 reads + 1 write, `vfadd.vf` costs 1 read + 1 write.
The measurements track that exactly:

- `vfmacc.vf` and `vfadd.vv` have identical VRF traffic and identical
  cost — 56.0 both, to the cycle.
- `vfadd.vf` drops one vector read per element and costs exactly 7
  cycles less; nominal capacity predicts 8.

And at full rate the sum has **zero headroom**: arithmetic at 8
elem/cycle needs 24 accesses/cycle, a load at 32 B/cycle needs the
remaining 8, so load + MAC at peak is exactly 32 of 32.  The mix cannot
overlap for free even in principle, and the 57% realized says another
~43% is lost to bank conflicts and arbitration on top of that.

This supersedes the in-flight-window story: the window
(`chainingSize + 1`, capped at 5 by a hardcoded two-bit tag —
[#183](https://github.com/xinpian-tech/T1/pull/183)) is real but not
what binds.  Steady-state arithmetic is at the datapath roofline
already at `chainingSize = 4`, and the mixed stream measures 56.0
cycles per pair identically at cs4, cs8, and cs8 + #183.

### 18.3 What would actually buy the GEMV a factor

The transposed matvec's inner loop is one column load plus
`vfmacc.vf v8, ft0, v16` — 4 VRF accesses per element (load write, two
reads, one write), the minimum a VRF-resident accumulator allows.  Two
hardware changes would move it, filed as suggestions on
[#182](https://github.com/xinpian-tech/T1/issues/182):

1. **Accumulator forwarding for MAC chains.** Consecutive MACs into the
   same `vd` could keep the accumulator group VFU-local, removing its
   read *and* write from the VRF: 4 accesses per element → 2, i.e. up
   to **2x** on the dominant LLM-decode kernel shape, with no extra
   bandwidth or banks.
2. **More bank ports** (`portFactor`, or the `p0rp1w` split measured in
   §7) so load writes stop competing with execution reads.

Software has no move left here: the kernel already issues the minimum
number of VRF accesses the ISA allows for a matvec, which is why every
schedule variant lands within 1% (§16).

## 19. Hitting the roofline: VRF ports, and the scalar tax

§18 left the mix at 57% of every roofline with the diagnosis "VRF port
bandwidth, provisioned with zero margin".  Both halves of that are now
tested — the hardware side by sweeping `portFactor`/`vrfRamType`, the
software side by fixing what turned out to be a larger loss than either.

### 19.1 The scalar tax, and how to pay less of it

The layer kernel ran at 111 cy/column while the same instruction pair
in a microbenchmark ran at 61.4.  Isolating the difference (vl=288, m8,
one column per iteration, stock config):

| shape | cy/column |
|---|---|
| load + MAC, address constant | 61.4 |
| load + MAC, streaming addresses | **61.4** (streaming is free) |
| load + MAC + one `flw` per column | **104.0** |
| same, `flw` from a constant address | 106.8 |
| load + MAC, 4 `flw` hoisted per 4 columns | 72.8 |
| load + MAC, 8 `flw` hoisted per 8 columns | **71.4** |
| 12 / 16 hoisted (tuned config) | 57.4 / 57.1 |
| next batch's `flw` interleaved between MACs | **98.6** |

The scalar load of `x[j]` for `vfmacc.vf` costs ~43 cycles of stalled
issue when it sits immediately before its consumer.  It is tempting to
read that as the RAW through `f[rs1]` being exposed.  It is not — the
dependency is free, and the cost is structural, on the scalar *memory*
path.  Same column stream, only the injected scalar instruction varies:

| injected instruction | RAW with the MAC? | cy/column |
|---|---|---|
| none, or `addi` | — | 61.4 |
| `fmv.w.x ft0` (writes the MAC's operand, no memory) | **yes** | **61.5** |
| `fadd.s ft0` (same, FP ALU) | **yes** | **61.5** |
| `lw` (integer load, result never read) | no | **104.6** |
| `flw ft9` (FP load, result never read) | no | **104.6** |
| `flw ft0` (FP load the MAC consumes) | yes | 104.6 |
| `sw` (a *store* — returns no data at all) | no | **104.6** |

A register producer feeding the vector instruction one cycle later is
free; any memory operation costs 43 cycles whether it is integer or
floating point, load or store, consumed or dead.  The RTL says why:

- The vector instruction's scalar operand never waits at T1.  `vfmacc.vf`
  decodes as `vectorReadFRs1`, Rocket's own FPU reads the FP regfile in
  EX, and WB enqueues instruction *and* operand into the 32-deep T1
  issue queue (`rocketvzaozi/src/Rocket.scala:2063-2073`).
- The whole data region is uncached: `--cacheable=1111...1`
  (`designs/blastoise.toml:41`) is a 32-bit exact-match bitpat, so only
  `0xFFFF_FFFF` is cacheable while the SRAM lives at `0x8000_0000`.
  Every scalar access is therefore a full AXI round trip returned as a
  replay (`rocketvzaozi/src/HellaCache.scala:1151`).
- `maxUncachedInFlight == 1` (`HellaCache.scala:823`): a second uncached
  access is nacked into `replayWb`/`takePcWb`, a full flush and refetch.
- Issue is in-order and single (`IBuf.scala:28`, ID dequeue gated by
  `ctrlStalld`), so that stall blocks every later instruction —
  including vector instructions that have nothing to do with it.  The
  vector unit drains, and the round trip is fully exposed.

Related but separate: `fpDataHazardEx/Mem/Wb` are gated by
`idDecodeOutput(d.fp)` (`Rocket.scala:1796-1801`), which is 0 for rv_v,
so the precise interlock does not cover a vector `frs1` read at all —
the correctness hole behind #178, fixed by #179.

So batching does not remove the scalar serialization; it moves it out
from between a load and its dependent MAC, so the ~43 cycles per access
run under the shadow of vector work already dispatched (up to five
instructions in flight).  Eight accesses at ~43 = ~344 cycles of scalar
time fit under eight columns at ~47 = ~376 cycles of vector time, and
what does not fit is the residual ~10 cy/column.  The same model
predicts the rest of the table: interleaving destroys the shadow
(98.6), deeper batching cannot help once the shadow is saturated (57.1
at sixteen), and back-to-back accesses are cheap after the first (+43
for one, then ~+8.7 each) because a replay costs far less than a round
trip once an earlier one is already in flight.  Layer 341,281 →
**254,311** (1.34x), head 2,644,595 → **2,105,093**.

The hardware fixes this implies, in order of value: make the device
window cacheable for the D$ (32-byte lines would amortize `x[j]` over
eight columns — but the same launch's vector stores feed those scalar
reads, so it needs a fence or a coherent D$ first); allow more than one
uncached access in flight; and land #179 so the precise interlock
covers vector `frs1` reads.

### 19.2 The `portFactor` sweep

`rfBankNum = rowWidth / ramWidth = portFactor`, so `portFactor` is
literally the number of single-ported VRF banks per lane, and
`vrfRamType` decides whether a bank port is shared between reads and
writes (`p0rw`) or split (`p0rp1w`).  Steady-state m4 streams:

| stream | pf4/p0rw (stock) | pf8 | pf4/p0rp1w | pf8+p0rp1w | pf16+p0rp1w |
|---|---|---|---|---|---|
| 2× `vle32` | 74.0 | 74.0 | 74.0 | 74.0 | 74.0 |
| 2× `vfadd.vv` | 66.3 | 66.0 | 65.9 | 66.4 | 65.0 |
| `vle32` + `vfadd.vv` | 56.0 | 43.2 | 49.0 | **43.2** | 43.2 |
| `vle32` + `vfmacc.vf` | 56.0 | 46.4 | 43.5 | **43.1** | 43.1 |
| llama layer kernel | 254,311 | 213,238 | 210,592 | **201,368** | 200,928 |
| llama head kernel | 2,105,093 | 1,682,274 | 1,676,202 | **1,454,575** | — |

Reading it:

- Pure streams are unmoved — they were never VRF-bound (AXI 88%,
  datapath 96%).
- The mix goes from 57% to **86%** of its roofline; the diagnosis in
  §18.2 was right, and doubling the banks is what fixes it.
- **`portFactor` saturates at 8**: 16 banks measure identically, so the
  residual 14% is not VRF bandwidth.
- The two knobs overlap: 8 banks or split ports each recover most of
  it, and together they recover slightly more than either.

### 19.3 Where the kernel now stands

At the kernel's own shape (vl=288, m8) on `pf8 + p0rp1w`:

| | cy/column | vs roofline |
|---|---|---|
| pure load stream (the roofline: 1152 B at 88% of AXI) | 41.0 | — |
| load + MAC, no scalars | 47.3 | **87%** |
| load + MAC + batched scalars | 58.1 | 71% |
| the real kernel (mixed vl across strips) | ~65 | — |

The vector half of the GEMV is within 13% of what the memory system
can deliver, and `portFactor` is no longer what stands between it and
the roofline.  What remains is the ~10 cy/column scalar tax, which is
structural in an in-order scalar core coupled to the vector unit: it
cannot be batched away (12/16-deep is flat) and must not be spread out
(interleaving costs 1.7x).

### 19.4 Cost scales with `vl`, not VLMAX

A partially-filled register group was suspected of paying full-width cost —
the layer's 111 cy/column at vl=288 is exactly 512 × 0.217, which looks like
an m8 instruction being charged for all 512 element slots.  It is not.
Steady-state marginals, independent `vfadd.vv`:

| shape | vl | cy/instruction | elem/cycle |
|---|---|---|---|
| m8 | 512 | 65.5 | 7.82 |
| m8 | 288 | 36.9 | 7.81 |
| m8 | 264 | 34.0 | 7.76 |
| m4 | 256 | 33.1 | 7.73 |
| m4 | 160 | 21.3 | 7.50 |
| m2 | 128 | 18.5 | 6.91 |

Efficiency is flat at ~7.8 elem/cycle (98% of DLEN) from vl=256 upward and
only softens below ~160, where per-instruction startup stops amortizing.  So
LMUL is free to follow the register budget for strip-mined streaming code —
the earlier VLEN result (§10) is about init/fold/reduce steps written at
VLMAX, not about strip-mined loads and MACs.  The GEMV pair behaves the same
way: 61.4 cy at vl=288 (m8) and 54.5 at vl=256 (m4), both 4.7 elem/cycle.

Layer 341,281 → **200,928** cycles overall: 1.34x from the software
change, 1.26x from the VRF configuration, **1.70x** together, with
token-exact output on both simulators throughout.

## 20. Reproducing

```sh
# run any workload on the RTL simulator with per-launch traces kept:
T1RT_KEEP_RTL_EVENTS=1 SIM=rtl ./run-t1llama.sh stories "" 1
# analyze a launch trace:
python3 T1/tools/analyze-rtl-trace.py run/rtl-event.0.jsonl
```

The trace analyzer (occupancy/gap tables, in-flight histogram,
scalar-overlap ratio) lives in `T1/tools/analyze-rtl-trace.py`; the
dynamic instruction-mix tool from the previous commit is
`T1/tools/analyze-rvv-mix.py`.
