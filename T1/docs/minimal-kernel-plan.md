# T1 minimal-kernel bring-up plan — file-level changes and ordering

**Goal**: a Mojo program on an x86 host does `DeviceContext(api="t1")` → allocates buffers → compiles a
vector-add kernel to RVV code → loads it into the T1 instruction window → raises an interrupt to start it
→ synchronizes → reads results back → **obtains the simulation cycle count**.

**System shape (decided)**: host = x86 (compiler, scheduling and runtime all native x86); device = T1,
**RV32 scalar frontend + RVV (zve32f, FP32)**, bare metal, zero runtime.
The only RISC-V machine code in the whole chain is the T1 kernel itself; there is no RV64 and no
compiler running inside QEMU.

**Decisions already made**: Option B (implement the AsyncRT C ABI; T1 becomes a `DeviceContext`
backend); non-all-ones `grid_dim`/`block_dim` → REJECT; every simulation backend shares one runtime,
only the transport differs; the timing interface returns simulation cycles directly.

**Definition of the minimal kernel** (written against the current `Pointer` API; illustrative):

```mojo
def vadd(a: Pointer[Float32, MutAnyOrigin], b: Pointer[Float32, MutAnyOrigin],
         c: Pointer[Float32, MutAnyOrigin], n: Int32):
    comptime W = simd_width_of[DType.float32]()     # under the T1 target = VLEN / 32
    for i in range(0, Int(n), W):
        c.store(i, a.load[width=W](i) + b.load[width=W](i))
```

Only pointer and `Int32` arguments (not `Int`: 8 bytes on the host, 4 on the device — see C1), no heap
allocation, no `raises`, no print — this sidesteps the two big traps of C-ABI aggregate passing and a
device-side runtime.
An even smaller "milestone 0.5" drops the loop and requires `n == W`: one
`vsetvli / vle32.v / vadd.vv / vse32.v` group, used to validate the trampoline for the first time.

---

## 0. Dependency overview

```
 A compiler emits T1 code ─────┐
                               ├──► E end-to-end minimal kernel ──► F hardening
 B Mojo knows the T1 target ───┤
                               │
 C device-side contract (trampoline/transport) ─► D runtime libT1RT.so ─┘
```

- **Lines A, B and C are mutually independent and can proceed in parallel**; D depends on C; E depends on A+B+D.
- **A's artifact (the ELF) is C's first test input** — no need to wait for B/D.
- **The host is always x86**; only the kernel is RV32+RVV. Therefore no RISC-V host support is needed
  (no C ABI, no CompilerRT, no stdlib port — none of that is in scope); the compiler side only gains
  one **offload** target.

---

## A. Teach the compiler to emit T1 (RV32+RVV) machine code

> Precondition already satisfied: LLVM's RISCV backend is in the default build list
> (`bazel/public-patches/llvm_project.bzl:5-9`).
> `KGEN/lib/Target/` and `ObjectCompiler/Target/` only contain `Host`; the GPU backends are not in the
> open-source tree — **T1 will be the first offload backend in the open-source tree**.

### A1. New `KGEN/lib/Target/T1/T1Traits.h` / `T1Traits.cpp`

Model on `KGEN/lib/Target/Host/HostTraits.{h,cpp}` (~60 lines):

| Member | Value | Reason |
|---|---|---|
| `name()` | `"t1"` | |
| `matches(triple)` | `triple.isRISCV32()` (`riscv32-unknown-elf`) | only T1 uses RV32; no finer discrimination needed |
| `isBaseTarget()` | **`true`** | bypasses `requireMaxForAccelerator` (`TargetTraits.cpp:24`): non-base targets demand "install MAX" |
| `isGPU()` | `false` | only affects `overrideExported` (`SlicingUtils.cpp:158`, a tidiness optimization that un-exports non-entry functions; not blocking) |
| `emitsOffloadObjectFile()` | `true` (default) | |
| `getAsmExtension/LLVM/Object` | `.s / .ll / .o` | |
| `defaultCPU()` | `"generic-rv32"` | |
| static registration | `RegisterTargetTraits<T1Traits>` | the `:TargetTraits` target is `glob(lib/Target/**)` + `alwayslink=True` (`KGEN/BUILD.bazel:1098-1122`) → **no BUILD change needed** |

### A2. New `KGEN/lib/Compiler/ObjectCompiler/Target/T1/T1Backend.h` / `T1Backend.cpp`

Model on `ObjectCompiler/Target/Host/HostBackend.{h,cpp}` (~80 lines):

| Member | Approach |
|---|---|
| `traits()` | return the `T1Traits` singleton |
| `isBaseTarget()` | `true` (same reason as A1; `TargetBackend.cpp:136` has the same gate) |
| `isOffload()` | `true` |
| `emitAssembly()` | same as Host: `ctx.runLlc(module, buf, /*obj=*/false)` |
| `emitObject()` | `ctx.runLlc(..., /*obj=*/true)` to get a relocatable `.o`, then **invoke `ctx.linkerPath` (lld, already in the build: `KGEN/BUILD.bazel:1348`) yourself**: `-m elf32lriscv -static -nostdlib -Ttext=<instruction-window base>`, producing a **static executable ELF32** (do not use `--oformat=binary`; the runtime finds the entry by `functionName` in the symbol table). Do not reuse `ctx.linkObject` — it goes through `createSharedObject` and produces a `.so` |
| `createArchive()` | return not-wired like Host for now; the offline path does not need it |

`emitObject`'s output is exactly the byte stream in `CompiledFunctionInfo.asm` on the Mojo side
(`DeviceFunction._emission_kind` is `"object"` for non-NVIDIA targets, `device_context.mojo:2796`),
i.e. what the runtime's `loadFunction(data, dataLen)` receives.

### A3. Change `KGEN/BUILD.bazel`

Add `T1Backend.cpp/.h` to `:HostBackend`'s `srcs/hdrs` (`:1298-1328`; that target is
`alwayslink=True`, made for static registrars).
`:ObjectCompiler`'s glob would also scan `Target/**`, but object files no one references may be
dropped from a static library by the linker — putting them explicitly in the alwayslink target is the
safe move.

### A4. May need relaxing: `KGEN/lib/Target/TargetTraits.cpp:35` `requireMaxForAcceleratorRequest`

When `--target-accelerator` is non-empty and `isMaxInstalled()==false` this is a straight
`report_fatal_error`. `isMaxInstalled()` (`Support/lib/Configuration.cpp:663`) is always true inside
Bazel and also true when `max.package_root` is unset; it is only false when "package_root is set but
libmax.so is missing". Most layouts never hit it; if you do, make the fatal a pass for base targets.

**Verification point A**:
```
mojo build --emit=object --target-triple=riscv32-unknown-elf --target-cpu=generic-rv32 \
           --target-features=+m,+f,+v,+zve32f,+zvl4096b leaf.mojo
llvm-objdump -d leaf.o | grep -E 'vsetvli|vle32|vadd\.vv|vse32'
```
Pass if RVV instructions appear and a `<128 x float>` lowers to single LMUL≤8 group instructions
rather than being scalarized into a loop.
If no vector instructions show up: LLVM's fixed-vector→RVV lowering keys the minimum VLEN off
`zvl*b` (`RISCVSubtarget::getRealMinVLen`); if that does not take effect, append
`-riscv-v-vector-bits-min=<VLEN>` through `compile_options`.

---

## B. Teach Mojo the T1 accelerator target

### B1. Change `mojo/stdlib/std/gpu/host/info.mojo` (the only Mojo file that must change)

Follow the steps in `mojo/stdlib/docs/adding-gpu-targets.md`, five places:

1. **`_get_t1_target()`** (model on `_get_h100_target()`, `:843`):
   ```mojo
   `#kgen.target<triple = "riscv32-unknown-elf", stdlib_plugin = "default", `
   `arch = "generic-rv32", features = "+m,+f,+v,+zve32f,+zvl4096b", `     # ← fill +a/+c from the T1 scalar frontend's actual ISA subset; zve32f requires +f
   `tune_cpu = "generic-rv32", `
   `data_layout = "e-m:e-p:32:32-i64:64-n32-S128", `                     # ← standard RV32
   `index_bit_width = 32, simd_bit_width = 4096> : !kgen.target`         # ← device Int/pointers are 32-bit; simd_bit_width = VLEN, the carrier of SEW semantics
   ```
2. **`T1Family = AcceleratorArchitectureFamily(...)`** and **`T1 = GPUInfo.from_family(api="t1", ...)`**:
   `api="t1"` is the key — `DeviceContext.__init__`'s default `api` comes from it
   (`device_context.mojo:3887`).
   `warp_size/sm_count/shared_memory_*` are only consumed on `is_gpu()` branches; 1/0 is fine.
3. **`_all_targets`** gains `"t1"` (`:2056`).
4. **`_get_info_from_target`** gains `elif target_arch == "t1": return materialize[T1]()` (`:2107`).
   Note the function starts with a chain of **substring replacements** (`.replace("sm", "sm_")` and
   friends); pick a name that no rule rewrites; `"t1"` is safe.
5. **`GPUInfo.target()`** gains `if self.name == "T1": return _get_t1_target()` (`:1771`).

### B2. Optional: add `is_t1()` / `has_rvv()` to `mojo/stdlib/std/sys/info.mojo`

`is_triple["riscv32-unknown-elf"]()` and `_has_feature["v"]()` (`_has_feature` is the compile-time
`target_has_feature`, `:86-92`; a new target gets it automatically).
The minimal kernel does not need this; it matters once kernels grow specialization branches.

### B3. **No change needed** in `max/mojo/max/gpu/host/device_context.mojo`

- `DeviceContext.default_device_info = GPUInfo.from_name[_accelerator_arch()]()` (`:3814`) —
  `--target-accelerator=t1` wires up the api and target in one step;
- `compile_function` uses `default_device_info.target()` as the offload target (`:4258`);
- `enqueue_function`'s checked path uses `DefaultDeviceTypeEncoder` whenever `api() != "metal"`
  (`:3424`), encoding a `DeviceBuffer` as a raw device address — correct for T1.
  Precise residency tracking (a `T1DeviceTypeEncoder` alongside `MetalDeviceTypeEncoder`) waits
  until F.

**Verification point B** (still no runtime needed): in a host program print
`compile_info[vadd, target=get_gpu_target["t1"]()]().asm`, or
`ctx.compile_function[vadd, dump_asm=True]()`, and see RVV assembly.

---

## C. Device-side contract (parallel with A/B; owned by the T1 team)

### C1. `t1rt/device/mailbox.h` + `t1rt/device/trampoline.S`

- **Mailbox layout** (fixed addresses in the shared window): `entry_addr`, `argblock_addr`, `argc`,
  `status`, `cycles`.
- **Trampoline** (preloaded at the T1 reset/interrupt vector; assembly, zero dependencies):
  read the mailbox → load the first ≤8 arguments into `a0..a7` per ilp32 (see the "64→32 slot
  convention" below; beyond 8, degrade to passing the argblock pointer) → `jalr` to the entry
  → write `status=done` (the cycle count is taken on the testbench side, or written directly if the
  scalar frontend has `rdcycle/rdcycleh`) → `wfi`/halt.
- Depends on the instruction subset the T1 scalar frontend can execute (stack pointer or not,
  `jalr`/`ret`, CSRs, `+a/+c`) — this determines A1's `features` string and **is a parameter the T1
  team must supply**. The ABI is derived from the features (LLVM `computeDefaultABI`: `+f` without
  `+d` → `ilp32f`); the trampoline just has to agree with it.

**64→32 slot convention** (host is 64-bit, device 32-bit — unique to T1; GPUs never face this):
the `argSizes[]` that `enqueueFunctionDirect` receives are **host-side** sizes: a `DeviceBuffer`
encodes to an 8-byte pointer, a host `Int` is 8 bytes (the checked path goes through
`DefaultDeviceTypeEncoder`, whose `target()` is the host, `device_context.mojo:1386`). Milestone 1
convention:
- the runtime writes each argument into an **8-byte-aligned slot** at its host size;
- the trampoline `lw`s the **low 32 bits** of each slot into `a_i` — valid for pointers as long as
  the shared-window addresses stay below 4 GiB (or host↔device mapping is identity), and valid for
  integers as long as values stay below 2^31;
- the minimal kernel's scalar argument is an **`Int32`**, not an `Int`, eliminating the ambiguity at
  the source level.

The real fix lands in F: a `T1DeviceTypeEncoder` whose `target()` returns the T1 target, so
`size_of[..., target=]` naturally yields 4 bytes and an `Int` argument is rejected at compile time by
the encoder's size assertion — the correct behavior for a 32-bit device.

### C2. `t1rt/transport.h` + `t1rt/transport_t1emu.c`

Six functions: `open / close / read_mem / write_mem / raise_irq / poll_status / read_cycles`.
Pick the fastest-iterating backend first (t1emu in-process or a socket); verilator DPI, FPGA
(BAR/DMA) and Palladium (transactor) are just replacements of `transport_*.c` with zero change above.
Make the wait policy (block/poll/batch) a runtime setting (e.g. `T1RT_WAIT_POLICY`), not a
compile-time fork.

**Verification point C**: a pure-C `t1rt_smoke.c`, no Mojo: `write_mem` **A's ELF** into the
instruction window → write the mailbox → `raise_irq` → poll `status` → read the cycle count.
This is **the first time in the whole chain that T1 executes compiler output** and the most valuable
early integration point.

---

## D. Host runtime `libT1RT.so` (implements the AsyncRT C ABI)

### D1. `t1rt/abi.c` — per the delivered inventory

- **CORE 22 + COH 4 + DEGEN 11 = 37 symbols** plus the **3 timing symbols from PERF**
  (`_startTimer` / `_stopTimer` / `DeviceTimer_release`, promoted into the first milestone per your
  decision; `stopTimer` backfills the simulation cycle count directly).
- The remaining ~70 are all one-liners: `return "unsupported on T1";` (the ABI's error convention is
  a non-NULL `const char*`).
- Signatures follow `t1_asyncrt_abi.h` and the inventory; in `enqueueFunctionDirect`, non-all-ones
  grid/block → return an error string (REJECT).

Internal split:
| File | Responsibility |
|---|---|
| `t1rt/alloc.c` | allocator over the shared window (`createBuffer_async` / `createHostBuffer` / `createBuffer_owning` / `createSubBuffer` / release); tracks live buffer ranges (residency) |
| `t1rt/loader.c` | parse the static ELF from A2: `write_mem` each `PT_LOAD` to its `p_paddr`; find the entry in `.symtab` by `functionName` (**no sanitizing**: `get_linkage_name` has `sanitize=false` on non-GPU targets, so the ELF carries the raw mangled name) |
| `t1rt/launch.c` | pack `args[]/argSizes[]` into a contiguous argblock in the shared window → clean the residency ranges → write the mailbox → `raise_irq` → record the job; `synchronize` waits on status → invalidates the residency ranges; `stopTimer` reads `read_cycles` |
| `t1rt/coherence.c` | `HtoD_async` = clean, `DtoH_async` = invalidate (**COH semantics**, not memcpy — this makes the generic `map_to_host()` cheap without touching Mojo code) |

### D2. Link integration (no compiler changes)

`mojo build` automatically links `<package>/lib/libAsyncRTMojoBindings.so` when present —
`KGEN/lib/Support/Configuration.cpp:187-190`, whose comment reads: "The AsyncRT Mojo bindings ship
in max-core". Therefore:

- install layout: put `libT1RT.so` at that path (or install under that name); or set the
  `mojo.shared_libs` config key;
- inside Bazel: hang it off the host program as a `cc_library` dependency.

**Verification point D**: a pure-C test calling, in order, `AsyncRT_DeviceContext_create("t1",0)` →
`createBuffer_async` → `HtoD_async` → `loadFunction(A's ELF)` →
`enqueueFunctionDirect(1,1,1,1,1,1,0,…)` → `synchronize` → `DtoH_async` → `stopTimer`.

---

## E. End-to-end minimal kernel (host on x86, T1 in the simulator)

### E1. New `examples/t1/vadd.mojo`

```
ctx = DeviceContext()                                # api defaults to "t1" (from --target-accelerator)
a/b/c = ctx.enqueue_create_buffer[DType.float32](Int(n))   # n: Int32, the same value passed to the kernel
with a.map_to_host() as h: fill …                    # HtoD = clean (COH)
cycles = ctx.execution_time[...](
    lambda: ctx.enqueue_function[vadd](a, b, c, n, grid_dim=1, block_dim=1), 1)
ctx.synchronize()
with c.map_to_host() as h: verify …                  # DtoH = invalidate
print("cycles:", cycles)                              # on T1, execution_time is in cycles, not ns — document this
```

### E2. Build and run

```
mojo build --target-accelerator=t1 examples/t1/vadd.mojo -o vadd    # host = x86, kernel = T1
./vadd
```

**Verification point E (the milestone)**: correct results + printed simulation cycle count.

---

## F. Hardening after the kernel runs (off the minimal path)

| Item | Files |
|---|---|
| `EnsureT1Profile` validation pass: reject residual calls / stack allocation / external symbols / `raises` | new `KGEN/lib/Transforms/EnsureT1Profile.cpp` (model on `EnsureNoParameters.cpp`) + `KGEN/include/KGEN/KGENPasses.td` + hook in after `KGEN/lib/Compiler/Pipeline/Pipeline.cpp:256` |
| `T1DeviceTypeEncoder`: encode 32-bit pointers/integers against the T1 target (replaces the 64→32 slot convention) + precise residency tracking (flush only the buffers a launch uses) | `max/mojo/max/gpu/host/device_context.mojo:3156` add `elif api()=="t1"` + new `_device_context_t1.mojo` (model on `_device_context_metal.mojo`) |
| Device Graph (compress N host↔simulator round-trips into 1) | the 32 `DeviceGraphBuilder_*` in `t1rt/abi.c` |
| `block_dim` → multiple threads inside one T1 (MSP), `grid_dim` → multiple T1s on the bus | lift the REJECT in the runtime's `enqueueFunctionDirect`; the peer/multicast family |

---

## G. The first thing that breaks at each step (know it before you debug RTL)

| Stage | Most likely first failure | Where to look |
|---|---|---|
| A | static registrar dropped by the linker → "target 'riscv32-…' is not supported by this build" | A3 alwayslink; or the MAX gate at `TargetTraits.cpp:74` (A1 `isBaseTarget=true`) |
| A | scalar loop emitted instead of RVV | `zvl*b` not taking effect → `-riscv-v-vector-bits-min`; or `+v` missing from features |
| A/B | stdlib `comptime assert`s on a 32-bit target (`Int` width, `is_64bit()` assumptions) | 32-bit targets have support (`is_32bit()`, ILP32 handling in `std/ffi/__init__.mojo:154`) but the path is little-traveled; a kernel that only touches `SIMD`/`Pointer` has the smallest exposure |
| A/C | ABI mismatch: kernel expects float args in FP registers per ilp32f, trampoline loads per ilp32 | zve32f implies `+f` → default ilp32f; align the trampoline; the minimal kernel passing only pointers/integers sidesteps it anyway |
| B | the `comptime assert … in _all_targets` inside `_get_info_from_target` fails | name missing from `_all_targets`, or mangled by the replacement rules |
| D/E | `loadFunction` cannot find `functionName` | mangled names contain `::`/`(` and are raw bytes in the ELF — the loader must strcmp exactly, no normalization |
| D/E | pointers/integers truncated to wrong values | the 64→32 slot convention's precondition broke: window address ≥ 4 GiB, or the kernel used `Int` instead of `Int32` |
| E | results all zero / stale | missing clean (before launch) or invalidate (before readback) — COH semantics not wired |
| E | cycle count is 0 | `stopTimer` not wired to `read_cycles`, or the testbench does not count between IRQ and done |
