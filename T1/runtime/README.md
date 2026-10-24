# T1 HAL driver

`libT1Driver.so` implements upstream's `SDK/include/HAL/M_driver_api.h`
(HAL 0.1.0). It does not export or interpose the old AsyncRT ABI. The host
loads it through `MODULAR_DRIVER_PLUGINS=T1@/path/libT1Driver.so`.

The driver composes the launch trampoline, kernel ELF, argument block and
buffer shadows into a bare-metal image. Synchronization runs the simulator
and reads back results. The address map remains in `t1_layout.h`.

## Build and smoke test

From the repository root:

```sh
nix build .#t1-runtime --out-link result-t1-runtime
export T1RT_EMULATOR=/path/to/pokedex
export T1RT_EMULATOR_KIND=pokedex
export T1RT_MEMDUMP=1
export T1RT_TRAMPOLINE=$PWD/result-t1-runtime/share/t1/trampoline.bin
result-t1-runtime/bin/t1-hal-smoke \
  result-t1-runtime/share/t1/test_saxpy.elf saxpy
```

This test drives the official HAL C ABI, including invalid memory ranges,
offset copies, bundle/function lifetime, launch validation, events, buffer
readback and SAXPY results. The Meson `hal-abi` test, also run by the Nix
build, checks every required HAL symbol and rejects an exported legacy
AsyncRT entry point.

For RTL simulation, set `T1RT_EMULATOR` to the T1 Verilated simulator and
unset `T1RT_EMULATOR_KIND`. Fast memory readback requires the simulator
changes in `emulator-memdump.patch`; otherwise unset `T1RT_MEMDUMP` to use
PERF MMIO readback. The emulator is supplied separately.

## Launch ABI and timing

- RV32, Zve32f, per-thread VLEN 1024, soft-float `ilp32` calling convention.
- One block with 1–4 independent harts; other grids and block shapes are rejected.
- `T1RT_THREAD_COUNT` must match the simulator configuration. The default is
  one; use four for the `vme32t-threaded` RTL configuration.
- Official `thread_idx`, `block_dim` and block-barrier hooks are implemented.
  This is not a SIMT warp: warp collectives and dynamic shared memory are
  not supported. Shared device buffers carry inter-hart data.
- At most eight arguments, each occupying a 4- or 8-byte host slot. The
  trampoline consumes the low 32 bits. Arbitrary 64-bit values and aggregate
  argument layouts are not supported.
- Upstream HAL `DeviceContext.execution_time` measures **host nanoseconds**,
  including simulator launch and synchronization, not device cycles.
- The T1-specific queue property `t1.simulation_count` reports the latest
  launch's RTL cycles, or retired instructions under pokedex.

`T1RT_VERBOSE=1` logs launches. `T1RT_WORKDIR` chooses a scratch directory;
otherwise each context gets a separate temporary directory.
`T1RT_DISABLE_RTL_EVENTS=1` disables the per-instruction RTL trace;
use this for full-model inference to avoid multi-gigabyte trace files. It
does not disable PERF markers, result readback or correctness comparisons.
`T1RT_DUMP_MAX_BYTES` excludes larger buffers from readback: keep it above
every writable buffer's size. `T1RT_PC_HISTOGRAM` enables the patched
pokedex instruction histogram consumed by `../tools/analyze-rvv-mix.py`.

See `../README.md` for the separate Mojo and T1 packages. The documents in
`../docs/` describing AsyncRT interposition are historical design records,
not instructions for the HAL implementation.
