# Evaluating T1 kernel performance

## Scope

The current integration supports one block with one to four harts. The
threaded configuration described in [the T1 README](../README.md) has
1024-bit per-thread VLEN and a VME shared across harts. Do not transfer
throughput, latency or VRF-port limits from the older blastoise setup
without measuring the actual configuration.

The maintained performance baseline is the TinyLlama first-layer RVV/VME
comparison in the T1 README. It compares FP32 RVV with group-32 quantized
VME, not equal-precision kernels. It is not full-inference acceptance.

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

## Timing and bottlenecks

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
