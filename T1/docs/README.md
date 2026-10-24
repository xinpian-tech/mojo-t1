# T1 documentation

Use these documents for the current upstream-plugin/HAL implementation:

| Document | Scope |
| --- | --- |
| [Repository README](../../README.md) | Nix packages, Meson build and toolchains |
| [T1 README](../README.md) | Integration, supported threading, VME and recorded single-layer results |
| [Runtime README](../runtime/README.md) | HAL setup, launch ABI, simulator prerequisites and timing units |
| [Tuning guide](tuning-guide.md) | Performance comparison procedure and interpretation limits |

The former AsyncRT symbol inventory, handwritten ABI header and initial
bring-up plan are obsolete: T1 uses the upstream compiler-plugin and HAL
interfaces, not AsyncRT symbol interposition. HAL declarations come from
the matching `mojo-dev` package, not a copied header in this directory.

The old blastoise performance report and tuning rules describe a different
hardware configuration and kernel revisions. They are not current VME or
multi-hart baselines. Those documents remain available in Git history
(for example, commit `227cf9b`); they are no longer maintained here.

Recorded performance results are not rerun automatically by a package
build. Building examples and passing HAL ABI tests do not establish RTL
correctness, performance improvements or full-model inference acceptance.
