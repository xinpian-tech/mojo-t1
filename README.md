# mojo-t1

Standalone Nix packaging for upstream Mojo and the T1 accelerator extension.
Mojo sources are fetched exclusively through the pinned `mojo-upstream`
GitHub URL in `flake.nix` and `flake.lock`. No Mojo checkout or fork is needed.

## Build and test

```sh
nix build .#mojo .#mojo-dev .#t1 --cores 4
nix build .#checks.x86_64-linux.mojo-run \
  .#checks.x86_64-linux.mojo-dev-export --cores 4
```

The T1 package's install checks compile its examples with the compiler
plugin. The HAL package checks the exported driver ABI. These checks do not
run a full model or RTL simulation. See [T1/README.md](T1/README.md) for
runtime setup, single-layer measurements and RTL regression tools.

## Layout

- `meson.build`, `meson.options`: local compiler plugin and runtime build.
- `flake.nix`, `flake.lock`: pinned upstream and Nix dependencies.
- `nix/`: upstream toolchain packaging, development SDK and T1 packages.
- `T1/`: compiler plugin, HAL driver, Mojo hooks, examples and tests.

`mojo` contains no T1 code. `mojo-dev` (also `mojo.dev`) exports matching
headers and libraries. The T1 compiler package compiles only its own two
C++ files against this development package. T1-only changes do not rebuild
the upstream compiler.

## Local build system

Meson builds the T1 compiler plugin, HAL driver, smoke-test executable and
RV32 trampoline/test artifacts. Nix invokes the same Meson targets with
the pinned toolchains; upstream Mojo continues to use its upstream Bazel
build, without changes to the SDK derivation.

For a local Meson build, provide Clang 23 with libstdc++ for the host C++
compiler, a C23 compiler, Meson, Ninja, Python 3, and LLVM's RISC-V tools:

```sh
meson setup build -Dmojo_dev=/path/to/mojo-dev --prefix=/path/to/install
meson compile -C build
meson test -C build --print-errorlogs
meson install -C build
```

Set `CXX=clang++` at setup when Clang is not your default C++ compiler.
Use `-Dcompiler=false` or `-Druntime=false` to build one package. The
`rv_clang`, `rv_lld` and `rv_objcopy` options select the RV32 tools; Nix
pins these to LLVM 22. HAL tests check symbols without an RTL simulator.

The initial toolchain build is expensive. Supported validation currently
targets `x86_64-linux`; the `aarch64-linux` dependency-cache hash still needs
to be established. See [nix/README.md](nix/README.md) for packaging details.
