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

- `flake.nix`, `flake.lock`: pinned upstream and Nix dependencies.
- `nix/`: upstream toolchain packaging, development SDK and T1 packages.
- `T1/`: compiler plugin, HAL driver, Mojo hooks, examples and tests.

`mojo` contains no T1 code. `mojo-dev` (also `mojo.dev`) exports matching
headers and libraries. The T1 compiler package compiles only its own two
C++ files against this development package. T1-only changes do not rebuild
the upstream compiler.

The initial toolchain build is expensive. Supported validation currently
targets `x86_64-linux`; the `aarch64-linux` dependency-cache hash still needs
to be established. See [nix/README.md](nix/README.md) for packaging details.
