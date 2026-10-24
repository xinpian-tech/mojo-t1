# Building the Mojo release with Nix — from source, pure, offline

`flake.nix` (+ this directory) rebuilds the open-source Mojo release
(compiler, standard library, debugger, LSP server and their wheels) from the
locked `mojo-upstream` GitHub input: Bazel, clang, the sysroot and CPython
come from nixpkgs, and the whole build runs natively in the Nix
sandbox — no FHS environment, no patchelf, no nix-ld.

```text
nix build .#mojo            # Mojo compiler + stdlib + debugger/LSP
nix build .#mojo-wheels     # the release wheels of the Mojo family
nix build .#release         # wheels + PEP 503 index (offline pip install) + source tarball
nix run   .#mojo -- --version
nix build .#t1              # independent T1 extension; depends on unmodified Mojo
nix develop                 # optional development tools; package builds use nix build
nix flake check             # compiles, runs and debugs a Mojo program with the result
```

## What is built (and what is deliberately not)

| wheel                                          | contents                                                        | status                |
| ---------------------------------------------- | --------------------------------------------------------------- | --------------------- |
| `mojo_compiler-<v>-py3-none-linux_<arch>.whl`  | `bin/mojo`, `bin/lld`, crashpad handler, runtime `.so`, py pkg  | **built from source** |
| `mojo_compiler_mojo_libs-<v>-py3-none-any.whl` | `lib/mojo/std.mojoc` (standard library)                         | **built from source** |
| `mojo_lldb_libs-<v>-py3-none-linux_<arch>.whl` | `libMojoLLDB.so`, `libMojoJupyter.so`, `liblldb*.so`            | **built from source** |
| `mojo-<v>-py3-none-linux_<arch>.whl`           | `mojo-lldb`, `lldb-server`, `lldb-dap`, `mojo-lsp-server`, REPL | **built from source** ¹ |
| `mblack-<v>-py3-none-any.whl`                  | the Mojo formatter (pure python)                                | **built from source** |
| `max-*`, `max_core-*`, `max_mojo_libs-*`, …    | MAX engine/serving                                              | **not built** ²       |

¹ The package layout is limited to the targets listed in `sdk.nix`.
² The pinned October upstream includes MAX implementation sources, with
  portions under the MAX License. This flake's base release still packages
  Mojo, not the MAX serving stack. T1 uses the official compiler plugin and
  HAL interfaces; its MAX source overlay belongs to the separate T1 package.
  (The upstream *Mojo* wheels remain pinned in `release-wheels.json` solely
  for `nix build .#release-compare`, which diffs our wheels' file lists
  against upstream's.)

The wheels use a `linux_<arch>` platform tag instead of upstream's
`manylinux_2_34_<arch>`: the binaries are ordinary Nix binaries (interpreter
and rpath point into `/nix/store`), not manylinux ones.  Versions
(`mojo --version`, wheel names, `Requires-Dist`) come from
`bazel/mojo.MODULE.bazel` via `nix/versions.nix` — the release this checkout
pins through `mojo-upstream`.

## How it works

Modular's Bazel build normally downloads: a Bazel-at-commit binary (via the
BuildBuddy CLI), a prebuilt clang 22 + Ubuntu jammy sysroot, prebuilt
llvm-ifs and python-build-standalone. These toolchain downloads
of those is replaced by a source-built nixpkgs equivalent:

| downloaded by upstream                     | replaced by (all from nixpkgs source builds)                            |
| ------------------------------------------ | ----------------------------------------------------------------------- |
| Bazel (pinned commit binary)               | `pkgs.bazel_9` (9.1.1, bootstrapped from source by nixpkgs)             |
| `@clang-linux-*`                         | `llvmPackages_23` clang/lld/LLVM tools, laid out like the tarball, described by Modular's own `bazel/public-patches/clang.BUILD` |
| `sysroot-jammy-*` (Ubuntu sysroot tarball) | a sysroot assembled from nixpkgs: glibc (headers incl. kernel headers, crt, libc), gcc (libstdc++ headers/libs, libgcc, crtbegin), ncurses/libedit/libbsd/libmd/zlib for LLDB |
| `@llvm-ifs` (prebuilt)                     | `llvm-ifs`/`llvm-readtapi` from `llvmPackages_23`                       |
| python-build-standalone (rules_python)     | nixpkgs CPython (the default `python3`), one single version, in rules_python's hermetic-runtime layout |
| Mojo compiler                             | compiled from the locked upstream URL; no local Mojo sources are used |

The glue is `nix/bazelrc.nix`, an extra `--bazelrc` carrying the
`--override_repository`/`--override_module` lines, the Nix dynamic-linker and
store rpaths for everything the toolchain links, and a fixed PATH/shell for
actions.  `nix/toolchain-repos.nix` builds the replacement repositories.
Three small accommodations for the newer toolchain live there too:
glibc's linker scripts are rewritten to bare sonames (lld refuses absolute
paths inside a sysroot), crashpad gets `-include cstdint` (gcc 15 headers no
longer include it transitively), and protobuf's protoc-authenticity check is
downgraded (it needs a shell PATH the sandbox doesn't have; protoc itself is
the module's own, compiled from source).

Build pipeline (same shape as before):

```
nix/deps.nix   ──►  fixed-output derivation: `bazel build --nobuild` of the release
     │              targets with network; output = Bazel repository cache (source
     │              archives, module registry files, pure-python helper wheels of
     │              rules_pycross — no native binaries) + MODULE.bazel.lock.
     │              The LLVM tarball is fetched by Nix itself (nix/seeds.nix).
     │
nix/sdk.nix    ──►  the real build, offline (--repository_cache,
     │              --lockfile_mode=error, --repository_disable_download):
     │              11k actions — LLVM/MLIR/Clang/LLDB + the Mojo compiler —
     │              then `mojo` compiles the standard library, all natively in
     │              the Nix sandbox.  Installed stripped (llvm-strip), laid out
     │              like the wheels' `modular/` root.
     │
nix/mojo.nix   nix/wheels.nix   nix/release.nix
```

### Purity / reproducibility properties

* Mojo is built from upstream sources; nixpkgs toolchain dependencies can
  be substituted from the configured Nix binary caches. The repository
  cache of `modular-bazel-deps` contains source archives, BCR
  registry metadata and a handful of pure-Python wheels (rules_pycross's
  internal `pip`/`installer`/`packaging` helpers used at fetch time).
* Only fixed-output derivations touch the network (`modular-bazel-deps`, the
  LLVM source tarball, nixpkgs' own sources), all content-hash checked;
  everything rebuilds with `nix build --offline` afterwards.
* Bazel actions embed no timestamps (`--nostamp`, `__DATE__` redacted) and no
  absolute source paths (`-ffile-compilation-dir=.`); repeated builds of the
  SDK tree are expected to be bit-identical (verified for the previous
  toolchain generation; re-verify with `nix build --rebuild`).
* The produced binaries reference the Nix store (glibc 2.42, libstdc++ from
  gcc 15, ncurses…) via interpreter/rpath — they are real Nix binaries, which
  is why no patching step exists anywhere.

### Trade-off versus upstream binaries

This flake uses nixpkgs LLVM 23 and the nixpkgs sysroot. Its artifacts are
Nix-native, not portable manylinux wheels, and are not expected to be
bit-identical to Modular's published binaries.

### T1 package separation

#### T1 depends on `mojo-dev`

`nix build .#mojo-dev` selects `mojo.dev`, which exposes the SDK's `dev`
output and propagates the wrapped compiler. The upstream SDK build exports compiler headers, generated
headers, HAL headers and matching link libraries using Bazel's configured
`CcInfo`. `nix/mojo-dev/` supplies the export target in a private build
workspace; no upstream source files are modified for this interface.

The T1 compiler derivation consumes this output and compiles only
`T1Backend.cpp` and `T1Traits.cpp`, without running Bazel or rebuilding
LLVM/MLIR/Mojo. The HAL driver also consumes the exported HAL headers.
The development output requires a one-time SDK rebuild. Subsequent T1-only
edits do not change the SDK derivation. The development interface remains
revision-specific and must be used with the matching Mojo package.

`nix build .#mojo` contains no T1 code. `nix build .#t1` adds the compiler
plugin, HAL driver, target hooks and launch trampoline without modifying
the checkout or the installed Mojo package. See [T1/README.md](../T1/README.md)
for invocation and migration validation status. `t1-runtime` and `t1-overlay`
are independently buildable diagnostic outputs.

`rules-rs.nix` pins the Rust redistribution manifest checksum in the external
Bazel rule so offline analysis can reuse the dependency cache. It does not
patch Modular sources. Update that checksum when changing the Rust version.

### Updating

* New source revision: nothing to do unless the external dependency set
  changed — then `nix build .#modular-bazel-deps` prints the new hash for
  `depsHashes` in `nix/default.nix`.  If `bazel/public-patches/llvm_source.bzl`
  bumps LLVM, update `nix/seeds.nix`.
* nixpkgs bump: `nix flake update` (toolchain versions move with nixpkgs).
* New release version: `python3 nix/update-release-wheels.py` refreshes
  `release-wheels.json` (only used by `release-compare`).

## Layout of the results

```
result/                          (.#mojo-sdk-unwrapped, wheel `modular/` root layout)
├── bin/  mojo lld modular-crashpad-handler mojo-lldb lldb-server lldb-dap lldb-argdumper
│         llvm-symbolizer mojo-lsp-server
├── lib/  libKGENCompilerRTShared.so libAsyncRTRuntimeGlobals.so libMSupportGlobals.so
│         libMojoLLDB.so libMojoJupyter.so liblldb24.0.0git.so mojo-repl-entry-point
│         lldb-visualizers/  mojo/std.mojoc
└── python/mojo/*.py

result/                          (.#release)
├── wheels/*.whl                 built from source
├── simple/<project>/index.html  PEP 503 index:  pip install --index-url file://$PWD/result/simple mojo
├── source/modular-<v>.tar.gz    reproducible source archive
├── SHA256SUMS
└── MANIFEST.txt
```

## Requirements

* Linux x86_64 (aarch64-linux wired, untested); no special sandbox features
  (Bazel uses its process-wrapper sandbox inside the Nix sandbox).
* `big-parallel`: ~11k C++ actions; ≥ 16 GB RAM recommended (`--jobs` is
  bounded by RAM at ~1.5 GB/job), ~20 GB scratch, roughly an hour on 16 cores.
* Editing `nix/`, flake files or top-level Markdown does not invalidate the
  compiler build (filtered out of the Bazel source tree); other changes and
  new commits (the git revision lands in `mojo --version`) do.
