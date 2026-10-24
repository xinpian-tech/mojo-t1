{
  lib,
  stdenvNoCC,
  bash,
  bazel,
  jdk_headless,
  nixBazelrc,
  toolchainRepos,
  src,
  deps,
  bazelFlags,
  seeds,
  versions,
  python3,
}:
let
  seed = import ./seed.nix { inherit lib; };
  inherit (stdenvNoCC.hostPlatform) system;



  layout = [
    { target = "//Mojo/tools/mojo:mojo-full"; dest = "bin"; rename = { "mojo-full" = "mojo"; }; }
    { target = "@llvm-project//lld:lld"; dest = "bin"; }
    { target = "@crashpad//:modular-crashpad-handler"; dest = "bin"; }
    { target = "//Mojo:mojo-lldb"; dest = "bin"; rename = { "lldb" = "mojo-lldb"; }; }
    { target = "//Mojo:gdb-server"; dest = "bin"; }
    { target = "@llvm-project//lldb:lldb-dap"; dest = "bin"; }
    { target = "@llvm-project//lldb:lldb-argdumper"; dest = "bin"; }
    { target = "@llvm-project//llvm:llvm-symbolizer"; dest = "bin"; }
    { target = "//Mojo/tools/mojo-lsp-server"; dest = "bin"; }
    { target = "//Mojo:CompilerRT"; dest = "lib"; }
    { target = "//AsyncRT:RuntimeGlobals"; dest = "lib"; }
    { target = "//Support:Globals"; dest = "lib"; }
    { target = "//Mojo:MojoLLDB"; dest = "lib"; }
    { target = "//Mojo:MojoJupyter"; dest = "lib"; }
    { target = "@llvm-project//lldb:lldb24.0.0git"; dest = "lib"; }
    { target = "//Mojo/tools/mojo-repl-entry-point"; dest = "lib"; }
    { target = "//Mojo/stdlib/std"; dest = "lib/mojo"; }
  ];
  layoutTable = lib.concatMapStrings (
    e:
    let
      renames = lib.concatStringsSep "," (lib.mapAttrsToList (k: v: "${k}=${v}") (e.rename or { }));
    in
    "${e.target}\t${e.dest}\t${renames}\n"
  ) layout;
in
stdenvNoCC.mkDerivation {
  pname = "mojo-sdk-unwrapped";
  version = versions.mojo.full;
  inherit src;
  outputs = [ "out" "dev" ];

  nativeBuildInputs = [ bazel ];
  requiredSystemFeatures = [ "big-parallel" ];

  dontUnpack = true;
  dontConfigure = true;
  dontStrip = true;
  dontPatchELF = true;
  dontPatchShebangs = true;

  passAsFile = [ "layoutTable" ];
  inherit layoutTable;
  NIX_BAZELRC = nixBazelrc;
  NIX_BASH = bash;
  BAZEL_SERVER_JAVABASE = jdk_headless;

  buildPhase = ''
    runHook preBuild
    export repositoryCache="$TMPDIR/repocache"
    cp -r ${deps}/repocache "$repositoryCache"
    chmod -R u+w "$repositoryCache"
    ${seed.add "$repositoryCache" seeds}

    source ${./bazel-env.sh}
    cp ${deps}/MODULE.bazel.lock MODULE.bazel.lock
    cp -r ${./mojo-dev} nix_mojo_dev

    # Bazel's memory estimate per C++ compile is far below what LLVM/MLIR/KGEN
    # translation units really need (1-2 GB with -O3 -g), so bound the job count
    # by RAM as well as by cores: ~1.5 GB per job.
    jobs="$NIX_BUILD_CORES"
    mem_kb="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
    max_by_mem=$(( mem_kb / (1536 * 1024) ))
    (( max_by_mem < 1 )) && max_by_mem=1
    (( jobs > max_by_mem )) && jobs="$max_by_mem"
    echo "bazel: using --jobs=$jobs (cores=$NIX_BUILD_CORES, MemTotal=$((mem_kb / 1024)) MB)"

    build_flags=(
      --lockfile_mode=error
      --repository_disable_download
      --jobs="$jobs"
      --local_resources=cpu="$NIX_BUILD_CORES"
      # linux-sandbox needs /sys and user namespaces, which the Nix sandbox
      # does not provide; the process-wrapper sandbox (symlink forest, no
      # namespaces) is sufficient inside an already hermetic build.
      --spawn_strategy=processwrapper-sandbox,local
      ${lib.escapeShellArgs bazelFlags.build}
    )

    run_bazel build "''${build_flags[@]}" -- ${lib.escapeShellArgs bazelFlags.targets} //nix_mojo_dev:compiler

    # ---- assemble the SDK tree -------------------------------------------
    execroot="$(run_bazel info "''${build_flags[@]}" execution_root)"
    output_base="$(run_bazel info "''${build_flags[@]}" output_base)"
    llvm_strip="${toolchainRepos.clangRepo}/bin/llvm-strip"

    while IFS=$'\t' read -r target dest renames; do
      [[ -n "$target" ]] || continue
      mkdir -p "$out/$dest"
      # config(..., target): only the target configuration (some libraries are
      # also built in the exec configuration as tool dependencies).
      run_bazel cquery "''${build_flags[@]}" --output=files "config($target, target)" | while read -r rel; do
        f="$execroot/$rel"
        name="$(basename "$f")"
        IFS=',' read -ra pairs <<< "$renames"
        for p in "''${pairs[@]}"; do
          [[ "''${p%%=*}" == "$name" ]] && name="''${p#*=}"
        done
        echo "install $target -> $dest/$name"
        install -m 644 "$f" "$out/$dest/$name"
        if [[ -x "$f" ]]; then chmod 755 "$out/$dest/$name"; fi
        # ELF binaries and shared objects ship stripped, like the release.
        if head -c4 "$f" | grep -q "ELF"; then
          "$llvm_strip" --strip-all "$out/$dest/$name"
        fi
      done
    done < "$layoutTablePath"

    # LLDB data formatters shipped in lib/lldb-visualizers.
    llvm_src="$output_base/external/+llvm_source+llvm-raw"
    install -Dm644 "$llvm_src/llvm/utils/lldbDataFormatters.py" "$out/lib/lldb-visualizers/lldbDataFormatters.py"
    install -Dm644 "$llvm_src/mlir/utils/lldb-scripts/mlirDataFormatters.py" "$out/lib/lldb-visualizers/mlirDataFormatters.py"

    manifest="$(run_bazel cquery "''${build_flags[@]}" --output=files \
      'config(//nix_mojo_dev:compiler, target)' | grep '/compiler.json$')"
    ${python3}/bin/python3 ${./mojo-dev/install.py} "$execroot" "$execroot/$manifest" "$dev" "$llvm_strip"
    cp -r "$src/SDK/include/HAL" "$dev/include/HAL"

    stop_bazel
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    # Python package sources of the `mojo-compiler` wheel (import mojo).
    mkdir -p "$out/python"
    cp -r "$src/Mojo/python/mojo" "$out/python/mojo"
    chmod -R u+w "$out/python"
    find "$out/python" -name BUILD.bazel -delete
    runHook postInstall
  '';

  passthru = { inherit deps versions; };

  meta = {
    description = "Mojo compiler, standard library and tools built from source (raw SDK tree)";
    homepage = "https://www.modular.com/mojo";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" "aarch64-linux" ];
  };
}
