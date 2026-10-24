{
  lib,
  stdenvNoCC,
  cacert,
  bash,
  bazel,
  jdk_headless,
  nixBazelrc,
  src,
  bazelFlags,
  seeds ? [ ],
  hash ? lib.fakeHash,
}:
let


  seed = import ./seed.nix { inherit lib; };
in
stdenvNoCC.mkDerivation {
  name = "modular-bazel-deps";
  inherit src;

  outputHashMode = "recursive";
  outputHashAlgo = "sha256";
  outputHash = hash;


  impureEnvVars = lib.fetchers.proxyImpureEnvVars ++ [ "GIT_PROXY_COMMAND" "SOCKS_SERVER" ];

  nativeBuildInputs = [ bazel ];

  dontUnpack = true;
  dontConfigure = true;
  dontFixup = true;

  SSL_CERT_FILE = "${cacert}/etc/ssl/certs/ca-bundle.crt";
  NIX_SSL_CERT_FILE = "${cacert}/etc/ssl/certs/ca-bundle.crt";
  NIX_BAZELRC = nixBazelrc;
  NIX_BASH = bash;
  BAZEL_SERVER_JAVABASE = jdk_headless;

  buildPhase = ''
    runHook preBuild
    export repositoryCache="$TMPDIR/repocache"
    mkdir -p "$repositoryCache"
    ${seed.add "$repositoryCache" seeds}

    source ${./bazel-env.sh}
    run_bazel build --nobuild \
      --lockfile_mode=update \
      --loading_phase_threads=4 \
      --experimental_repository_downloader_retries=10 \
      ${lib.escapeShellArgs bazelFlags.build} \
      -- ${lib.escapeShellArgs bazelFlags.targets}
    stop_bazel
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p "$out"
    cp "$TMPDIR/workspace/MODULE.bazel.lock" "$out/MODULE.bazel.lock"
    cp -r "$repositoryCache" "$out/repocache"
    # The seeded artefacts are inputs of this derivation, not outputs; drop
    # them again so the FOD only carries what Bazel itself downloaded.
    ${seed.remove "$out/repocache" seeds}
    # Bazel 9 may still create the (empty) contents cache directory.
    rm -rf "$out/repocache/contents"
    # No stray temporary files may survive (they would break reproducibility).
    find "$out/repocache" -name '*.tmp' -delete
    chmod -R u+w,a-w "$out"
    runHook postInstall
  '';

  passthru = { inherit seeds bazelFlags; };
}
