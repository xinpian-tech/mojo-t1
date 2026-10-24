{
  lib,
  stdenvNoCC,
  python3,
  gnutar,
  gzip,
  src,
  mojoWheels,
  versions,
}:
stdenvNoCC.mkDerivation {
  pname = "modular-release";
  version = versions.mojo.full;

  dontUnpack = true;
  nativeBuildInputs = [
    python3
    gnutar
    gzip
  ];

  buildPhase = ''
    runHook preBuild
    mkdir -p "$out/wheels" "$out/source"

    # Reproducible source archive (like the GitHub release's "Source code").
    tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
      --transform 's,^\.,modular-${versions.max.full},' \
      -C ${src} -cf - . | gzip -n > "$out/source/modular-${versions.max.full}.tar.gz"
    echo "source             modular-${versions.max.full}.tar.gz" >> "$out/MANIFEST.txt"

    for w in ${mojoWheels}/*.whl; do
      cp "$w" "$out/wheels/"
      echo "built-from-source  $(basename "$w")" >> "$out/MANIFEST.txt"
    done
    chmod -R u+w "$out"

    # PEP 503 simple index.
    python3 - "$out" <<'PY'
    import hashlib, html, os, sys, re
    out = sys.argv[1]
    wheels = []
    for root, _, files in os.walk(os.path.join(out, "wheels")):
        for f in files:
            if f.endswith(".whl"):
                wheels.append(os.path.join(root, f))
    projects = {}
    for w in sorted(wheels):
        name = os.path.basename(w).split("-")[0]
        norm = re.sub(r"[-_.]+", "-", name).lower()
        digest = hashlib.sha256(open(w, "rb").read()).hexdigest()
        projects.setdefault(norm, []).append((os.path.basename(w), os.path.relpath(w, os.path.join(out, "simple", norm)), digest))
    simple = os.path.join(out, "simple")
    os.makedirs(simple, exist_ok=True)
    with open(os.path.join(simple, "index.html"), "w") as f:
        f.write("<!DOCTYPE html><html><body>\n")
        for p in sorted(projects):
            f.write(f'<a href="{p}/">{p}</a><br>\n')
        f.write("</body></html>\n")
    for p, files in projects.items():
        os.makedirs(os.path.join(simple, p), exist_ok=True)
        with open(os.path.join(simple, p, "index.html"), "w") as f:
            f.write(f"<!DOCTYPE html><html><body><h1>Links for {p}</h1>\n")
            for fn, rel, digest in files:
                f.write(f'<a href="{html.escape(rel)}#sha256={digest}">{html.escape(fn)}</a><br>\n')
            f.write("</body></html>\n")
    PY

    (cd "$out" && find wheels source -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
    sort -o "$out/MANIFEST.txt" "$out/MANIFEST.txt"
    runHook postBuild
  '';

  dontInstall = true;

  passthru = { inherit mojoWheels versions; };

  meta.description = "Mojo release ${versions.mojo.full}: wheels + offline pip index";
}
