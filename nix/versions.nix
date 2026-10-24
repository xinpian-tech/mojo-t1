{ lib, src }:
let
  mojoModule = builtins.readFile (src + "/bazel/mojo.MODULE.bazel");
  grab =
    name:
    let
      m = builtins.match ".*${name} = \"([^\"]+)\".*" mojoModule;
    in
    if m == null then throw "versions.nix: cannot find ${name} in bazel/mojo.MODULE.bazel" else builtins.head m;
  split =
    v:
    let
      m = builtins.match "([0-9]+\\.[0-9]+\\.[0-9]+)(.*)" v;
    in
    if m == null then
      throw "versions.nix: cannot parse version ${v}"
    else
      {
        full = v;
        base = builtins.elemAt m 0;
        label = builtins.elemAt m 1;
      };
  mojo = split (grab "MOJO_PACKAGE_VERSION");
  max = split (grab "MAX_PACKAGE_VERSION");
in
{
  inherit mojo max;


  rev = if src ? shortRev then src.shortRev else if src ? dirtyShortRev then src.dirtyShortRev else "deadbeef";

  bazelFlags = [
    "--//:mojo_base_version=${mojo.base}"
    "--//:mojo_version_label=${mojo.label}"
    "--//:max_base_version=${max.base}"
    "--//:max_version_label=${max.label}"
  ];
}
