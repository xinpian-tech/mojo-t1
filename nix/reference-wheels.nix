{
  lib,
  stdenvNoCC,
  fetchurl,
  linkFarm,
}:
let
  json = builtins.fromJSON (builtins.readFile ./release-wheels.json);
  system = stdenvNoCC.hostPlatform.system;
  platformTag =
    {
      x86_64-linux = "manylinux_2_34_x86_64";
      aarch64-linux = "manylinux_2_34_aarch64";
      aarch64-darwin = "macosx_13_0_arm64";
    }
    .${system} or (throw "no upstream wheels for ${system}");


  filesFor =
    dist:
    lib.filterAttrs (fn: _: lib.hasSuffix "-any.whl" fn || lib.hasSuffix "-${platformTag}.whl" fn) (
      json.${dist} or { }
    );

  fetchWheel =
    fn: info:
    fetchurl {
      name = fn;
      url = info.url;
      sha256 = info.sha256;
    };
in
{
  inherit json platformTag filesFor fetchWheel;


  reference = linkFarm "modular-release-upstream-mojo" (
    lib.concatMap
      (
        dist: lib.mapAttrsToList (fn: info: { name = fn; path = fetchWheel fn info; }) (filesFor dist)
      )
      [
        "mojo"
        "mojo-compiler"
        "mojo-compiler-mojo-libs"
        "mojo-lldb-libs"
        "mblack"
      ]
  );
}
