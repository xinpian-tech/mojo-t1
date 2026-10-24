{ stdenvNoCC, fetchurl }:
stdenvNoCC.mkDerivation {
  pname = "rules-rs-offline";
  version = "0.1.0";
  src = fetchurl {
    url = "https://github.com/hermeticbuild/rules_rs/releases/download/v0.1.0/rules_rs-v0.1.0.tar.gz";
    hash = "sha256-/DX6AkzHILWsJWGEAzxiKd6tp86o791ioHSltSz5hec=";
  };
  dontConfigure = true;
  dontBuild = true;
  dontFixup = true;
  postPatch = ''
    substituteInPlace MODULE.bazel \
      --replace-fail 'name = "rules_rs",' 'name = "rules_rs", version = "0.1.0",'
    substituteInPlace rs/toolchains/module_extension.bzl \
      --replace-fail 'allow_fail = not locked_zstd,' \
      'allow_fail = not locked_zstd, sha256 = {"1.94.0": "00ced4bf4c91ea3e309d29a8c62a4b784044fd89bf25e4857b59ef34c7c22c00"}.get(base_version, ""),'
  '';
  installPhase = ''
    mkdir -p "$out"
    cp -r . "$out/"
  '';
}
