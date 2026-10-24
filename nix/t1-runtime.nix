{
  lib,
  stdenv,
  llvmPackages_22,
  meson,
  ninja,
  python3,
  source,
  mojo-dev,
}:
stdenv.mkDerivation {
  pname = "t1-hal-driver";
  version = "0.1.0";
  src = lib.fileset.toSource {
    root = /. + builtins.unsafeDiscardStringContext source.outPath;
    fileset = lib.fileset.unions (map
      (path: /. + builtins.unsafeDiscardStringContext source.outPath + path)
      [ "/meson.build" "/meson.options" "/T1/runtime" ]);
  };
  nativeBuildInputs = [ meson ninja python3 ];
  mesonFlags = [
    "-Dmojo_dev=${mojo-dev}"
    "-Dcompiler=false"
    "-Drv_clang=${llvmPackages_22.clang-unwrapped}/bin/clang"
    "-Drv_lld=${llvmPackages_22.lld}/bin/ld.lld"
    "-Drv_objcopy=${llvmPackages_22.llvm}/bin/llvm-objcopy"
  ];
  doCheck = true;
  meta.platforms = [ "x86_64-linux" "aarch64-linux" ];
}
