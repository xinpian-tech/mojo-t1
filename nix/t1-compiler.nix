{
  lib,
  llvmPackages_23,
  meson,
  ninja,
  mojo-dev,
  src,
}:
llvmPackages_23.stdenv.mkDerivation {
  pname = "t1-compiler-plugin";
  version = "0.1.0";
  inherit src;
  nativeBuildInputs = [ meson ninja llvmPackages_23.lld ];
  mesonFlags = [ "-Dmojo_dev=${mojo-dev}" "-Druntime=false" ];
  passthru = { inherit mojo-dev; };
  meta.platforms = [ "x86_64-linux" "aarch64-linux" ];
}
