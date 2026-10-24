{
  lib,
  llvmPackages_23,
  mojo-dev,
  src,
}:
llvmPackages_23.stdenv.mkDerivation {
  pname = "t1-compiler-plugin";
  version = "0.1.0";
  inherit src;
  nativeBuildInputs = [ llvmPackages_23.lld ];
  dontConfigure = true;
  buildPhase = ''
    runHook preBuild
    $CXX @${mojo-dev}/share/mojo-dev/compile.rsp -O3 \
      -c T1/compiler/T1Backend.cpp -o T1Backend.o
    $CXX @${mojo-dev}/share/mojo-dev/compile.rsp -O3 \
      -c T1/compiler/T1Traits.cpp -o T1Traits.o
    $CXX -shared -fuse-ld=lld -Wl,-z,defs -Wl,-soname,libT1Compiler.so \
      -Wl,-z,nodelete -Wl,--disable-new-dtags -Wl,--gc-sections \
      T1Backend.o T1Traits.o @${mojo-dev}/share/mojo-dev/link.rsp \
      -o libT1Compiler.so
    runHook postBuild
  '';
  installPhase = ''
    runHook preInstall
    install -Dm755 libT1Compiler.so "$out/lib/libT1Compiler.so"
    runHook postInstall
  '';
  passthru = { inherit mojo-dev; };
  meta.platforms = [ "x86_64-linux" "aarch64-linux" ];
}
