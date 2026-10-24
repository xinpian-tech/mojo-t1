{
  lib,
  stdenv,
  llvmPackages_22,
  python3,
  source,
  mojo-dev,
  runCommand,
}:
let
  runtimeSource = lib.fileset.toSource {
    root = /. + builtins.unsafeDiscardStringContext source.outPath;
    fileset = /. + builtins.unsafeDiscardStringContext source.outPath + "/T1/runtime";
  };
in
stdenv.mkDerivation {
  pname = "t1-hal-driver";
  version = "0.1.0";
  src = runCommand "t1-runtime-source" { } ''
    mkdir -p "$out/T1" "$out/SDK/include"
    cp -r ${runtimeSource}/T1/runtime "$out/T1/runtime"
    cp -r ${mojo-dev}/include/HAL "$out/SDK/include/HAL"
  '';
  nativeBuildInputs = [ llvmPackages_22.clang-unwrapped llvmPackages_22.lld llvmPackages_22.llvm python3 ];
  dontConfigure = true;
  buildPhase = ''
    runHook preBuild
    CC="$CC" RV_CLANG=clang RV_LLD=ld.lld OBJCOPY=llvm-objcopy \
      bash T1/runtime/build.sh
    clang --target=riscv32-unknown-elf -march=rv32imafc_zve32f_zvl2048b \
      -mabi=ilp32 -O3 -ffreestanding -c T1/runtime/test_saxpy.c -o saxpy.o
    ld.lld -m elf32lriscv -static -nostdlib -n --image-base=0x80100000 \
      --entry=saxpy saxpy.o -o test_saxpy.elf
    runHook postBuild
  '';
  doCheck = true;
  checkPhase = ''
    runHook preCheck
    python3 - <<'PY'
    import ctypes
    import pathlib
    import re
    library = ctypes.CDLL(str(pathlib.Path("T1/runtime/out/libT1Driver.so").resolve()))
    symbols = pathlib.Path("SDK/include/HAL/M_driver_hal_symbols.inc").read_text()
    for name in re.findall(r"M_DRIVER_HAL_SYMBOL\((\w+)\)", symbols):
        getattr(library, "M_driver_" + name)
    try:
        library.AsyncRT_DeviceContext_create
    except AttributeError:
        pass
    else:
        raise AssertionError("legacy AsyncRT ABI must not be exported")
    PY
    runHook postCheck
  '';
  installPhase = ''
    runHook preInstall
    mkdir -p "$out/lib" "$out/bin" "$out/share/t1"
    cp T1/runtime/out/libT1Driver.so "$out/lib/"
    $CC -ISDK/include T1/runtime/hal_smoke.c -L"$out/lib" -lT1Driver \
      -Wl,-rpath,"$out/lib" -o "$out/bin/t1-hal-smoke"
    cp T1/runtime/out/trampoline.bin test_saxpy.elf "$out/share/t1/"
    runHook postInstall
  '';
  meta.platforms = [ "x86_64-linux" "aarch64-linux" ];
}
