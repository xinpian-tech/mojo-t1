{
  lib,
  stdenvNoCC,
  stdenv,
  makeWrapper,
  llvmPackages_23,
  mojo,
  compiler,
  runtime,
  overlay,
  extension,
}:
stdenvNoCC.mkDerivation {
  pname = "mojo-t1";
  version = "0.1.0";
  dontUnpack = true;
  nativeBuildInputs = [ makeWrapper ];
  dontBuild = true;
  installPhase = ''
    runHook preInstall
    mkdir -p "$out/lib/mojo" "$out/share/t1" "$out/bin"
    cp -r ${overlay}/max "$out/lib/mojo/"
    ln -s ${compiler}/lib/libT1Compiler.so "$out/lib/libT1Compiler.so"
    ln -s ${runtime}/lib/libT1Driver.so "$out/lib/libT1Driver.so"
    cp -r ${extension}/examples "$out/share/t1/"
    ln -s ${runtime}/share/t1/trampoline.bin "$out/share/t1/trampoline.bin"
    makeWrapper ${mojo}/bin/.mojo-real "$out/bin/mojo-t1" \
      --unset MODULAR_MAX_PACKAGE_ROOT \
      --set MODULAR_MOJO_MAX_PACKAGE_ROOT "${mojo}" \
      --set MODULAR_MOJO_MAX_DRIVER_PATH "${mojo}/bin/.mojo-real" \
      --set-default MODULAR_CRASH_REPORTING_ENABLED 0 \
      --set-default MODULAR_TELEMETRY_ENABLED 0 \
      --prefix MODULAR_COMPILER_PLUGINS ';' "$out/lib/libT1Compiler.so" \
      --prefix MODULAR_DRIVER_PLUGINS ';' "T1@$out/lib/libT1Driver.so" \
      --set MODULAR_MOJO_MAX_IMPORT_PATH "$out/lib/mojo,${mojo}/lib/mojo" \
      --set-default T1RT_TRAMPOLINE "$out/share/t1/trampoline.bin" \
      --prefix PATH : "${lib.makeBinPath [ stdenv.cc llvmPackages_23.lld ]}"
    runHook postInstall
  '';
  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    export HOME="$TMPDIR"
    "$out/bin/mojo-t1" build --target-accelerator=t1 "$out/share/t1/examples/saxpy.mojo" -o saxpy
    install -m755 saxpy "$out/bin/t1-saxpy"
    "$out/bin/mojo-t1" build --target-accelerator=t1 "$out/share/t1/examples/t1llama.mojo" -o t1llama
    install -m755 t1llama "$out/bin/t1llama"
    "$out/bin/mojo-t1" build --target-accelerator=t1 -D T1_PREFILL_GEMM=0 "$out/share/t1/examples/t1llama.mojo" -o t1llama-prefill-serial
    install -m755 t1llama-prefill-serial "$out/bin/t1llama-prefill-serial"
    "$out/bin/mojo-t1" build --target-accelerator=t1 -D T1_THREADS=1 "$out/share/t1/examples/t1llama.mojo" -o t1llama-single
    install -m755 t1llama-single "$out/bin/t1llama-single"
    "$out/bin/mojo-t1" build --target-accelerator=t1 -D T1_VME=0 "$out/share/t1/examples/t1llama.mojo" -o t1llama-rvv
    install -m755 t1llama-rvv "$out/bin/t1llama-rvv"
    "$out/bin/mojo-t1" build --target-accelerator=t1 "$out/share/t1/examples/threaded.mojo" -o t1-threaded
    install -m755 t1-threaded "$out/bin/t1-threaded"
    "$out/bin/mojo-t1" build --target-accelerator=t1 "$out/share/t1/examples/vme.mojo" -o t1-vme
    install -m755 t1-vme "$out/bin/t1-vme"
    "$out/bin/mojo-t1" build --target-accelerator=t1 "$out/share/t1/examples/vme_gemm.mojo" -o t1-vme-gemm
    install -m755 t1-vme-gemm "$out/bin/t1-vme-gemm"
    "$out/bin/mojo-t1" build --target-accelerator=t1 -D T1_GEMM_ROWS=1 "$out/share/t1/examples/vme_gemm.mojo" -o t1-vme-gemm-serial
    install -m755 t1-vme-gemm-serial "$out/bin/t1-vme-gemm-serial"
    "$out/bin/mojo-t1" build --target-accelerator=t1 "$out/share/t1/examples/vme_gemm_quantized.mojo" -o t1-vme-gemm-quantized
    install -m755 t1-vme-gemm-quantized "$out/bin/t1-vme-gemm-quantized"
    "$out/bin/mojo-t1" build --target-accelerator=t1 -D T1_PREFILL_GEMM=0 "$out/share/t1/examples/vme_gemm_quantized.mojo" -o t1-vme-gemm-quantized-serial
    install -m755 t1-vme-gemm-quantized-serial "$out/bin/t1-vme-gemm-quantized-serial"
    "$out/bin/mojo-t1" build --target-accelerator=t1 -D T1_PREFILL_INTERLEAVE=1 "$out/share/t1/examples/vme_gemm_quantized.mojo" -o t1-vme-gemm-quantized-interleaved
    install -m755 t1-vme-gemm-quantized-interleaved "$out/bin/t1-vme-gemm-quantized-interleaved"
    "$out/bin/mojo-t1" build --target-accelerator=t1 "$out/share/t1/examples/prefill.mojo" -o t1-prefill
    install -m755 t1-prefill "$out/bin/t1-prefill"
    "$out/bin/mojo-t1" build --target-accelerator=t1 "$out/share/t1/examples/swiglu.mojo" -o t1-swiglu
    install -m755 t1-swiglu "$out/bin/t1-swiglu"
    "$out/bin/mojo-t1" build --target-accelerator=t1 "$out/share/t1/examples/scalar_vector_hazard.mojo" -o t1-scalar-vector-hazard
    install -m755 t1-scalar-vector-hazard "$out/bin/t1-scalar-vector-hazard"
    "$out/bin/mojo-t1" build --target-accelerator=t1 "$out/share/t1/examples/vme_quantized.mojo" -o t1-vme-quantized
    install -m755 t1-vme-quantized "$out/bin/t1-vme-quantized"
    "$out/bin/mojo-t1" build --target-accelerator=t1 "$out/share/t1/examples/rmsnorm.mojo" -o t1-rmsnorm
    install -m755 t1-rmsnorm "$out/bin/t1-rmsnorm"
    "$out/bin/mojo-t1" build --target-accelerator=t1 -D T1_EW_LMUL=1 "$out/share/t1/examples/rmsnorm.mojo" -o t1-rmsnorm-m1
    install -m755 t1-rmsnorm-m1 "$out/bin/t1-rmsnorm-m1"
    runHook postInstallCheck
  '';
  passthru = { inherit mojo compiler runtime; };
  meta = {
    description = "Out-of-tree T1 compiler plugin, HAL driver and Mojo target overlay";
    mainProgram = "mojo-t1";
    platforms = [ "x86_64-linux" "aarch64-linux" ];
  };
}
