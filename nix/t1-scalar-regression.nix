{ stdenv, verilator, python3, rtl, dwbb, applyCandidate ? false, expectFixed ? applyCandidate }:
stdenv.mkDerivation {
  pname = "t1-scalar-contention-${if applyCandidate then "candidate" else "baseline"}";
  version = "0.1.0";
  dontUnpack = true;
  nativeBuildInputs = [ verilator python3 ];
  buildPhase = ''
    cp ${rtl}/Rocket.sv .
    ${if applyCandidate then ''
      substituteInPlace Rocket.sv \
        --replace-fail 'assign mulDiv_resp_ready = ~_GEN_1436 & ~_GEN_857;' \
          'assign mulDiv_resp_ready = ~_GEN_1436 & ~_GEN_857 & ~_GEN_1447;' \
        --replace-fail '(~_GEN_1492_probe | ~_GEN_1435);' \
          '(~_GEN_1449 | ~_GEN_1435);'
    '' else ""}
    verilator --binary --timing -Wno-fatal -Wno-PINMISSING \
      --top-module scalar_contention -GEXPECT_FIXED=${if expectFixed then "1" else "0"} \
      -y . -y ${rtl} -y ${dwbb}/sim_ver +libext+.sv \
      Rocket.sv ${../T1/tests/rtl/scalar_contention.sv} -j 2
  '';
  installPhase = ''
    mkdir -p "$out"
    ./obj_dir/Vscalar_contention > "$out/result.log" 2>&1 || {
      cat "$out/result.log"
      exit 1
    }
    grep -q 'Scalar contention regression matched EXPECT_FIXED=' "$out/result.log"
    cat "$out/result.log"
  '';
}
