{ stdenv, verilator, python3, rtl, applyCandidate ? false, expectFixed ? applyCandidate }:
stdenv.mkDerivation {
  pname = "t1-srt-mode-${if applyCandidate then "candidate" else "baseline"}";
  version = "0.1.0";
  dontUnpack = true;
  nativeBuildInputs = [ verilator python3 ];
  buildPhase = ''
    cp ${rtl}/SRTFPWrapper.sv .
    ${if applyCandidate then ''
      substituteInPlace SRTFPWrapper.sv \
        --replace-fail 'opSqrtReg | input_bits_opSqrt & inputFire' \
          '(inputFire ? input_bits_opSqrt : opSqrtReg)' \
        --replace-fail 'opFloatReg | input_bits_opFloat & inputFire' \
          '(inputFire ? input_bits_opFloat : opFloatReg)'
    '' else ""}
    verilator --binary --timing -Wno-fatal --top-module srt_mode \
      -y . -y ${rtl} +libext+.sv SRTFPWrapper.sv ${../T1/tests/rtl/srt_mode.sv} \
      -j 2
  '';
  installPhase = ''
    mkdir -p "$out"
    ${if expectFixed then ''
      ./obj_dir/Vsrt_mode > "$out/result.log" 2>&1 || {
        cat "$out/result.log"
        exit 1
      }
      grep -q 'SRT mode-switch regression passed' "$out/result.log"
    '' else ''
      if ./obj_dir/Vsrt_mode > "$out/result.log" 2>&1; then
        echo "Expected the unmodified RTL to reproduce the mode-switch failure"
        exit 1
      fi
      grep -q 'MISMATCH integer after float got=09000000 expected=00000090' "$out/result.log"
    ''}
    cat "$out/result.log"
  '';
}
