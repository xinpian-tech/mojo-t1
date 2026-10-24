{ stdenv, referenceSource }:
stdenv.mkDerivation {
  pname = "t1-quantized-reference";
  version = "0.1.0";
  dontUnpack = true;
  buildPhase = ''
    cp ${referenceSource}/run.c run.c
    substituteInPlace run.c --replace-fail \
      'void matmul(float* xout, float* x, float* w, int n, int d)' \
      'void fp32_matmul(float* xout, float* x, float* w, int n, int d)'
    $CC -O2 -fopenmp -ffp-contract=off \
      -include ${../T1/tools/quantized-matmul.c} run.c -lm -o t1-quantized-reference
  '';
  installPhase = ''
    mkdir -p "$out/bin"
    install -m755 t1-quantized-reference "$out/bin/"
  '';
}
