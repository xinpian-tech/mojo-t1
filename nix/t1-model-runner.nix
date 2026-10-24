{ stdenv, makeWrapper, pokedexSource, model }:
stdenv.mkDerivation {
  pname = "t1-model-runner";
  version = "0.1.0";
  dontUnpack = true;
  nativeBuildInputs = [ makeWrapper ];
  buildPhase = ''
    $CC -O2 -Wall -Wextra -Wno-unused-parameter \
      -I${pokedexSource}/pokedex/simulator/include -I${../T1/runtime} \
      ${../T1/runtime/model_runner.c} -ldl -o t1-model-runner
  '';
  installPhase = ''
    mkdir -p "$out/bin"
    install -m755 t1-model-runner "$out/bin/"
    wrapProgram "$out/bin/t1-model-runner" \
      --set POKEDEX_MODEL_LIB "${model}/lib/libpokedex_model.so"
  '';
}
