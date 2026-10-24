{ rtlSource, slots ? 4 }:
assert builtins.elem slots [ 4 8 16 ];
let
  flake = builtins.getFlake rtlSource;
  pkgs = flake.legacyPackages.x86_64-linux;
  scope = pkgs.t1.overrideScope (_: previous: {
    zaozi-config = args:
      let base = previous.zaozi-config args;
      in if slots == 4 then base else pkgs.runCommand
        "t1-pr213-control-slots-${toString slots}-config"
        { nativeBuildInputs = [ pkgs.jq ]; }
        ''
          mkdir -p "$out"
          jq --argjson slots ${toString slots} '
            if .t1Parameter.chainingSize != 4 then
              error("expected four-slot baseline")
            else .t1Parameter.chainingSize = $slots end
          ' ${base}/T1RocketTile.json > "$out/T1RocketTile.json"
        '';
  });
  artifacts = scope.vme32t-threaded.t1rocketemu;
in {
  config = artifacts.zaozi-config;
  simulator = artifacts.verilator-emu.override {
    extraVerilatorArgs = [ "--threads" "1" "-CFLAGS" "-O3" ];
  };
}
