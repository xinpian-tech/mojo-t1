{
  description = "Standalone T1 extension and Nix packaging for pinned upstream Mojo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    mojo-upstream = {
      url = "github:modular/modular/da0db31d422e19ad5952d8d106b6da1d6899cf0a";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      mojo-upstream,
    }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        modular = pkgs.callPackage ./nix {
          inherit self;
          upstream = mojo-upstream;
        };
      in
      {
        inherit (modular) packages devShells checks;
      }
    )
    // {
      overlays.default = final: prev: {
        modular = final.callPackage ./nix {
          inherit self;
          upstream = mojo-upstream;
        };
        mojo = final.modular.packages.mojo;
      };
    };
}
