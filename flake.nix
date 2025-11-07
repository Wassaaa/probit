{
  description = "Env for CPP and Python assignment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [

            cmake
            gcc
            gdb
            clang
            python313
            python313Packages.numpy
            python313Packages.scipy
          ];

          shellHook = ''
            echo "Environment loaded"
          '';
        };
      }
    );
}
