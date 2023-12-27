{
  description = "A flake that sets up the devShell with rust nightly";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    let supportedSystems = [ "aarch64-linux" "i686-linux" "x86_64-linux" ];
    in flake-utils.lib.eachSystem supportedSystems (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };
      in {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs;
            [
              (rust-bin.selectLatestNightlyWith (toolchain: toolchain.default))
            ];
        };
      });
}
