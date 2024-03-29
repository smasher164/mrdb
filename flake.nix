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
        devShell = with pkgs;
          mkShell {
            nativeBuildInputs = # with pkgs;
              [
                wine64Packages.staging
                pkg-config
                (rust-bin.nightly."2023-10-25".default.override {
                  extensions = [ "rust-src" "miri" ];
                  targets = [
                    "x86_64-pc-windows-gnu"
                    "x86_64-unknown-freebsd"
                    "x86_64-apple-darwin"
                  ];
                })
              ];
          };
      });
}
