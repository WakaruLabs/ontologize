{
  description = "LaTeX toolchain for writeup/ and description/";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (s: f nixpkgs.legacyPackages.${s});
    in {
      devShells = forAll (pkgs:
        let
          tex = pkgs.texlive.combine {
            inherit (pkgs.texlive)
              scheme-medium
              algorithm2e ifoddpage relsize
              biblatex biber logreq
              bera plex
              newpx newtx kastrup fontaxes
              mdframed zref needspace
              titlesec
              units
              caption
              ;
          };
        in {
          default = pkgs.mkShell {
            packages = [ tex pkgs.gnumake ];
          };
        });
    };
}
