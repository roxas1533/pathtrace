{
  description = "A Nix-based development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
      # 必要なライブラリのパスを生成
      libPath = pkgs.lib.makeLibraryPath [
        pkgs.wayland
        pkgs.libxkbcommon
        pkgs.libGL
        # その他、アプリケーションが必要とするライブラリがあれば追加
        # pkgs.vulkan-loader
      ];
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          # Wayland関連のパッケージ
          wayland
          libxkbcommon
          pkg-config
          vulkan-loader

        ];

        # 環境変数を設定
        LD_LIBRARY_PATH = libPath;

      };
    };
}
