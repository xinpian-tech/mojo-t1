{ lib, fetchurl }:
let

  llvmCommit = "6941494e2ae550d753728ba13454311b344d82cf";
  llvmSha256 = "fd9814a9c57f7340cd3b4b2c60777214eb2d193458e434afc131de2c96a61a2c";
in
[
  rec {
    name = "llvm-project-${llvmCommit}.tar.gz";
    urls = [ "https://github.com/llvm/llvm-project/archive/${llvmCommit}.tar.gz" ];
    sha256 = llvmSha256;
    src = fetchurl {
      inherit name urls sha256;
    };
  }
]
