#!/usr/bin/env bash
# Copyright (c) 2026, Xinpian Tech. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# Builds libT1RT.so, the launch trampoline and the pure-C smoke test.
#
#   CC        host C compiler                      (default: cc)
#   RV_CLANG  clang able to target riscv32         (default: clang)
#   RV_LLD    ld.lld for the trampoline link       (default: ld.lld)
#   OBJCOPY   llvm-objcopy                          (default: llvm-objcopy)
#   OUT       output directory                      (default: ./out)
set -euo pipefail
cd "$(dirname "$0")"
CC=${CC:-cc}
RV_CLANG=${RV_CLANG:-clang}
RV_LLD=${RV_LLD:-ld.lld}
OBJCOPY=${OBJCOPY:-llvm-objcopy}
OUT=${OUT:-./out}
mkdir -p "$OUT"

"$CC" -std=gnu23 -shared -fPIC -fvisibility=hidden -O2 -Wall -I../../SDK/include \
  -o "$OUT/libT1Driver.so" driver.c

"$RV_CLANG" --target=riscv32-unknown-elf -march=rv32imafc_zve32f_zvl2048b \
  -mabi=ilp32 -c trampoline.S -o "$OUT/trampoline.o"
"$RV_LLD" -m elf32lriscv -nostdlib -static -n -Ttext=0x80000000 \
  --entry=_start "$OUT/trampoline.o" -o "$OUT/trampoline.elf"
"$OBJCOPY" -O binary "$OUT/trampoline.elf" "$OUT/trampoline.bin"

"$CC" -O1 -I../../SDK/include -o "$OUT/hal_smoke" hal_smoke.c \
  "$OUT/libT1Driver.so" -Wl,-rpath,"$(realpath "$OUT")"

# Regenerate the mem* builtins object embedded in the compiler backend
# (KGEN/.../T1/t1_builtins.inc) whenever t1_builtins.S changes.
"$RV_CLANG" --target=riscv32-unknown-elf -march=rv32imafc_zve32f_zvl2048b   -mabi=ilp32 -c t1_builtins.S -o "$OUT/t1_builtins.o"

echo "built: $OUT/libT1Driver.so $OUT/trampoline.bin $OUT/hal_smoke $OUT/t1_builtins.o"
