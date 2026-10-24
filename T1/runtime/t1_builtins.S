/*===--------------------------------------------------------------------===*\
 * Copyright (c) 2026, Xinpian Tech. All rights reserved.
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions:
 * https://llvm.org/LICENSE.txt
\*===--------------------------------------------------------------------===*/
/* Freestanding mem* for T1 kernels (-nostdlib): LLVM synthesizes calls to
 * these for aggregate zero-init / copy loops.  Byte loops are fine — bulk
 * data movement in kernels is vectorized long before it gets here.
 * Assembled once and embedded in T1Backend.cpp (see t1_builtins.inc). */

  .text
  .globl memset
memset:                       /* (dst, c, n) -> dst */
  mv a3, a0
1: beqz a2, 2f
  sb a1, 0(a3)
  addi a3, a3, 1
  addi a2, a2, -1
  j 1b
2: ret

  .globl memcpy
memcpy:                       /* (dst, src, n) -> dst */
  mv a3, a0
1: beqz a2, 2f
  lbu a4, 0(a1)
  sb a4, 0(a3)
  addi a1, a1, 1
  addi a3, a3, 1
  addi a2, a2, -1
  j 1b
2: ret

  .globl memmove
memmove:                      /* (dst, src, n) -> dst; overlap-safe */
  bltu a1, a0, 3f
  /* src >= dst: forward copy */
  mv a3, a0
1: beqz a2, 2f
  lbu a4, 0(a1)
  sb a4, 0(a3)
  addi a1, a1, 1
  addi a3, a3, 1
  addi a2, a2, -1
  j 1b
2: ret
3: /* backward copy */
  add a3, a0, a2
  add a1, a1, a2
4: beqz a2, 5f
  addi a1, a1, -1
  addi a3, a3, -1
  lbu a4, 0(a1)
  sb a4, 0(a3)
  addi a2, a2, -1
  j 4b
5: ret

  .globl memcmp
memcmp:                       /* (a, b, n) -> int */
1: beqz a2, 2f
  lbu a3, 0(a0)
  lbu a4, 0(a1)
  bne a3, a4, 3f
  addi a0, a0, 1
  addi a1, a1, 1
  addi a2, a2, -1
  j 1b
2: li a0, 0
  ret
3: sub a0, a3, a4
  ret
