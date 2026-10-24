/*===--------------------------------------------------------------------===*\
 * Copyright (c) 2026, Xinpian Tech. All rights reserved.
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions:
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
\*===--------------------------------------------------------------------===*/

/*===--------------------------------------------------------------------===*\
 * trampoline.S — T1 bare-metal launch trampoline (RV32IMAFC, ilp32).
 *
 * Linked at the reset target (T1_TRAMPOLINE_BASE). Reads the argblock
 * written by libT1RT, calls the kernel with up to 8 register arguments,
 * streams the requested result ranges to the PERF MMIO and exits through
 * HTIF. Zero dependencies; the kernel is compiled with the soft-float
 * (ilp32) calling convention, so every argument slot goes to a GPR.
 *
 * SYNC: constants mirror t1_layout.h (the assembler cannot parse the
 * C header's `u` suffixes).
\*===--------------------------------------------------------------------===*/

#define ARGBLOCK   0x800F0000
#define PERF_MMIO  0x01001000
#define HTIF_TOHOST 0x01000008
#define MAGIC_BEGIN 0x71B00001
#define MAGIC_END   0x71B00002
#define MAGIC_BUF   0x71B0000B
#define MAGIC_DONE  0x71B0000D

  .section .text._start, "ax", @progbits
  .globl _start
  .option norelax
_start:
  csrr t2, mhartid
  li t0, ARGBLOCK
  lw t1, 12(t0)
  bgeu t2, t1, 4f
  li sp, 0x800d0000
  slli t3, t2, 16
  sub sp, sp, t3
  /* Enable the FPU and the vector unit: mstatus.FS = mstatus.VS = Initial. */
  li t0, (1 << 29) | (1 << 13) | (1 << 9)
  csrs mstatus, t0
  /* Argblock: [0] entry, [1] argc, [2] ndump, [3] harts, [4..] 8 slots of
   * 8 bytes (low word used), then ndump (addr, nbytes) pairs. */
  li t0, ARGBLOCK
  lw t1, 0(t0)            /* kernel entry */
  li s0, PERF_MMIO
  /* Load the low word of every slot; the kernel reads only its argc. */
  lw a0, 16(t0)
  lw a1, 24(t0)
  lw a2, 32(t0)
  lw a3, 40(t0)
  lw a4, 48(t0)
  lw a5, 56(t0)
  lw a6, 64(t0)
  lw a7, 72(t0)
  /* MAGIC_BEGIN, kernel call, MAGIC_END: cycle delta = kernel cycles. */
  li t3, MAGIC_BEGIN
  bnez t2, 5f
  sw t3, 0(s0)
5:
  jalr ra, t1, 0
  csrr t0, mhartid
  slli t1, t0, 6
  li t2, 0x800e0200
  add t1, t1, t2
  li t3, 1
  vsetvli zero, t3, e32, m1, ta, ma
  vmv.v.i v1, 1
  fence iorw, iorw
  vse32.v v1, (t1)
  fence iorw, iorw
  bnez t0, 4f
  li t0, ARGBLOCK
  lw t3, 12(t0)
6:
  vle32.v v1, (t2)
  vmv.x.s t4, v1
  beqz t4, 6b
  addi t2, t2, 64
  addi t3, t3, -1
  bnez t3, 6b
  li s0, PERF_MMIO
  li t3, MAGIC_END
  sw t3, 0(s0)
  /* Stream the dump ranges (pairs start at ARGBLOCK + 16 + 8*8). */
  li t0, ARGBLOCK
  lw t2, 8(t0)            /* ndump */
  addi t0, t0, 80
1:
  beqz t2, 3f
  lw t3, 0(t0)            /* range addr    */
  lw t4, 4(t0)            /* range nbytes  */
  li t5, MAGIC_BUF
  sw t5, 0(s0)
  sw t3, 0(s0)
  srli t6, t4, 2          /* nwords */
  sw t6, 0(s0)
2:
  beqz t6, 21f
  lw t5, 0(t3)
  sw t5, 0(s0)
  addi t3, t3, 4
  addi t6, t6, -1
  j 2b
21:
  addi t0, t0, 8
  addi t2, t2, -1
  j 1b
3:
  li t3, MAGIC_DONE
  sw t3, 0(s0)
  /* HTIF exit(0): tohost = (0 << 1) | 1. */
  li t0, HTIF_TOHOST
  li t1, 1
  sw t1, 0(t0)
4:
  wfi
  j 4b
