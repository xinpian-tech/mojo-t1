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
 * t1_layout.h — memory-map contract between the Mojo T1 compiler backend,
 * the launch trampoline and libT1RT.  Single source of truth; keep in sync
 * with KGEN/lib/Compiler/ObjectCompiler/Target/T1/T1Backend.cpp (link base)
 * and the emulator address map (difftest/dpi_t1rocketemu/src/addrspace.rs).
\*===--------------------------------------------------------------------===*/

#ifndef T1RT_LAYOUT_H
#define T1RT_LAYOUT_H

/* Emulator MMIO map (t1rocketemu). */
#define T1_HTIF_TOHOST 0x01000008u /* odd write => exit(code = value >> 1)  */
#define T1_PERF_MMIO 0x01001000u   /* u32 writes land in mmio-event.jsonl   */
#define T1_UART_BASE 0x10000000u

/* SRAM window: 0x8000_0000 .. 0xa000_0000 (512 MiB). */
#define T1_SRAM_BASE 0x80000000u
#define T1_SRAM_SIZE 0x20000000u

/* Layout inside the window:
 *   trampoline text/stack   0x8000_0000 .. 0x800F_0000 (reset jumps here)
 *   argblock                0x800F_0000 .. 0x8010_0000
 *   kernel image            0x8010_0000 .. 0x8200_0000 (T1Backend -Ttext)
 *   buffer arena            0x8200_0000 .. window end
 */
#define T1_TRAMPOLINE_BASE T1_SRAM_BASE
#define T1_ARGBLOCK_ADDR 0x800F0000u
#define T1_KERNEL_BASE 0x80100000u
#define T1_ARENA_BASE 0x82000000u
#define T1_ARENA_END (T1_SRAM_BASE + T1_SRAM_SIZE)

/* Argblock layout (all little-endian u32):
 *   [0] entry address of the kernel
 *   [1] argc (<= 8)
 *   [2] ndump — number of result ranges to stream back after the kernel
 *   [3] active hart count (1..4)
 *   [4 ..]           argc 8-byte slots; the trampoline loads the LOW word of
 *                    each slot into a0..a{argc-1} (ilp32: floats in GPRs too)
 *   [4 + 2*8 ..]     ndump (addr, nbytes) u32 pairs
 */
#define T1_ARGBLOCK_MAX_ARGS 8u

/* Result/marker protocol on the PERF stream (each u32 write becomes one
 * {"cycle": N, "event": "profile", "value": V} line in mmio-event.jsonl):
 *   MAGIC_BEGIN                    right before the kernel call
 *   MAGIC_END                      right after it returns (cycle delta = kernel cycles)
 *   per range: MAGIC_BUF, addr, nwords, nwords * data
 *   MAGIC_DONE                     then HTIF exit(0)
 */
#define T1_MAGIC_BEGIN 0x71B00001u
#define T1_MAGIC_END 0x71B00002u
#define T1_MAGIC_BUF 0x71B0000Bu
#define T1_MAGIC_DONE 0x71B0000Du

#endif /* T1RT_LAYOUT_H */
