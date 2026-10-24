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
 * t1rt.c — T1 simulation engine for the T1 RVV accelerator
 * driven through the t1rocketemu batch simulator.
 *
 * Execution model (milestone 1, see T1/docs/minimal-kernel-plan.md):
 * the simulator loads one ELF at reset and runs it to completion, so the
 * runtime keeps a host-side shadow of the device window and, on synchronize,
 * composes trampoline + kernel + argblock + buffer contents into a single
 * ELF image, runs the simulator, and reads results and the kernel cycle
 * count back from the PERF MMIO event stream (mmio-event.jsonl).
 *
 * Environment:
 *   T1RT_EMULATOR       path to the simulator binary             (required)
 *   T1RT_TRAMPOLINE     path to trampoline.bin (flat, SRAM base) (required)
 *   T1RT_WORKDIR        scratch directory (default: mkdtemp under $TMPDIR)
 *   T1RT_VERBOSE        1 => log launches and cycle counts to stderr
 *   T1RT_MEMDUMP        1 => read results back from the simulator's
 *                       exit-time memory dump (needs emulator-memdump.patch)
 *                       instead of streaming them over the PERF MMIO
 *   T1RT_DUMP_MAX_BYTES buffers larger than this are loaded but never read
 *                       back (streamed read-only weights)
 *   T1RT_EMULATOR_KIND  "pokedex" => drive the pokedex ISA model's batch
 *                       mode instead of the verilated RTL (same protocol;
 *                       timestamps are retired instructions, not cycles)
 *   T1RT_POKEDEX_VLEN   VLEN passed to pokedex (default 1024)
 *
 * One block of up to four harts is supported. The configured emulator
 * hart count must cover the launch width; larger grids are rejected.
\*===--------------------------------------------------------------------===*/

#include "t1_layout.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

/*--- error convention ----------------------------------------------------*/

static const char *t1_err(const char *msg) { return strdup(msg); }
#define T1_OK NULL
#define T1_TRY(cond, msg)                                                      \
  do {                                                                         \
    if (!(cond))                                                               \
      return t1_err(msg);                                                      \
  } while (0)

/*--- handles -------------------------------------------------------------*/

typedef struct T1Buffer {
  uint32_t device_addr; /* address inside the SRAM window (0 for host-only) */
  size_t len;           /* elements */
  size_t elem_size;
  size_t bytes;
  uint8_t *shadow; /* host mirror of the device contents */
  int refcount;
  int device_resident;      /* allocated from the arena => dumped after runs */
  struct T1Buffer *parent;  /* sub-buffer parent (owns the shadow) */
  struct T1Context *ctx;
  struct T1Buffer *next;
} T1Buffer;

typedef struct T1Function {
  uint8_t *elf;
  size_t elf_len;
  uint32_t entry; /* resolved vaddr of the kernel symbol */
  int refcount;
  struct T1Context *ctx;
} T1Function;

typedef struct T1Launch {
  T1Function *func;
  uint32_t argc;
  uint32_t threads;
  uint64_t arg_slots[T1_ARGBLOCK_MAX_ARGS];
  struct T1Launch *next;
} T1Launch;

typedef struct T1Timer {
  uint64_t start_cycles; /* accumulated cycles when the timer started */
} T1Timer;

typedef struct T1Context {
  int refcount;
  int id;
  uint32_t arena_next;
  T1Buffer *buffers;
  T1Launch *pending, *pending_tail;
  uint64_t total_cycles;  /* accumulated kernel cycles over all runs */
  uint64_t last_cycles;   /* kernel cycles of the most recent run */
  uint64_t dump_max;      /* buffers larger than this are not dumped back */
  int memdump;            /* readback via simulator memory dump, not PERF */
  int pokedex;            /* emulator is pokedex (ISA model), not the RTL sim */
  char *workdir;
  const char *emulator;
  const char *trampoline_path;
  uint8_t *trampoline;
  size_t trampoline_len;
  int verbose;
} T1Context;

/* The Mojo side treats DeviceStream as opaque; a degenerate single-queue
 * backend can hand back the context itself. */
typedef T1Context T1Stream;



/*--- small utils ---------------------------------------------------------*/

static const char *read_file(const char *path, uint8_t **out, size_t *out_len) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return t1_err("T1RT: cannot open file");
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *buf = malloc(n > 0 ? (size_t)n : 1);
  if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
    fclose(f);
    free(buf);
    return t1_err("T1RT: short read");
  }
  fclose(f);
  *out = buf;
  *out_len = (size_t)n;
  return T1_OK;
}

/*--- ELF32 (little-endian RISC-V) ---------------------------------------*/

typedef struct {
  uint8_t ident[16];
  uint16_t type, machine;
  uint32_t version, entry, phoff, shoff, flags;
  uint16_t ehsize, phentsize, phnum, shentsize, shnum, shstrndx;
} Elf32Ehdr;

typedef struct {
  uint32_t type, offset, vaddr, paddr, filesz, memsz, flags, align;
} Elf32Phdr;

typedef struct {
  uint32_t name, type, flags, addr, offset, size, link, info, addralign,
      entsize;
} Elf32Shdr;

typedef struct {
  uint32_t name, value, size;
  uint8_t info, other;
  uint16_t shndx;
} Elf32Sym;

#define PT_LOAD 1
#define SHT_SYMTAB 2

/* Finds `symbol` (exact byte-for-byte match, no normalization: Mojo mangled
 * names contain ::, (, ] ...) in the ELF's symtab. */
static const char *elf_find_symbol(const uint8_t *elf, size_t len,
                                   const char *symbol, uint32_t *out_vaddr) {
  const Elf32Ehdr *eh = (const Elf32Ehdr *)elf;
  T1_TRY(len > sizeof(*eh) && memcmp(eh->ident, "\x7f""ELF", 4) == 0,
         "T1RT: kernel is not an ELF");
  T1_TRY(eh->ident[4] == 1, "T1RT: kernel ELF is not ELF32");
  const Elf32Shdr *sh = (const Elf32Shdr *)(elf + eh->shoff);
  for (unsigned i = 0; i < eh->shnum; i++) {
    if (sh[i].type != SHT_SYMTAB)
      continue;
    const Elf32Sym *syms = (const Elf32Sym *)(elf + sh[i].offset);
    unsigned nsyms = sh[i].size / sizeof(Elf32Sym);
    const char *strtab = (const char *)(elf + sh[sh[i].link].offset);
    for (unsigned s = 0; s < nsyms; s++) {
      if (strcmp(strtab + syms[s].name, symbol) == 0) {
        *out_vaddr = syms[s].value;
        return T1_OK;
      }
    }
  }
  return t1_err("T1RT: kernel entry symbol not found in ELF symtab");
}

/*--- composed image writer ----------------------------------------------*/

typedef struct {
  uint32_t vaddr;
  const uint8_t *data;
  uint32_t filesz;
  uint32_t memsz;
} Segment;

static const char *write_image(const char *path, const Segment *segs,
                               unsigned nsegs, uint32_t entry) {
  FILE *f = fopen(path, "wb");
  T1_TRY(f != NULL, "T1RT: cannot create image file");
  Elf32Ehdr eh = {0};
  memcpy(eh.ident, "\x7f""ELF\x01\x01\x01", 7);
  eh.type = 2 /*ET_EXEC*/;
  eh.machine = 243 /*EM_RISCV*/;
  eh.version = 1;
  eh.entry = entry;
  eh.phoff = sizeof(Elf32Ehdr);
  eh.flags = 0x3; /* RVC, single-float ABI (matches the toolchain output) */
  eh.ehsize = sizeof(Elf32Ehdr);
  eh.phentsize = sizeof(Elf32Phdr);
  eh.phnum = (uint16_t)nsegs;
  uint32_t off = sizeof(Elf32Ehdr) + nsegs * sizeof(Elf32Phdr);
  fwrite(&eh, sizeof(eh), 1, f);
  for (unsigned i = 0; i < nsegs; i++) {
    Elf32Phdr ph = {0};
    ph.type = PT_LOAD;
    ph.offset = off;
    ph.vaddr = ph.paddr = segs[i].vaddr;
    ph.filesz = segs[i].filesz;
    ph.memsz = segs[i].memsz;
    ph.flags = 0x7; /* rwx */
    ph.align = 4;
    fwrite(&ph, sizeof(ph), 1, f);
    off += segs[i].filesz;
  }
  for (unsigned i = 0; i < nsegs; i++)
    fwrite(segs[i].data, 1, segs[i].filesz, f);
  int bad = ferror(f);
  fclose(f);
  T1_TRY(!bad, "T1RT: image write failed");
  return T1_OK;
}

/*--- the batch run -------------------------------------------------------*/

static unsigned g_launch_no; /* for per-launch RTL event traces */

static const char *run_emulator(T1Context *ctx, const char *image,
                                uint32_t dump_lo, uint32_t dump_hi) {
  char event_path[4096], rtl_path[4096], memdump_path[4096];
  snprintf(event_path, sizeof event_path, "%s/mmio-event.jsonl", ctx->workdir);
  /* T1RT_KEEP_RTL_EVENTS=1 keeps one (large) RTL retirement trace per
   * launch instead of overwriting a single file. */
  if (getenv("T1RT_KEEP_RTL_EVENTS"))
    snprintf(rtl_path, sizeof rtl_path, "%s/rtl-event.%u.jsonl", ctx->workdir,
             g_launch_no++);
  else
    snprintf(rtl_path, sizeof rtl_path, "%s/rtl-event.jsonl", ctx->workdir);
  snprintf(memdump_path, sizeof memdump_path, "%s/memdump.bin", ctx->workdir);
  unlink(event_path);
  unlink(memdump_path);

  pid_t pid = fork();
  T1_TRY(pid >= 0, "T1RT: fork failed");
  if (pid == 0) {
    if (chdir(ctx->workdir) != 0)
      _exit(126);
    char elf_arg[4200], rtl_arg[4200], md_arg[4200], mr_arg[128];
    snprintf(elf_arg, sizeof elf_arg, "+t1_elf_file=%s", image);
    if (getenv("T1RT_DISABLE_RTL_EVENTS"))
      snprintf(rtl_arg, sizeof rtl_arg, "+t1_dev_rtl_event_off=1");
    else
      snprintf(rtl_arg, sizeof rtl_arg, "+t1_dev_rtl_event_path=%s", rtl_path);
    if (!ctx->verbose) {
      if (!freopen("/dev/null", "w", stdout) ||
          !freopen("emulator.log", "w", stderr))
        _exit(126);
    }
    if (ctx->pokedex) {
      char range_arg[64];
      snprintf(range_arg, sizeof range_arg, "0x%x:0x%x", dump_lo, dump_hi);
      const char *vlen = getenv("T1RT_POKEDEX_VLEN");
      /* Accumulating histogram: one CSV per launch, appended sequence. */
      const char *hist = getenv("T1RT_PC_HISTOGRAM");
      char hist_path[4300];
      if (hist)
        snprintf(hist_path, sizeof hist_path, "%s.%u.csv", hist,
                 (unsigned)getpid());
      if (ctx->memdump && dump_hi > dump_lo && hist)
        execl(ctx->emulator, ctx->emulator, "run", image, "--machine", "t1emu",
              "--vlen", vlen ? vlen : "1024", "--perf-event-path", event_path,
              "--memory-dump-path", memdump_path, "--memory-dump-range",
              range_arg, "--pc-histogram-path", hist_path, (char *)NULL);
      else if (ctx->memdump && dump_hi > dump_lo)
        execl(ctx->emulator, ctx->emulator, "run", image, "--machine", "t1emu",
              "--vlen", vlen ? vlen : "1024", "--perf-event-path", event_path,
              "--memory-dump-path", memdump_path, "--memory-dump-range",
              range_arg, (char *)NULL);
      else
        execl(ctx->emulator, ctx->emulator, "run", image, "--machine", "t1emu",
              "--vlen", vlen ? vlen : "1024", "--perf-event-path", event_path,
              (char *)NULL);
    } else if (ctx->memdump && dump_hi > dump_lo) {
      snprintf(md_arg, sizeof md_arg, "+t1_memory_dump_path=%s", memdump_path);
      snprintf(mr_arg, sizeof mr_arg, "+t1_memory_dump_range=0x%x:0x%x",
               dump_lo, dump_hi);
      execl(ctx->emulator, ctx->emulator, elf_arg, rtl_arg, md_arg, mr_arg,
            (char *)NULL);
    } else {
      execl(ctx->emulator, ctx->emulator, elf_arg, rtl_arg, (char *)NULL);
    }
    _exit(127);
  }
  int status = 0;
  pid_t waited;
  do {
    waited = waitpid(pid, &status, 0);
  } while (waited < 0 && errno == EINTR);
  if (waited != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
    return t1_err("T1RT: emulator failed; inspect emulator.log in T1RT_WORKDIR");
  return T1_OK;
}

/* Parses mmio-event.jsonl.  Every PERF store is observed `k` times on the bus
 * (k is detected from the number of consecutive MAGIC_BEGIN records), so the
 * value stream is decimated by k before protocol parsing. */
static const char *parse_events(T1Context *ctx) {
  char event_path[4096];
  snprintf(event_path, sizeof event_path, "%s/mmio-event.jsonl", ctx->workdir);
  FILE *f = fopen(event_path, "r");
  T1_TRY(f != NULL, "T1RT: emulator produced no mmio-event.jsonl");

  uint32_t *vals = NULL;
  uint64_t *cycles = NULL;
  size_t n = 0, cap = 0;
  char line[256];
  while (fgets(line, sizeof line, f)) {
    uint64_t cyc;
    uint32_t val;
    if (sscanf(line, "{\"cycle\": %lu, \"event\": \"profile\", \"value\": %u}",
               &cyc, &val) == 2) {
      if (n == cap) {
        cap = cap ? cap * 2 : 1024;
        vals = realloc(vals, cap * sizeof(*vals));
        cycles = realloc(cycles, cap * sizeof(*cycles));
      }
      vals[n] = val;
      cycles[n] = cyc;
      n++;
    }
  }
  fclose(f);
  const char *fail = NULL;
  size_t k = 0, i = 0;
  while (i < n && vals[i] == T1_MAGIC_BEGIN)
    k++, i++;
  if (k == 0) {
    fail = "T1RT: MAGIC_BEGIN missing from the event stream "
           "(kernel image did not run)";
    goto out;
  }
  /* From here on, consume every k-th record. */
#define NEXT(vout, cout)                                                       \
  do {                                                                         \
    if (i + k > n) {                                                           \
      fail = "T1RT: event stream truncated";                                   \
      goto out;                                                                \
    }                                                                          \
    vout = vals[i];                                                            \
    cout = cycles[i];                                                          \
    i += k;                                                                    \
  } while (0)
  uint64_t begin_cycle = cycles[0], c;
  uint32_t v;
  NEXT(v, c);
  if (v != T1_MAGIC_END) {
    fail = "T1RT: MAGIC_END missing (kernel did not return)";
    goto out;
  }
  ctx->last_cycles = c - begin_cycle;
  ctx->total_cycles += ctx->last_cycles;
  for (;;) {
    NEXT(v, c);
    if (v == T1_MAGIC_DONE)
      break;
    if (v != T1_MAGIC_BUF) {
      fail = "T1RT: malformed dump stream";
      goto out;
    }
    uint32_t addr, nwords;
    NEXT(addr, c);
    NEXT(nwords, c);
    /* Locate the buffer covering [addr, addr + 4*nwords). */
    T1Buffer *buf = NULL;
    for (T1Buffer *b = ctx->buffers; b; b = b->next)
      if (b->device_resident && b->device_addr == addr &&
          b->bytes >= 4u * nwords) {
        buf = b;
        break;
      }
    for (uint32_t w = 0; w < nwords; w++) {
      NEXT(v, c);
      if (buf)
        memcpy(buf->shadow + 4u * w, &v, 4);
    }
  }
#undef NEXT
out:
  free(vals);
  free(cycles);
  return fail ? t1_err(fail) : T1_OK;
}

static const char *flush_launches(T1Context *ctx) {
  if (!ctx->pending)
    return T1_OK;
  const char *err = NULL;
  while (ctx->pending && !err) {
    T1Launch *launch = ctx->pending;
    ctx->pending = launch->next;
    if (!ctx->pending)
      ctx->pending_tail = NULL;

    /* Argblock: header, 8 slots, dump descriptors (device buffers up to
     * dump_max bytes; larger ones are loaded but not read back).  In memdump
     * mode nothing is streamed over PERF; the same buffer set is instead
     * refreshed from the simulator's exit-time memory dump. */
    uint32_t ndump = 0, nbufsegs = 0;
    uint32_t dump_lo = UINT32_MAX, dump_hi = 0;
    for (T1Buffer *b = ctx->buffers; b; b = b->next)
      if (b->device_resident && !b->parent) {
        nbufsegs++;
        if (b->bytes <= ctx->dump_max) {
          if (b->device_addr < dump_lo)
            dump_lo = b->device_addr;
          if (b->device_addr + (uint32_t)b->bytes > dump_hi)
            dump_hi = b->device_addr + (uint32_t)b->bytes;
          if (!ctx->memdump)
            ndump++;
        }
      }
    size_t argblock_words = 4 + 2 * T1_ARGBLOCK_MAX_ARGS + 2 * ndump;
    uint32_t *argblock = calloc(argblock_words, 4);
    argblock[0] = launch->func->entry;
    argblock[1] = launch->argc;
    argblock[2] = ndump;
    argblock[3] = launch->threads;
    for (uint32_t a = 0; a < launch->argc; a++) {
      argblock[4 + 2 * a] = (uint32_t)(launch->arg_slots[a] & 0xffffffffu);
      argblock[4 + 2 * a + 1] = (uint32_t)(launch->arg_slots[a] >> 32);
    }
    uint32_t d = 0;
    if (!ctx->memdump)
      for (T1Buffer *b = ctx->buffers; b; b = b->next)
        if (b->device_resident && !b->parent && b->bytes <= ctx->dump_max) {
          argblock[4 + 2 * T1_ARGBLOCK_MAX_ARGS + 2 * d] = b->device_addr;
          argblock[4 + 2 * T1_ARGBLOCK_MAX_ARGS + 2 * d + 1] =
              (uint32_t)((b->bytes + 3) & ~3u);
          d++;
        }

    /* Segments: trampoline, argblock, kernel PT_LOADs, device buffers. */
    unsigned nsegs = 3 + nbufsegs;
    const Elf32Ehdr *keh = (const Elf32Ehdr *)launch->func->elf;
    const Elf32Phdr *kph = (const Elf32Phdr *)(launch->func->elf + keh->phoff);
    for (unsigned p = 0; p < keh->phnum; p++)
      if (kph[p].type == PT_LOAD && kph[p].vaddr >= T1_SRAM_BASE)
        nsegs++;
    Segment *segs = calloc(nsegs, sizeof(Segment));
    unsigned s = 0;
    uint8_t thread_state[1024] = {0};
    segs[s++] = (Segment){0x800e0000u, thread_state, sizeof(thread_state), sizeof(thread_state)};
    segs[s++] = (Segment){T1_TRAMPOLINE_BASE, ctx->trampoline,
                          (uint32_t)ctx->trampoline_len,
                          (uint32_t)ctx->trampoline_len};
    segs[s++] = (Segment){T1_ARGBLOCK_ADDR, (const uint8_t *)argblock,
                          (uint32_t)(argblock_words * 4),
                          (uint32_t)(argblock_words * 4)};
    for (unsigned p = 0; p < keh->phnum; p++)
      if (kph[p].type == PT_LOAD && kph[p].vaddr >= T1_SRAM_BASE)
        segs[s++] = (Segment){kph[p].vaddr, launch->func->elf + kph[p].offset,
                              kph[p].filesz, kph[p].memsz};
    for (T1Buffer *b = ctx->buffers; b; b = b->next)
      if (b->device_resident && !b->parent)
        segs[s++] = (Segment){b->device_addr, b->shadow, (uint32_t)b->bytes,
                              (uint32_t)b->bytes};

    char image[4096];
    snprintf(image, sizeof image, "%s/image.elf", ctx->workdir);
    err = (char *)write_image(image, segs, s, T1_TRAMPOLINE_BASE);
    if (!err && ctx->verbose)
      fprintf(stderr, "T1RT: launching kernel entry=0x%08x argc=%u ndump=%u\n",
              launch->func->entry, launch->argc, ndump);
    if (!err)
      err = (char *)run_emulator(ctx, image, dump_lo, dump_hi);
    if (!err)
      err = (char *)parse_events(ctx);
    if (!err && ctx->memdump && dump_hi > dump_lo) {
      char memdump_path[4096];
      snprintf(memdump_path, sizeof memdump_path, "%s/memdump.bin",
               ctx->workdir);
      uint8_t *dump = NULL;
      size_t dump_len = 0;
      err = (char *)read_file(memdump_path, &dump, &dump_len);
      if (!err && dump_len != dump_hi - dump_lo)
        err = (char *)t1_err("T1RT: memory dump has unexpected size");
      if (!err)
        for (T1Buffer *b = ctx->buffers; b; b = b->next)
          if (b->device_resident && !b->parent && b->bytes <= ctx->dump_max)
            memcpy(b->shadow, dump + (b->device_addr - dump_lo), b->bytes);
      free(dump);
    }
    if (!err && ctx->verbose)
      fprintf(stderr, "T1RT: kernel done: %llu cycles (total %llu)\n",
              (unsigned long long)ctx->last_cycles,
              (unsigned long long)ctx->total_cycles);

    free(segs);
    free(argblock);
    free(launch);
  }
  return err;
}

/*=========================================================================*\
 * AsyncRT ABI
\*=========================================================================*/

const char *t1_DeviceContext_create(const T1Context **result,
                                         const char *api, int id) {
  if (strcmp(api, "t1") != 0)
    return t1_err("T1RT: unsupported api (expected \"t1\")");
  {
    T1Context *ctx = calloc(1, sizeof(T1Context));
    ctx->refcount = 1;
    ctx->id = id > 0 ? id : 0;
    ctx->arena_next = T1_ARENA_BASE;
    ctx->emulator = getenv("T1RT_EMULATOR");
    ctx->trampoline_path = getenv("T1RT_TRAMPOLINE");
    const char *verbose = getenv("T1RT_VERBOSE");
    ctx->verbose = verbose && verbose[0] == '1';
    /* The PERF dump stream costs one MMIO store per 4 bytes; readback of
     * large read-only inputs (e.g. streamed weights) would dominate the
     * simulation.  Buffers above this size are still loaded into the image
     * but their device contents are not read back — set it above the size
     * of every buffer a kernel writes. */
    const char *dump_max = getenv("T1RT_DUMP_MAX_BYTES");
    ctx->dump_max = dump_max ? strtoull(dump_max, NULL, 0) : UINT64_MAX;
    /* With a simulator that supports +t1_memory_dump_path (see
     * T1/runtime/emulator-memdump.patch), results are read back from a file
     * written at exit instead of being streamed one word at a time over the
     * PERF MMIO: zero simulation-cycle cost.  The dump_max threshold then
     * selects which buffers are refreshed from the dump. */
    const char *memdump = getenv("T1RT_MEMDUMP");
    ctx->memdump = memdump && memdump[0] == '1';
    /* T1RT_EMULATOR_KIND=pokedex switches the invocation to the pokedex ISA
     * model's batch mode (--machine t1emu): same address map, HTIF and PERF
     * protocol, ~1000x faster than the verilated RTL, but timestamps are a
     * retired-instruction count (1-IPC approximation), not RTL cycles. */
    const char *kind = getenv("T1RT_EMULATOR_KIND");
    ctx->pokedex = kind && strcmp(kind, "pokedex") == 0;
    if (!ctx->emulator || !ctx->trampoline_path) {
      free(ctx);
      return t1_err("T1RT: set T1RT_EMULATOR and T1RT_TRAMPOLINE");
    }
    if (const char *err = read_file(ctx->trampoline_path, &ctx->trampoline,
                                    &ctx->trampoline_len)) {
      free(ctx);
      return err;
    }
    const char *workdir = getenv("T1RT_WORKDIR");
    if (workdir) {
      ctx->workdir = strdup(workdir);
    } else {
      char tmpl[4096];
      snprintf(tmpl, sizeof tmpl, "%s/t1rt-XXXXXX",
               getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
      char *created = mkdtemp(tmpl);
      if (!created) {
        free(ctx->trampoline);
        free(ctx);
        return t1_err("T1: cannot create simulator work directory");
      }
      ctx->workdir = strdup(created);
    }
    *result = ctx;
  }
  return T1_OK;
}

void t1_DeviceContext_retain(const T1Context *ctx) {
  ((T1Context *)ctx)->refcount++;
}

void t1_DeviceContext_release(const T1Context *ctx) {
  ((T1Context *)ctx)->refcount--;
}

const char *t1_DeviceContext_synchronize(const T1Context *ctx) {
  return flush_launches((T1Context *)ctx);
}

void t1_DeviceContext_strfree(const char *ptr) { free((void *)ptr); }

typedef struct {
  const char *data;
  size_t len;
} T1StringRef;

void t1_DeviceContext_deviceApi(T1StringRef *result,
                                     const T1Context *ctx) {
  result->data = "t1";
  result->len = 2;
}

void t1_DeviceContext_archName(T1StringRef *result,
                                    const T1Context *ctx) {
  static const char arch[] = "t1";
  result->data = arch;
  result->len = sizeof(arch) - 1;
}

const char *t1_DeviceContext_deviceName(const T1Context *ctx) {
  return strdup("T1 RVV accelerator (t1rocketemu)");
}

int64_t t1_DeviceContext_id(const T1Context *ctx) { return ctx->id; }

int32_t *t1_DeviceContext_numberOfDevices(const char *kind) {
  static int32_t one = 1;
  static int32_t zero = 0;
  return strcmp(kind, "t1") == 0 ? &one : &zero;
}

/*--- memory --------------------------------------------------------------*/

static T1Buffer *new_buffer(T1Context *ctx, size_t len, size_t elem_size,
                            int device_resident) {
  T1Buffer *b = calloc(1, sizeof(T1Buffer));
  b->refcount = 1;
  b->len = len;
  b->elem_size = elem_size;
  b->bytes = len * elem_size;
  b->ctx = ctx;
  b->device_resident = device_resident;
  size_t alloc = (b->bytes + 63) & ~(size_t)63;
  b->shadow = calloc(1, alloc ? alloc : 64);
  if (device_resident) {
    uint32_t addr = (ctx->arena_next + 63) & ~63u;
    b->device_addr = addr;
    ctx->arena_next = addr + (uint32_t)alloc;
  }
  b->next = ctx->buffers;
  ctx->buffers = b;
  return b;
}

const char *t1_DeviceContext_createBuffer_async(const T1Buffer **result,
                                                     void **device_ptr,
                                                     const T1Context *ctx,
                                                     size_t len,
                                                     size_t elem_size) {
  T1Context *c = (T1Context *)ctx;
  T1_TRY(c->arena_next + len * elem_size < T1_ARENA_END,
         "T1RT: device window exhausted");
  T1Buffer *b = new_buffer(c, len, elem_size, /*device_resident=*/1);
  *result = b;
  *device_ptr = (void *)(uintptr_t)b->device_addr;
  return T1_OK;
}

const char *t1_DeviceContext_createHostBuffer(const T1Buffer **result,
                                                   void **host_ptr,
                                                   const T1Context *ctx,
                                                   size_t len,
                                                   size_t elem_size) {
  T1Buffer *b = new_buffer((T1Context *)ctx, len, elem_size, 0);
  *result = b;
  *host_ptr = b->shadow;
  return T1_OK;
}

void t1_DeviceBuffer_retain(const T1Buffer *b) {
  ((T1Buffer *)b)->refcount++;
}

void t1_DeviceBuffer_release(const T1Buffer *b) {
  ((T1Buffer *)b)->refcount--;
  /* Buffers stay in the context's list; the batch model needs their
   * addresses stable for the lifetime of the context. */
}

void t1_DeviceBuffer_release_ptr(const T1Buffer *b) {
  t1_DeviceBuffer_release(b);
}

int64_t t1_DeviceBuffer_bytesize(const T1Buffer *b) {
  return (int64_t)b->bytes;
}

const T1Context *t1_DeviceBuffer_context(const T1Buffer *b) {
  return b->ctx;
}

const char *t1_DeviceBuffer_createSubBuffer(const T1Buffer **result,
                                                 const T1Buffer *parent,
                                                 size_t offset, size_t len,
                                                 size_t elem_size) {
  T1Buffer *b = calloc(1, sizeof(T1Buffer));
  b->refcount = 1;
  b->len = len;
  b->elem_size = elem_size;
  b->bytes = len * elem_size;
  b->ctx = parent->ctx;
  b->device_resident = parent->device_resident;
  b->parent = (T1Buffer *)parent;
  b->shadow = parent->shadow + offset * elem_size;
  b->device_addr = parent->device_addr + (uint32_t)(offset * elem_size);
  b->next = parent->ctx->buffers;
  ((T1Context *)parent->ctx)->buffers = b;
  return T1_OK;
}

/* COH: the "copy" directions maintain the host shadow, which *is* the
 * device image at the next doorbell (composed into the ELF) — HtoD is a
 * clean, DtoH an invalidate-and-read. Transfer size comes from the buffer. */
const char *t1_DeviceContext_HtoD_async(const T1Context *ctx,
                                             const T1Buffer *dst,
                                             const void *src) {
  /* Stream order: pending launches must observe the buffer's previous
   * contents, so drain them before mutating the shadow. */
  const char *err = flush_launches((T1Context *)ctx);
  if (err)
    return err;
  memcpy(dst->shadow, src, dst->bytes);
  return T1_OK;
}

const char *t1_DeviceContext_DtoH_async(const T1Context *ctx, void *dst,
                                             const T1Buffer *src) {
  const char *err = flush_launches((T1Context *)ctx);
  if (err)
    return err;
  memcpy(dst, src->shadow, src->bytes);
  return T1_OK;
}

const char *t1_DeviceContext_DtoD_async(const T1Context *ctx,
                                             const T1Buffer *dst,
                                             const T1Buffer *src) {
  const char *err = flush_launches((T1Context *)ctx);
  if (err)
    return err;
  memcpy(dst->shadow, src->shadow,
         dst->bytes < src->bytes ? dst->bytes : src->bytes);
  return T1_OK;
}

const char *t1_DeviceContext_setMemory_async(const T1Context *ctx,
                                                  const T1Buffer *dst,
                                                  uint64_t val,
                                                  size_t val_size) {
  const char *err = flush_launches((T1Context *)ctx);
  if (err)
    return err;
  for (size_t i = 0; i + val_size <= dst->bytes; i += val_size)
    memcpy(dst->shadow + i, &val, val_size);
  return T1_OK;
}

/*--- code loading --------------------------------------------------------*/

const char *t1_DeviceContext_loadFunction(
    const T1Function **result, const T1Context *ctx, const char *moduleName,
    const char *functionName, const char *data, size_t dataLen,
    int32_t maxDynamicSharedBytes, const char *debugLevel,
    int32_t optimizationLevel) {
  if (maxDynamicSharedBytes > 0)
    return t1_err("T1RT: dynamic shared memory is a GPU concept; T1 kernels "
                  "must not request it");
  T1Function *fn = calloc(1, sizeof(T1Function));
  fn->refcount = 1;
  fn->ctx = (T1Context *)ctx;
  fn->elf = malloc(dataLen);
  fn->elf_len = dataLen;
  memcpy(fn->elf, data, dataLen);
  if (const char *err =
          elf_find_symbol(fn->elf, fn->elf_len, functionName, &fn->entry)) {
    free(fn->elf);
    free(fn);
    return err;
  }
  *result = fn;
  return T1_OK;
}

void t1_DeviceFunction_retain(const T1Function *fn) {
  ((T1Function *)fn)->refcount++;
}

void t1_DeviceFunction_release(const T1Function *fn) {
  ((T1Function *)fn)->refcount--;
}

/*--- launch --------------------------------------------------------------*/

typedef struct T1LaunchAttribute T1LaunchAttribute;

const char *t1_DeviceContext_enqueueFunctionDirect(
    const T1Context *ctx, const T1Function *func, uint32_t gridX,
    uint32_t gridY, uint32_t gridZ, uint32_t blockX, uint32_t blockY,
    uint32_t blockZ, uint32_t sharedMemBytes, T1LaunchAttribute *attributes,
    uint32_t numAttributes, void **args, uint32_t argCount,
    uint64_t *argSizes) {
  if (gridX != 1 || gridY != 1 || gridZ != 1 || blockX < 1 || blockX > 4 || blockY != 1 ||
      blockZ != 1)
    return t1_err("T1RT: supported geometry is grid=(1,1,1), block=(1..4,1,1)");
  const char *configured_threads = getenv("T1RT_THREAD_COUNT");
  char *thread_end = NULL;
  unsigned long available_threads = configured_threads
      ? strtoul(configured_threads, &thread_end, 10) : 1;
  if (available_threads < 1 || available_threads > 4 ||
      (thread_end && *thread_end))
    return t1_err("T1RT: T1RT_THREAD_COUNT must be 1..4");
  if (blockX > available_threads)
    return t1_err("T1RT: launch exceeds configured emulator hart count");
  if (sharedMemBytes != 0)
    return t1_err("T1RT: shared memory is not available on T1");
  if (numAttributes != 0)
    return t1_err("T1RT: launch attributes are not supported on T1");
  if (argCount > T1_ARGBLOCK_MAX_ARGS)
    return t1_err("T1RT: more than 8 kernel arguments need the argblock "
                  "spill, which is not implemented yet");

  T1Launch *launch = calloc(1, sizeof(T1Launch));
  launch->func = (T1Function *)func;
  launch->argc = argCount;
  launch->threads = blockX;
  for (uint32_t i = 0; i < argCount; i++) {
    /* argSizes may be NULL (it is an optional out-of-band hint). The slot
     * convention only consumes the LOW 32 bits of every argument — device
     * pointers are 32-bit on T1 and scalar arguments are <= 4 bytes in the
     * minimal profile — so 4 bytes is always safe to read. */
    uint64_t size = argSizes ? argSizes[i] : 4;
    uint64_t slot = 0;
    memcpy(&slot, args[i], size < 8 ? size : 8);
    launch->arg_slots[i] = slot;
  }
  T1Context *c = (T1Context *)ctx;
  if (c->pending_tail)
    c->pending_tail->next = launch;
  else
    c->pending = launch;
  c->pending_tail = launch;
  return T1_OK;
}

/*--- timing (PERF: cycle-accurate, not wall clock) -----------------------*/

const char *t1_DeviceContext_startTimer(const T1Timer **result,
                                             const T1Context *ctx) {
  const char *err = flush_launches((T1Context *)ctx);
  if (err)
    return err;
  T1Timer *t = calloc(1, sizeof(T1Timer));
  t->start_cycles = ctx->total_cycles;
  *result = t;
  return T1_OK;
}

const char *t1_DeviceContext_stopTimer(int64_t *elapsed, const T1Context *ctx,
                                            const T1Timer *timer) {
  const char *err = flush_launches((T1Context *)ctx);
  if (err)
    return err;
  /* The unit is simulation cycles, not nanoseconds: on T1 the event/timer
   * interface is cycle-accurate (documented in T1/docs). */
  *elapsed = (int64_t)(ctx->total_cycles - timer->start_cycles);
  return T1_OK;
}

void t1_DeviceTimer_release(const T1Timer *timer) { free((void *)timer); }

/*--- degenerate single-stream family ------------------------------------*/

const char *t1_DeviceContext_stream(const T1Stream **result,
                                         const T1Context *ctx) {
  *result = (const T1Stream *)ctx;
  return T1_OK;
}

void t1_DeviceStream_retain(const T1Stream *s) {}
void t1_DeviceStream_release(const T1Stream *s) {}

const char *t1_DeviceStream_synchronize(const T1Stream *s) {
  return flush_launches((T1Context *)s);
}

const char *t1_DeviceContext_setAsCurrent(const T1Context *ctx) {
  return T1_OK;
}
