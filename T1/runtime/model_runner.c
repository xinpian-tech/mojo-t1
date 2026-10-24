#include <dlfcn.h>
#include <elf.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include <pokedex_interface.h>
#include "t1_layout.h"

typedef struct {
  uint8_t *memory;
  FILE *events;
  uint64_t instructions;
  bool exited;
  uint32_t exit_code;
} ModelBus;

static void fail(const char *message) {
  fprintf(stderr, "t1-model-runner: %s\n", message);
  exit(1);
}

static bool valid_range(uint32_t address, size_t size) {
  return address >= T1_SRAM_BASE &&
         (uint64_t)address + size <= (uint64_t)T1_SRAM_BASE + T1_SRAM_SIZE;
}

#define DEFINE_READ(width, type)                                               \
  static int read_##width(void *data, uint32_t address, type *value) {          \
    ModelBus *bus = data;                                                     \
    if (!valid_range(address, sizeof(*value)))                                 \
      return 1;                                                              \
    memcpy(value, bus->memory + address - T1_SRAM_BASE, sizeof(*value));        \
    return 0;                                                                \
  }

DEFINE_READ(1, uint8_t)
DEFINE_READ(2, uint16_t)
DEFINE_READ(4, uint32_t)

static int write_bytes(ModelBus *bus, uint32_t address, const void *value,
                       size_t size) {
  if (!valid_range(address, size))
    return 1;
  memcpy(bus->memory + address - T1_SRAM_BASE, value, size);
  return 0;
}

static int write_1(void *data, uint32_t address, uint8_t value) {
  return write_bytes(data, address, &value, sizeof(value));
}

static int write_2(void *data, uint32_t address, uint16_t value) {
  return write_bytes(data, address, &value, sizeof(value));
}

static int write_4(void *data, uint32_t address, uint32_t value) {
  ModelBus *bus = data;
  if (address == T1_PERF_MMIO) {
    if (fprintf(bus->events,
                "{\"cycle\": %" PRIu64 ", \"event\": \"profile\", \"value\": %u}\n",
                bus->instructions, value) < 0)
      fail("cannot write performance events");
    return 0;
  }
  if (address == T1_HTIF_TOHOST && (value & 1)) {
    bus->exited = true;
    bus->exit_code = value >> 1;
    return 0;
  }
  return write_bytes(bus, address, &value, sizeof(value));
}

static int unsupported_amo(void *data, uint32_t address, uint8_t operation,
                           uint32_t value, uint32_t *result) {
  fail("AMO is not implemented by this validation runner");
  return 1;
}

static int unsupported_lr(void *data, uint32_t address, uint32_t *result) {
  fail("LR/SC is not implemented by this validation runner");
  return 1;
}

static int unsupported_sc(void *data, uint32_t address, uint32_t value,
                          uint32_t *result) {
  fail("LR/SC is not implemented by this validation runner");
  return 1;
}

static uint32_t load_image(ModelBus *bus, const char *path) {
  FILE *image = fopen(path, "rb");
  if (!image)
    fail("cannot open ELF image");
  Elf32_Ehdr header;
  if (fread(&header, sizeof(header), 1, image) != 1 ||
      memcmp(header.e_ident, ELFMAG, SELFMAG) ||
      header.e_ident[EI_CLASS] != ELFCLASS32 ||
      header.e_ident[EI_DATA] != ELFDATA2LSB ||
      header.e_machine != EM_RISCV || header.e_phentsize != sizeof(Elf32_Phdr))
    fail("expected little-endian RV32 ELF");
  for (unsigned index = 0; index < header.e_phnum; ++index) {
    Elf32_Phdr segment;
    if (fseek(image, header.e_phoff + index * sizeof(segment), SEEK_SET) ||
        fread(&segment, sizeof(segment), 1, image) != 1)
      fail("invalid ELF program header");
    if (segment.p_type != PT_LOAD)
      continue;
    if (segment.p_filesz > segment.p_memsz ||
        !valid_range(segment.p_vaddr, segment.p_memsz))
      fail("ELF segment outside device memory");
    uint8_t *destination = bus->memory + segment.p_vaddr - T1_SRAM_BASE;
    memset(destination, 0, segment.p_memsz);
    if (fseek(image, segment.p_offset, SEEK_SET) ||
        fread(destination, 1, segment.p_filesz, image) != segment.p_filesz)
      fail("truncated ELF segment");
  }
  fclose(image);
  if (!valid_range(header.e_entry, 4))
    fail("invalid ELF entry");
  return header.e_entry;
}

int main(int argc, char **argv) {
  if (argc < 3 || strcmp(argv[1], "run"))
    fail("usage: t1-model-runner run image.elf [Pokedex memory/perf options]");
  const char *event_path = "mmio-event.jsonl";
  const char *dump_path = NULL;
  uint32_t dump_begin = 0, dump_end = 0;
  unsigned vlen = 1024;
  for (int index = 3; index < argc; index += 2) {
    if (index + 1 >= argc)
      fail("missing option argument");
    const char *value = argv[index + 1];
    if (!strcmp(argv[index], "--machine")) {
      if (strcmp(value, "t1emu"))
        fail("only the t1emu memory map is supported");
    } else if (!strcmp(argv[index], "--vlen")) {
      vlen = strtoul(value, NULL, 10);
    } else if (!strcmp(argv[index], "--perf-event-path")) {
      event_path = value;
    } else if (!strcmp(argv[index], "--memory-dump-path")) {
      dump_path = value;
    } else if (!strcmp(argv[index], "--memory-dump-range")) {
      if (sscanf(value, "0x%x:0x%x", &dump_begin, &dump_end) != 2 ||
          dump_end <= dump_begin || !valid_range(dump_begin, dump_end - dump_begin))
        fail("invalid memory dump range");
    } else {
      fail("unsupported option");
    }
  }
  if (vlen < 64 || vlen > 65536 || (vlen & (vlen - 1)))
    fail("invalid VLEN");
  const char *thread_option = getenv("T1RT_THREAD_COUNT");
  unsigned thread_count = thread_option ? strtoul(thread_option, NULL, 10) : 4;
  if (thread_count < 1 || thread_count > 4)
    fail("hart count must be 1..4");
  const char *library_path = getenv("POKEDEX_MODEL_LIB");
  if (!library_path)
    fail("POKEDEX_MODEL_LIB must name the thread-tilelang model library");
  void *library = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
  if (!library)
    fail(dlerror());
  pokedex_fn_abi_version abi = dlsym(library, "EXPORT_pokedex_abi_version");
  pokedex_fn_create_model create = dlsym(library, "EXPORT_pokedex_create_model");
  if (!abi || !create || strcmp(abi(), POKEDEX_ABI_VERSION))
    fail("Pokedex model ABI mismatch");
  ModelBus bus = {0};
  bus.memory = mmap(NULL, T1_SRAM_SIZE, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (bus.memory == MAP_FAILED)
    fail("cannot allocate device memory");
  bus.events = fopen(event_path, "w");
  if (!bus.events)
    fail("cannot open performance event file");
  uint32_t entry = load_image(&bus, argv[2]);
  unsigned log_vlen = 0;
  for (unsigned remaining = vlen; remaining > 1; remaining >>= 1)
    ++log_vlen;
  struct pokedex_create_info config = {
      .cfg0 = 2 | 8 | (log_vlen << 8) | (1 << 15) | (5 << 16),
      .cfg1 = 2 | (8 << 8) | (5 << 12),
  };
  struct pokedex_model models[4];
  bool parked[4] = {false};
  uint64_t retired[4] = {0};
  for (unsigned hart = 0; hart < thread_count; ++hart) {
    char error[256] = {0};
    models[hart] = create(&config, error, sizeof(error));
    if (!models[hart].data)
      fail(error);
    models[hart].vtable->set_hart_id(models[hart].data, hart);
    models[hart].vtable->reset(models[hart].data, entry);
  }
  const struct pokedex_mem_callback_vtable callbacks = {
      read_1, read_2, read_4, write_1, write_2, write_4,
      unsupported_amo, unsupported_lr, unsupported_sc,
  };
  struct pokedex_mem_callback memory = {&callbacks, &bus};
  while (!bus.exited) {
    bool progressed = false;
    for (unsigned hart = 0; hart < thread_count && !bus.exited; ++hart) {
      struct pokedex_model model = models[hart];
      for (unsigned quantum = 0; quantum < 64 && !parked[hart] && !bus.exited; ++quantum) {
        uint32_t pc = model.vtable->get_pc(model.data);
        uint16_t low;
        if (read_2(&bus, pc, &low))
          fail("instruction fetch outside memory");
        uint8_t status;
        uint32_t instruction = low;
        if ((low & 3) == 3) {
          if (read_4(&bus, pc, &instruction))
            fail("truncated instruction");
          if (instruction == 0x10500073) {
            parked[hart] = true;
            break;
          }
          status = model.vtable->execute(model.data, pc, instruction, memory);
        } else {
          status = model.vtable->execute_compressed(model.data, pc, low, memory);
        }
        if (status) {
          fprintf(stderr, "hart %u exception: pc=%08x instruction=%08x\n", hart, pc, instruction);
          fail("model execution exception");
        }
        progressed = true;
        ++retired[hart];
        if (++bus.instructions > UINT64_C(10000000000))
          fail("instruction limit exceeded (possible barrier deadlock)");
      }
    }
    if (!progressed && !bus.exited)
      fail("all harts parked before HTIF exit");
  }
  if (fclose(bus.events))
    fail("cannot close performance event file");
  if (bus.exit_code)
    fail("kernel returned a failing HTIF status");
  if (dump_path) {
    if (dump_end <= dump_begin)
      fail("memory dump requires a range");
    FILE *dump = fopen(dump_path, "wb");
    if (!dump || fwrite(bus.memory + dump_begin - T1_SRAM_BASE, 1,
                         dump_end - dump_begin, dump) != dump_end - dump_begin || fclose(dump))
      fail("cannot write memory dump");
  }
  for (unsigned hart = 0; hart < thread_count; ++hart) {
    fprintf(stderr, "hart %u: retired=%" PRIu64 "\n", hart, retired[hart]);
    models[hart].vtable->destroy(models[hart].data);
  }
  munmap(bus.memory, T1_SRAM_SIZE);
  dlclose(library);
  return 0;
}
