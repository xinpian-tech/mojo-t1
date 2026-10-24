#include "HAL/M_driver_api.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static M_driver_driver_t driver;

static void check(M_driver_status_t status) {
  if (!status)
    return;
  char message[1024];
  M_driver_status_message(driver, status, message, sizeof(message));
  fprintf(stderr, "HAL failure %lld: %s\n", (long long)status, message);
  exit(1);
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s kernel.elf entry-symbol\n", argv[0]);
    return 2;
  }
  FILE *file = fopen(argv[1], "rb");
  assert(file);
  assert(fseek(file, 0, SEEK_END) == 0);
  long size = ftell(file);
  assert(size > 0);
  rewind(file);
  char *image = malloc((size_t)size);
  assert(image && fread(image, 1, (size_t)size, file) == (size_t)size);
  fclose(file);

  M_driver_version version = {0, 1, 0};
  check(M_driver_create(&version, &driver));
  const char *api = NULL;
  check(M_driver_property(driver, M_DRIVER_PROPERTY_API, &api));
  assert(strcmp(api, "t1") == 0);
  int64_t count = 0;
  check(M_driver_device_count(driver, &count));
  assert(count == 1);
  M_driver_device_t device;
  check(M_driver_device_get(driver, 0, &device));
  assert(M_driver_device_get(driver, 1, &device) == M_DRIVER_STATUS_NO_DEVICES);
  M_driver_context_t context;
  check(M_driver_context_create(device, &context));
  M_driver_queue_t queue;
  check(M_driver_queue_create(context, &queue));
  bool stream = false;
  check(M_driver_queue_is_stream(queue, &stream));
  assert(stream);

  enum { length = 64 };
  M_driver_memory_t input_x, input_y, output;
  check(M_driver_memory_alloc_sync(context, length * 4, &input_x));
  check(M_driver_memory_alloc_sync(context, length * 4, &input_y));
  check(M_driver_memory_alloc_sync(context, length * 4, &output));
  float host_x[length], host_y[length], host_out[length];
  for (int index = 0; index < length; ++index) {
    host_x[index] = (float)index;
    host_y[index] = 100.0f + (float)index;
  }
  struct M_driver_memory_view view_x = {input_x, 0, sizeof(host_x)};
  struct M_driver_memory_view view_y = {input_y, 0, sizeof(host_y)};
  struct M_driver_memory_view view_out = {output, 0, sizeof(host_out)};
  check(M_driver_queue_copy_to_device(queue, &view_x, host_x));
  check(M_driver_queue_copy_to_device(queue, &view_y, host_y));
  struct M_driver_memory_view invalid = {input_x, sizeof(host_x), 1};
  assert(M_driver_queue_copy_to_device(queue, &invalid, host_x) ==
         M_DRIVER_STATUS_INVALID_ARG);
  struct M_driver_memory_view tail = {input_x, 4, 4};
  check(M_driver_queue_copy_from_device(queue, host_out, &tail));
  assert(host_out[0] == host_x[1]);

  struct M_driver_static_bundle bundle = {{image, (uint64_t)size}, "elf", 3};
  struct M_driver_bundle_compilation_options options = {"", 0, 3};
  M_driver_runtime_bundle_t loaded;
  check(M_driver_bundle_load(context, &bundle, &options, &loaded));
  M_driver_function_t function;
  check(M_driver_function_load(context, loaded, argv[2], strlen(argv[2]), &function));
  check(M_driver_bundle_unload(context, loaded));
  free(image);

  uint64_t addr_x, addr_y, addr_out;
  check(M_driver_memory_property(input_x, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &addr_x));
  check(M_driver_memory_property(input_y, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &addr_y));
  check(M_driver_memory_property(output, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &addr_out));
  float alpha = 2.5f;
  int32_t elements = length;
  void *args[] = {&addr_x, &addr_y, &addr_out, &alpha, &elements};
  uint64_t sizes[] = {8, 8, 8, 4, 4};
  M_driver_queue_execute_config_t config = {
      .mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU,
      .config.gpu = {{1, 1, 1}, {1, 1, 1}, 0, NULL, 0},
  };
  config.config.gpu.grid.x = 2;
  assert(M_driver_queue_execute(queue, function, &config, args, sizes, 5) != 0);
  config.config.gpu.grid.x = 1;
  check(M_driver_queue_execute(queue, function, &config, args, sizes, 5));
  M_driver_event_t event;
  check(M_driver_event_create(context, 0, &event));
  check(M_driver_queue_record_event(queue, event));
  bool ready = false;
  check(M_driver_is_event_ready(context, event, &ready));
  assert(ready);
  check(M_driver_queue_copy_from_device(queue, host_out, &view_out));
  for (int index = 0; index < length; ++index)
    assert(host_out[index] == alpha * host_x[index] + host_y[index]);

  uint64_t simulation_count = 0;
  check(M_driver_queue_property(queue, "t1.simulation_count", &simulation_count));
  assert(simulation_count > 0);
  printf("SAXPY OK through upstream HAL: n=%d simulation_count=%llu\n",
         length, (unsigned long long)simulation_count);
  check(M_driver_event_destroy(context, event));
  check(M_driver_function_unload(context, function));
  check(M_driver_memory_free_sync(context, output));
  check(M_driver_memory_free_sync(context, input_y));
  check(M_driver_memory_free_sync(context, input_x));
  check(M_driver_queue_destroy(context, queue));
  check(M_driver_context_destroy(context));
  check(M_driver_destroy(driver));
  return 0;
}
