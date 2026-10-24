#include "HAL/M_driver_api.h"
#include "simulator.c"

struct M_driver_driver { int device; };
struct M_driver_runtime_bundle { uint8_t *data; size_t size; };
struct M_driver_event { T1Context *context; bool ready; };

static _Thread_local char last_error[1024];

static M_driver_status_t status_from_error(const char *error) {
  if (!error)
    return M_DRIVER_STATUS_SUCCESS;
  snprintf(last_error, sizeof(last_error), "%s", error);
  free((void *)error);
  return M_DRIVER_STATUS_CUSTOM_CODES_BASE;
}

M_driver_status_t M_driver_status_message(M_driver_driver_t driver,
    M_driver_status_t status, char *message_buffer, int64_t message_buffer_size) {
  if (!message_buffer || message_buffer_size <= 0)
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;
  const char *message = status == 0 ? "success" :
      status == M_DRIVER_STATUS_CUSTOM_CODES_BASE ? last_error :
      status == M_DRIVER_STATUS_UNSUPPORTED ? "unsupported on T1" :
      "invalid T1 HAL operation";
  snprintf(message_buffer, (size_t)message_buffer_size, "%s", message);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_create(M_driver_version *version,
                                  M_driver_driver_t *driver) {
  if (!version || !driver)
    return M_DRIVER_STATUS_INVALID_ARG;
  uint32_t requested_major = version->major;
  *version = (M_driver_version){0, 1, 0};
  if (requested_major != version->major)
    return M_DRIVER_STATUS_INCOMPATIBLE_VERSION;
  *driver = calloc(1, sizeof(**driver));
  return *driver ? 0 : M_DRIVER_STATUS_OUT_OF_MEMORY;
}

M_driver_status_t M_driver_destroy(M_driver_driver_t driver) {
  free(driver);
  return 0;
}

M_driver_status_t M_driver_property(M_driver_driver_t driver,
                                    const char *name, void *value) {
  if (!driver || !name || !value)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (!strcmp(name, M_DRIVER_PROPERTY_API))
    *(const char **)value = "t1";
  else if (!strcmp(name, M_DRIVER_PROPERTY_VERSION))
    *(M_driver_version *)value = (M_driver_version){0, 1, 0};
  else
    return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
  return 0;
}

M_driver_status_t M_driver_device_count(M_driver_driver_t driver,
                                        int64_t *count) {
  if (!driver || !count)
    return M_DRIVER_STATUS_INVALID_ARG;
  *count = 1;
  return 0;
}

M_driver_status_t M_driver_device_get(M_driver_driver_t driver,
    int64_t device_id, M_driver_device_t *device) {
  if (!driver || !device)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (device_id != 0)
    return M_DRIVER_STATUS_NO_DEVICES;
  *device = (M_driver_device_t)&driver->device;
  return 0;
}

M_driver_status_t M_driver_device_property(M_driver_device_t device,
    const char *name, void *value) {
  if (!device || !name || !value)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (!strcmp(name, M_DRIVER_DEVICE_PROPERTY_NAME))
    snprintf(value, M_DRIVER_DEVICE_PROPERTY_MAX_LEN, "T1 RVV simulator");
  else if (!strcmp(name, M_DRIVER_DEVICE_PROPERTY_ARCH))
    snprintf(value, M_DRIVER_DEVICE_PROPERTY_MAX_LEN, "t1");
  else if (!strcmp(name, M_DRIVER_DEVICE_PROPERTY_HOST_UNIFIED))
    *(int64_t *)value = 0;
  else if (!strcmp(name, M_DRIVER_DEVICE_PROPERTY_MAX_ALLOCATION_SIZE))
    *(int64_t *)value = T1_ARENA_END - T1_ARENA_BASE;
  else if (!strcmp(name, M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE))
    *(M_driver_dlpack_device *)value = (M_driver_dlpack_device){12, 0};
  else if (!strcmp(name, M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE_PINNED))
    *(M_driver_dlpack_device *)value = (M_driver_dlpack_device){1, 0};
  else
    return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
  return 0;
}

M_driver_status_t M_driver_context_create(M_driver_device_t device,
                                          M_driver_context_t *context) {
  if (!device || !context)
    return M_DRIVER_STATUS_INVALID_ARG;
  return status_from_error(t1_DeviceContext_create(
      (const T1Context **)context, "t1", 0));
}

M_driver_status_t M_driver_context_destroy(M_driver_context_t context) {
  T1Context *ctx = (T1Context *)context;
  if (!ctx)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  M_driver_status_t status = status_from_error(flush_launches(ctx));
  while (ctx->buffers) {
    T1Buffer *buffer = ctx->buffers;
    ctx->buffers = buffer->next;
    if (!buffer->parent)
      free(buffer->shadow);
    free(buffer);
  }
  while (ctx->pending) {
    T1Launch *launch = ctx->pending;
    ctx->pending = launch->next;
    free(launch);
  }
  free(ctx->trampoline);
  free(ctx->workdir);
  free(ctx);
  return status;
}

M_driver_status_t M_driver_context_set_current(M_driver_context_t context) {
  return context ? 0 : M_DRIVER_STATUS_BAD_CONTEXT;
}

M_driver_status_t M_driver_context_memory_info(M_driver_context_t context,
    uint64_t *free_bytes, uint64_t *total) {
  if (!context || !free_bytes || !total)
    return M_DRIVER_STATUS_INVALID_ARG;
  *total = T1_ARENA_END - T1_ARENA_BASE;
  *free_bytes = T1_ARENA_END - ((T1Context *)context)->arena_next;
  return 0;
}

M_driver_status_t M_driver_memory_alloc_sync(M_driver_context_t context,
    uint64_t size, M_driver_memory_t *memory) {
  if (!context || !memory || size > T1_ARENA_END - T1_ARENA_BASE)
    return M_DRIVER_STATUS_INVALID_ARG;
  void *address;
  return status_from_error(t1_DeviceContext_createBuffer_async(
      (const T1Buffer **)memory, &address, (T1Context *)context, size, 1));
}

M_driver_status_t M_driver_memory_alloc_pinned(M_driver_context_t context,
    uint64_t size, M_driver_memory_t *memory) {
  if (!context || !memory || size > SIZE_MAX)
    return M_DRIVER_STATUS_INVALID_ARG;
  void *address;
  return status_from_error(t1_DeviceContext_createHostBuffer(
      (const T1Buffer **)memory, &address, (T1Context *)context, size, 1));
}

M_driver_status_t M_driver_memory_free_sync(M_driver_context_t context,
                                            M_driver_memory_t memory) {
  T1Buffer *buffer = (T1Buffer *)memory;
  T1Context *ctx = (T1Context *)context;
  if (!buffer || buffer->ctx != ctx)
    return M_DRIVER_STATUS_BAD_MEMORY;
  M_driver_status_t status = status_from_error(flush_launches(ctx));
  if (status)
    return status;
  T1Buffer **link = &ctx->buffers;
  while (*link && *link != buffer)
    link = &(*link)->next;
  if (!*link)
    return M_DRIVER_STATUS_BAD_MEMORY;
  *link = buffer->next;
  if (!buffer->parent)
    free(buffer->shadow);
  free(buffer);
  if (!ctx->buffers)
    ctx->arena_next = T1_ARENA_BASE;
  return 0;
}

M_driver_status_t M_driver_memory_free_pinned(M_driver_context_t context,
                                              M_driver_memory_t memory) {
  return M_driver_memory_free_sync(context, memory);
}

M_driver_status_t M_driver_memory_property(M_driver_memory_t memory,
    const char *name, void *value) {
  T1Buffer *buffer = (T1Buffer *)memory;
  if (!buffer || !name || !value)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (!strcmp(name, M_DRIVER_MEMORY_PROPERTY_ADDRESS))
    *(uint64_t *)value = buffer->device_resident ? buffer->device_addr :
        (uintptr_t)buffer->shadow;
  else if (!strcmp(name, M_DRIVER_MEMORY_PROPERTY_HOST_ADDRESS)) {
    if (buffer->device_resident)
      return M_DRIVER_STATUS_UNSUPPORTED;
    *(void **)value = buffer->shadow;
  } else
    return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
  return 0;
}

static bool valid_view(T1Context *context, M_driver_memory_view_t view) {
  if (!view || !view->memory)
    return false;
  T1Buffer *buffer = (T1Buffer *)view->memory;
  return buffer->ctx == context && view->offset <= buffer->bytes &&
      view->size <= buffer->bytes - view->offset;
}

M_driver_status_t M_driver_copy_to_device_sync(M_driver_context_t context,
    M_driver_memory_view_t dst, const void *src) {
  if (!src || !valid_view((T1Context *)context, dst))
    return M_DRIVER_STATUS_INVALID_ARG;
  M_driver_status_t status = status_from_error(flush_launches((T1Context *)context));
  if (!status)
    memcpy(((T1Buffer *)dst->memory)->shadow + dst->offset, src, dst->size);
  return status;
}

M_driver_status_t M_driver_copy_from_device_sync(M_driver_context_t context,
    void *dst, M_driver_memory_view_t src) {
  if (!dst || !valid_view((T1Context *)context, src))
    return M_DRIVER_STATUS_INVALID_ARG;
  M_driver_status_t status = status_from_error(flush_launches((T1Context *)context));
  if (!status)
    memcpy(dst, ((T1Buffer *)src->memory)->shadow + src->offset, src->size);
  return status;
}

M_driver_status_t M_driver_copy_intra_device_sync(M_driver_context_t context,
    M_driver_memory_view_t dst, M_driver_memory_view_t src) {
  if (!valid_view((T1Context *)context, dst) ||
      !valid_view((T1Context *)context, src) || dst->size != src->size)
    return M_DRIVER_STATUS_INVALID_ARG;
  M_driver_status_t status = status_from_error(flush_launches((T1Context *)context));
  if (!status)
    memmove(((T1Buffer *)dst->memory)->shadow + dst->offset,
            ((T1Buffer *)src->memory)->shadow + src->offset, src->size);
  return status;
}

M_driver_status_t M_driver_fill_sync(M_driver_context_t context,
    M_driver_memory_view_t dst, uint64_t value, uint64_t value_size) {
  if (!valid_view((T1Context *)context, dst) || !value_size ||
      value_size > 8 || dst->size % value_size)
    return M_DRIVER_STATUS_INVALID_ARG;
  M_driver_status_t status = status_from_error(flush_launches((T1Context *)context));
  if (status)
    return status;
  uint8_t *data = ((T1Buffer *)dst->memory)->shadow + dst->offset;
  for (uint64_t offset = 0; offset < dst->size; offset += value_size)
    memcpy(data + offset, &value, value_size);
  return 0;
}

M_driver_status_t M_driver_set_memory_sync(M_driver_context_t context,
    M_driver_memory_view_t dst, uint8_t value) {
  return M_driver_fill_sync(context, dst, value, 1);
}

M_driver_status_t M_driver_queue_create(M_driver_context_t context,
                                        M_driver_queue_t *queue) {
  if (!context || !queue)
    return M_DRIVER_STATUS_INVALID_ARG;
  *queue = (M_driver_queue_t)context;
  return 0;
}

M_driver_status_t M_driver_queue_synchronize(M_driver_queue_t queue) {
  if (!queue)
    return M_DRIVER_STATUS_BAD_QUEUE;
  return status_from_error(flush_launches((T1Context *)queue));
}

M_driver_status_t M_driver_queue_destroy(M_driver_context_t context,
                                         M_driver_queue_t queue) {
  if ((void *)context != (void *)queue)
    return M_DRIVER_STATUS_BAD_QUEUE;
  return M_driver_queue_synchronize(queue);
}

M_driver_status_t M_driver_queue_is_stream(M_driver_queue_t queue,
                                           bool *is_stream) {
  if (!queue || !is_stream)
    return M_DRIVER_STATUS_INVALID_ARG;
  *is_stream = true;
  return 0;
}

M_driver_status_t M_driver_bundle_load(M_driver_context_t context,
    M_driver_static_bundle_t bundle, M_driver_bundle_compilation_options_t opts,
    M_driver_runtime_bundle_t *result) {
  if (!context || !bundle || !result || bundle->mapped_data.size < 52 ||
      memcmp(bundle->mapped_data.data, "\177ELF\1\1", 6))
    return M_DRIVER_STATUS_BAD_BUNDLE;
  *result = calloc(1, sizeof(**result));
  if (!*result)
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  (*result)->size = bundle->mapped_data.size;
  (*result)->data = malloc((*result)->size);
  if (!(*result)->data) {
    free(*result);
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  }
  memcpy((*result)->data, bundle->mapped_data.data, (*result)->size);
  return 0;
}

M_driver_status_t M_driver_bundle_unload(M_driver_context_t context,
                                         M_driver_runtime_bundle_t bundle) {
  if (!bundle)
    return M_DRIVER_STATUS_BAD_BUNDLE;
  free(bundle->data);
  free(bundle);
  return 0;
}

M_driver_status_t M_driver_function_load(M_driver_context_t context,
    M_driver_runtime_bundle_t bundle, const char *name, uint64_t name_len,
    M_driver_function_t *function) {
  if (!context || !bundle || !name || !function)
    return M_DRIVER_STATUS_INVALID_ARG;
  char *terminated = strndup(name, name_len);
  if (!terminated)
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  const char *error = t1_DeviceContext_loadFunction(
      (const T1Function **)function, (T1Context *)context, "",
      terminated, (const char *)bundle->data, bundle->size, 0, "", 3);
  free(terminated);
  return status_from_error(error);
}

M_driver_status_t M_driver_function_unload(M_driver_context_t context,
                                           M_driver_function_t function) {
  if (!function)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  M_driver_status_t status = status_from_error(flush_launches((T1Context *)context));
  if (status)
    return status;
  free(((T1Function *)function)->elf);
  free(function);
  return 0;
}

M_driver_status_t M_driver_queue_execute(M_driver_queue_t queue,
    M_driver_function_t function, M_driver_queue_execute_config_t *config,
    void **args, uint64_t *arg_sizes, uint32_t num_args) {
  if (!queue || !function || !config || (num_args && (!args || !arg_sizes)))
    return M_DRIVER_STATUS_INVALID_ARG;
  if (((T1Function *)function)->ctx != (T1Context *)queue)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  if (config->mode != M_DRIVER_QUEUE_EXECUTE_MODE_GPU)
    return M_DRIVER_STATUS_EXECUTE_MODE_NOT_SUPPORTED;
  for (uint32_t index = 0; index < num_args; ++index)
    if (!args[index] || (arg_sizes[index] != 4 && arg_sizes[index] != 8))
      return M_DRIVER_STATUS_UNSUPPORTED;
  M_driver_queue_execute_config_gpu_t *gpu = &config->config.gpu;
  return status_from_error(t1_DeviceContext_enqueueFunctionDirect(
      (T1Context *)queue, (T1Function *)function, gpu->grid.x, gpu->grid.y,
      gpu->grid.z, gpu->block.x, gpu->block.y, gpu->block.z,
      gpu->shared_mem_bytes, gpu->attributes, gpu->num_attributes,
      args, num_args, arg_sizes));
}

M_driver_status_t M_driver_event_create(M_driver_context_t context,
    M_driver_event_flags_t flags, M_driver_event_t *event) {
  if (!context || !event)
    return M_DRIVER_STATUS_INVALID_ARG;
  *event = calloc(1, sizeof(**event));
  if (!*event)
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  (*event)->context = (T1Context *)context;
  return 0;
}

M_driver_status_t M_driver_event_destroy(M_driver_context_t context,
                                         M_driver_event_t event) {
  if (!event || event->context != (T1Context *)context)
    return M_DRIVER_STATUS_INVALID_ARG;
  free(event);
  return 0;
}

M_driver_status_t M_driver_queue_record_event(M_driver_queue_t queue,
                                              M_driver_event_t event) {
  if (!event || event->context != (T1Context *)queue)
    return M_DRIVER_STATUS_INVALID_ARG;
  M_driver_status_t status = M_driver_queue_synchronize(queue);
  if (!status)
    event->ready = true;
  return status;
}

M_driver_status_t M_driver_event_synchronize(M_driver_context_t context,
                                             M_driver_event_t event) {
  if (!event || event->context != (T1Context *)context)
    return M_DRIVER_STATUS_INVALID_ARG;
  return event->ready ? 0 : M_DRIVER_STATUS_INVALID_ARG;
}

M_driver_status_t M_driver_is_event_ready(M_driver_context_t context,
    M_driver_event_t event, bool *ready) {
  if (!event || !ready || event->context != (T1Context *)context)
    return M_DRIVER_STATUS_INVALID_ARG;
  *ready = event->ready;
  return 0;
}

M_driver_status_t M_driver_queue_wait_for_events(M_driver_queue_t queue,
    M_driver_event_t *events, uint32_t num_events) {
  if (!queue || (num_events && !events))
    return M_DRIVER_STATUS_INVALID_ARG;
  for (uint32_t index = 0; index < num_events; ++index) {
    M_driver_status_t status = M_driver_event_synchronize(
        (M_driver_context_t)queue, events[index]);
    if (status)
      return status;
  }
  return 0;
}

M_driver_status_t M_driver_queue_launch_host_func(M_driver_queue_t queue,
    void (*function)(void *), void *user_data) {
  if (!function)
    return M_DRIVER_STATUS_INVALID_ARG;
  M_driver_status_t status = M_driver_queue_synchronize(queue);
  if (!status)
    function(user_data);
  return status;
}

M_driver_status_t M_driver_queue_property(M_driver_queue_t queue,
    const char *name, void *value) {
  if (!queue || !name || !value)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (strcmp(name, "t1.simulation_count"))
    return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
  M_driver_status_t status = M_driver_queue_synchronize(queue);
  if (!status)
    *(uint64_t *)value = ((T1Context *)queue)->total_cycles;
  return status;
}

#include "driver_forwarders.inc"
#include "driver_unsupported.inc"
