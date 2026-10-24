M_driver_status_t M_driver_memory_alloc_async(M_driver_queue_t queue,
                                              uint64_t size,
                                              M_driver_memory_t *memory) {
  return M_driver_memory_alloc_sync((M_driver_context_t)queue, size, memory);
}

M_driver_status_t M_driver_memory_free_async(M_driver_queue_t queue,
                                             M_driver_memory_t memory) {
  return M_driver_memory_free_sync((M_driver_context_t)queue, memory);
}

M_driver_status_t M_driver_queue_copy_to_device(M_driver_queue_t queue,
                                                M_driver_memory_view_t dst,
                                                const void *src) {
  return M_driver_copy_to_device_sync((M_driver_context_t)queue, dst, src);
}

M_driver_status_t M_driver_queue_copy_from_device(M_driver_queue_t queue,
                                                  void *dst,
                                                  M_driver_memory_view_t src) {
  return M_driver_copy_from_device_sync((M_driver_context_t)queue, dst, src);
}

M_driver_status_t M_driver_queue_copy_intra_device(M_driver_queue_t queue,
                                                   M_driver_memory_view_t dst,
                                                   M_driver_memory_view_t src) {
  return M_driver_copy_intra_device_sync((M_driver_context_t)queue, dst, src);
}

M_driver_status_t M_driver_queue_set_memory(M_driver_queue_t queue,
                                            M_driver_memory_view_t dst,
                                            uint8_t value) {
  return M_driver_set_memory_sync((M_driver_context_t)queue, dst, value);
}

M_driver_status_t M_driver_queue_fill(M_driver_queue_t queue,
                                      M_driver_memory_view_t dst,
                                      uint64_t value, uint64_t value_size) {
  return M_driver_fill_sync((M_driver_context_t)queue, dst, value, value_size);
}
