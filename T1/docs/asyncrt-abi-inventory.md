# T1 as a Mojo `DeviceContext` backend — AsyncRT ABI symbol inventory and semantics

**Scope**: all **111** `AsyncRT_*` symbols referenced by `external_call` in
`max/mojo/max/gpu/host/*.mojo`. Implementing a subset is enough to make Mojo's `DeviceContext` work
on T1; the rest are stubbed per the policy column in these tables.

**Source**: symbols and C signatures are transcribed from the declaration comments in the
open-source Mojo bindings. The closed-source implementation lives in `MojoBindings.cpp` (named in
the comment at `device_context.mojo:3856`).

**Error convention**: functions returning `const char *` return `NULL`/`""` on success and a
NUL-terminated error string on failure, which the caller frees with
`AsyncRT_DeviceContext_strfree`. "Not implemented" is therefore the one-liner
`return "unsupported on T1";`.

---

## 0. Tier legend

| Tier | Meaning |
|---|---|
| **CORE** | nothing runs without it |
| **COH** | must implement, but the **semantics are redefined**: a copy on GPUs, coherence maintenance on T1 |
| **PERF** | semantics are **stronger** on T1 than on GPUs (cycle-accurate); implement early |
| **MSP** | maps to the future `block_dim`/`grid_dim` abstraction; stub now but **reserve the semantics** |
| **DEGEN** | a degenerate implementation (e.g. single stream) suffices; must not error — generic paths call it |
| **STUB** | return unsupported; does not affect the main path |
| **REJECT** | must **error out explicitly**; silently ignoring produces wrong results |
| **N/A** | vendor-specific escape hatch (cuda/hip/metal); never implement |

---

## 1. Reserving the semantics of `grid_dim` / `block_dim`

Aligned with the Cray X1 MSP (Multi-Streaming Processor) model — one MSP is built from 4 SSPs, each
SSP holding a scalar unit and a vector pipeline; the compiler multistreams one loop across the 4
SSPs, which share the Ecache:

| CUDA concept | Hardware semantics | T1 counterpart (planned) |
|---|---|---|
| `block_dim` | a group of threads sharing shared memory, barrier-synchronizable, running on one SM | **multiple threads inside one T1** (MSP-style multistreaming), sharing T1-internal storage |
| `grid_dim` | a group of blocks, mutually independent, no synchronization guarantee, scheduled across SMs | **multiple T1s on the bus** in parallel |
| `shared_mem_bytes` | dynamic shared memory per block | shared scratchpad between threads inside one T1 |
| `cluster_dim` | Hopper block cluster | undecided — keep as REJECT |

**This mapping is an isomorphism**: synchronizable/sharing inside a block, independent with no
guarantees across blocks — exactly "multiple threads inside a T1" versus "multiple T1s on the bus".
So the CUDA launch geometry is **not dead weight on T1; it is the future interface**.

**Current-stage handling**: when `enqueueFunctionDirect` receives non-all-ones grid/block dims, it
**errors out explicitly** instead of ignoring them. Rationale: a kernel written for
`grid_dim=1024` assumes 1024 copies are running; silently running one produces wrong results — the
hardest failure mode to chase in RTL simulation. But the **internal data structures should carry
all six components from day one**, so the MSP stage does not have to rewrite everything above the
ABI.

The same mapping moves several symbol families below from "never implement" to "implement at the
MSP stage": peer access, multicast, `enqueue_wait_for_context`, `isCompatible`, `occupancy*`.

---

## 2. Context lifecycle — 7 symbols

| Symbol | Signature | Tier | T1 semantics |
|---|---|---|---|
| `_create` | `const char *(const DeviceContext **result, const char *api, int id)` | **CORE** | `api="t1"`; `id` = T1 index on the bus (used at the MSP stage) |
| `_retain` | `void (const DeviceContext *)` | **CORE** | refcount |
| `_release` | `void (const DeviceContext *)` | **CORE** | refcount |
| `_synchronize` | `const char *(const DeviceContext *)` | **CORE** | drain the doorbell queue, wait for the last job |
| `_strfree` | `void (const char *)` | **CORE** | free strings handed out by this ABI |
| `_setAsCurrent` | *(signature TBV)* | DEGEN | thread-local "current device"; single-T1: no-op success |
| `_runHealthcheck` | `const char *(DeviceContext *)` | recommended | ping the transport. **Very useful for verilator/FPGA link-drop diagnosis** |

> `with DeviceContext() as ctx` goes through the Mojo-side `__enter__` (`:4000`) which just returns
> self — **no C call is made**. `DeviceContextScope_*` is only used by an explicit
> `push_context()` → STUB.

---

## 3. Device identity and attributes — 10 symbols

| Symbol | Signature | Tier | T1 semantics |
|---|---|---|---|
| `_deviceApi` | `void (llvm::StringRef *result, const DeviceContext *)` | **CORE** | returns `"t1"` — **this is what encoder dispatch keys off** (see §9) |
| `_archName` | `void (llvm::StringRef *result, const DeviceContext *)` | **CORE** | suggest `"t1-vlen4096-dlen256"` |
| `_deviceName` | `const char *(const DeviceContext *)` | **CORE** | human-readable name; caller strfrees |
| `_id` | `int64_t (const DeviceContext *)` | **CORE** | device ordinal |
| `_numberOfDevices` | `int32_t *(const char *kind)` | **CORE** | `kind` is `"t1"`; at the MSP stage returns the number of T1s on the bus |
| `_getMemoryInfo` | `const char *(const DeviceContext *, size_t *free, size_t *total)` | implement | free/total of the shared window |
| `_maxSingleAllocationSize` | *(signature TBV)* | implement | = window size |
| `_getApiVersion` | *(signature TBV)* | STUB | return your RT version |
| `_computeCapability` | `const char *(int32_t *result, const DeviceContext *)` | STUB | GPU concept; could return the T1 config version |
| `_getAttribute` | *(signature TBV)* | partial | return values for meaningful attributes, error on the rest |

The `llvm::StringRef` out-param is layout-compatible with
`struct { const char *data; size_t len; }` and with Mojo's `StaticString`.

---

## 4. Memory — 12 symbols (CORE + COH)

| Symbol | Signature | Tier | T1 semantics |
|---|---|---|---|
| `_createBuffer_async` | `const char *(const DeviceBuffer **result, void **device_ptr, const DeviceContext *, size_t len, size_t elem_size)` | **CORE** | allocate from the shared window; `device_ptr` = accelerator-visible address |
| `_createHostBuffer` | same as above | **CORE** | host-directly-accessible allocation. **Keep it distinct** from the above so residency tracking knows which ranges need maintenance at doorbell boundaries |
| `_createBuffer_owning` | `void (const DeviceBuffer **result, const DeviceContext *, void *device_ptr, size_t len, size_t elem_size, bool owning)` | **CORE** | wrap an existing address as a buffer; zero-copy takeover of external memory |
| `DeviceBuffer_retain` | `void (const DeviceBuffer *)` | **CORE** | |
| `DeviceBuffer_release` | `void (const DeviceBuffer *)` | **CORE** | |
| `DeviceBuffer_release_ptr` | `void (const DeviceBuffer *)` | **CORE** | release without reclaiming the underlying address |
| `DeviceBuffer_bytesize` | `int64_t (const DeviceBuffer *)` | **CORE** | |
| `DeviceBuffer_context` | *(signature TBV)* | **CORE** | look up the owning context |
| `DeviceBuffer_createSubBuffer` | `const char *(const DeviceBuffer **result, const DeviceBuffer *buf, size_t offset, size_t len, size_t elem_size)` | **CORE** | zero-copy slice |
| `_HtoD_async` | `const char *(const DeviceContext *, const DeviceBuffer *dst, const void *src)` | **COH** | **note: no length parameter** — the size travels with the buffer. Over a shared window, implement as **clean/flush**, not memcpy |
| `_DtoH_async` | `const char *(const DeviceContext *, void *dst, const DeviceBuffer *src)` | **COH** | implement as **invalidate** |
| `_DtoD_async` / `_DtoD_async_no_cross_stream_sync` | *(signature TBV)* | implement | memmove within the window |
| `_setMemory_async` | `const char *(const DeviceContext *, const DeviceBuffer *dst, uint64_t val, size_t val_size)` | implement | memset |
| `DeviceBuffer_reassignOwnershipTo` | *(signature TBV)* | STUB | cross-context handover; revisit at the MSP stage |

> **This is the most important semantic rewrite in the whole port.** Implementing `HtoD`/`DtoH` as
> coherence maintenance instead of copies makes `DeviceBuffer.map_to_host()` automatically cheap —
> its generic implementation (`device_context.mojo:6914-6922`) is "allocate a HostBuffer + D2H +
> 3 × synchronize + H2D", a disaster on verilator. The fix lives entirely in the runtime;
> **no Mojo code needs touching**.

---

## 5. Code loading — 6 symbols

| Symbol | Signature | Tier | T1 semantics |
|---|---|---|---|
| `_loadFunction` | `const char *(const DeviceFunction **result, const DeviceContext *, const char *moduleName, const char *functionName, const char *data, size_t dataLen, int32_t maxDynamicSharedBytes, const char *debugLevel, int32_t optimizationLevel)` | **CORE** | `data` = `CompiledFunctionInfo.asm` (object or flat binary); write into the instruction window and resolve the entry address |
| `DeviceFunction_retain` | `void (const DeviceFunction *)` | **CORE** | |
| `DeviceFunction_release` | `void (const DeviceFunction *)` | **CORE** | |
| `DeviceFunction_getAttribute` | `const char *(int32_t *result, const DeviceFunction *, int32_t attr_code)` | STUB | GPU function attributes |
| `DeviceFunction_copyToConstantMemory` | `const char *(const DeviceFunction *, const void *name, size_t nameSize, const void *data, size_t dataSize)` | STUB | stub initially; could map to an immediate/constant table if the T1 scalar frontend has one |
| `DeviceFunction_cuda_module` / `_hip_module` | — | **N/A** | vendor escape hatches |

`maxDynamicSharedBytes` is a GPU concept → accept only `-1`/`0` for now; at the MSP stage it
becomes "size of the inter-thread shared scratchpad".

---

## 6. Launch — 5 symbols

| Symbol | Signature | Tier | T1 semantics |
|---|---|---|---|
| `_enqueueFunctionDirect` | `const char *(ctx, func, uint32 gx,gy,gz, uint32 bx,by,bz, uint32 sharedMem, LaunchAttribute*, uint32 numAttrs, void **args, uint32 argCount, uint64 *argSizes)` | **CORE** | see §1; flush the residency set → doorbell |
| `DeviceStream_enqueueFunctionDirect` | *(TBV, same shape)* | DEGEN | forward to the above under a single stream |
| `_enqueueHostFunction` | *(signature TBV)* | recommended | insert a host callback into the queue — **useful for pipelining against verilator** |
| `_enqueueHostFunctionRange` | *(signature TBV)* | recommended | same, batched |
| `_occupancyMaxActiveBlocksPerMultiprocessor` | `const char *(int *numBlocks, const DeviceContext *, const DeviceFunction *, int blockSize, size_t dynamicSharedMemSize)` | **MSP** | return 1 for now; at the MSP stage = how many threads one T1 can run concurrently |

`args` holds `argCount` pointers to the argument **values**; `argSizes[i]` is each one's width.
Explicit arguments come first, closure captures written by the compiler-generated `populate` come
after; zero-size captures are already compacted out on the Mojo side (`device_context.mojo:3145`).

---

## 7. Streams — 13 symbols (all DEGEN)

With a single T1 queue, implement streams as an alias of the context — **but they must not error**:
the generic path calls `ctx.stream()`.

| Symbol | Tier | Note |
|---|---|---|
| `_stream` `const char *(const DeviceStream **result, const DeviceContext *)` | DEGEN | return the degenerate singleton stream |
| `_createStream` `const char *(const DeviceStream **, int priority, const DeviceContext *)` | DEGEN | ignore priority |
| `_selectStream` `const char *(..., unsigned int stream_id)` | DEGEN | accept only 0 |
| `_numStreams` `int (const DeviceContext *)` | DEGEN | return 1 |
| `_streamPriorityRange` `const char *(int *least, int *greatest, const DeviceContext *)` | DEGEN | return 0,0 |
| `_createExternalStream` `const char *(const DeviceStream **, void *externalStream, const DeviceContext *)` | STUB | |
| `DeviceStream_retain` / `_release` | DEGEN | |
| `DeviceStream_synchronize` `const char *(const DeviceStream *)` | DEGEN | forward to context synchronize |
| `DeviceStream_eventRecord` `const char *(const DeviceStream *, const DeviceEvent *)` | **PERF** | see §8 |
| `DeviceStream_waitForEvent` `const char *(const DeviceStream *, const DeviceEvent *)` | **PERF** | |
| `DeviceStream_enqueueHostFunc` `const char *(const DeviceStream *, void (*fn)(void *), void *userData)` | recommended | |
| `DeviceStream_enqueueWaitOnHostValue` `const char *(const DeviceStream *, CompletionFlag *flag, uint64_t value)` | STUB | pairs with `CompletionFlag_devicePtr` |
| `CompletionFlag_devicePtr` `int64_t (const CompletionFlag *)` | STUB | |

---

## 8. Events and timing — 8 symbols (**PERF: stronger semantics on T1 than on GPUs**)

| Symbol | Tier | T1 semantics |
|---|---|---|
| `_eventCreate` `const char *(const DeviceEvent **result, const DeviceContext *, unsigned int flags)` | **PERF** | |
| `_enqueue_event` `const char *(const DeviceEvent **result, const DeviceContext *)` | **PERF** | |
| `DeviceEvent_retain` / `_release` | **PERF** | |
| `DeviceEvent_synchronize` `const char *(const DeviceEvent *)` | **PERF** | |
| `_startTimer` / `_stopTimer` | **PERF** | |
| `DeviceTimer_release` `void (const DeviceTimer *)` | **PERF** | |

> **This family is worth implementing early, and it should return exact cycle counts, not
> wall-clock time.** On a GPU, event timing can only observe wall-clock; on T1
> (verilator/ncsim/Palladium) the same interface can return **cycle-accurate** results. This is a
> structural advantage of T1 over GPUs, and it is the precondition for hooking compile-time
> autotune up to cycle feedback (see "to be validated").

---

## 9. Device Graph — 32 symbols (STUB, but high long-term value)

`DeviceGraphBuilder_{add, addFunctionDirect, addEmpty, addCopyHostToDevice, addCopyDeviceToHost,
addCopyDeviceToDevice, addSetMemory, addInput, addInPlaceInput, addOutput, numInputs, numOutputs,
lastNodeIdOrNone, recordingContext, instantiate, release}`
`DeviceGraph_{createBuffer, replay, retain, release}`
`DeviceContext_createGraphBuilder`

The CUDA Graph equivalent: record a sequence of operations into a graph, then replay it wholesale,
eliminating per-submission overhead.

**All STUB initially. But note: this family is worth far more on verilator than it is on GPUs** —
it compresses N host↔simulator round-trips into 1. When a single round-trip costs seconds, that is
an order-of-magnitude win. Recommended as the first priority after CORE.

---

## 10. Multi-device: peer / multicast — 8 symbols (**MSP**)

Under the §1 mapping this family corresponds to "multiple T1s on the bus" — it is not
"never implement".

| Symbol | Tier | MSP-stage semantics |
|---|---|---|
| `_canAccess` `const char *(bool *result, const DeviceContext *, const DeviceContext *peer)` | MSP | can T1-A reach T1-B's memory |
| `_enablePeerAccess` `const char *(const DeviceContext *, const DeviceContext *peer)` | MSP | |
| `_enableAllPeerAccess` `const char *()` | MSP | |
| `_allPeerAccessEnabled` `const char *(bool *result)` | MSP | |
| `_isCompatible` *(signature TBV)* | MSP | are two T1 configs interoperable (matching VLEN/DLEN) |
| `_enqueue_wait_for_context` *(signature TBV)* | MSP | cross-T1 synchronization |
| `_supportsMulticast` `const char *(bool *result, const DeviceContext *)` | MSP | |
| `DeviceMulticastBuffer_{allocate, multicastBufferFor, unicastBufferFor}` | MSP | broadcast one payload to several T1s (allreduce-style) |

---

## 11. Vendor escape hatches — 9 symbols (**N/A**)

`_cuda_context`, `_cuda_current_context`, `AsyncRT_cuda_tensorMapEncodeTiled`,
`AsyncRT_cuda_tensorMapEncodeIm2col`, `_hip_device`, `_metal_device`,
`_setMetalPrintEnabled`, `_startMetalTraceCapture`, `_stopMetalTraceCapture`

Never implement. All `return "unsupported on T1";`

`DeviceContextScope_create` / `_release` — STUB (only used by `push_context()`, not on the
`with DeviceContext()` path).

---

## 12. Summary

| Tier | Count | Notes |
|---|---|---|
| CORE | 22 | nothing runs without them |
| COH | 4 | semantic rewrite: copy → coherence maintenance |
| DEGEN | 11 | degenerate implementations; must not error |
| PERF | 8 | implement early; cycle-accurate |
| recommended | 6 | healthcheck, host callbacks, memory info, … |
| MSP | 9 | semantics reserved, stubbed for now |
| STUB | 42 | includes the 32 Graph symbols |
| N/A | 9 | vendor-specific |

**Milestone 1 = CORE (22) + COH (4) + DEGEN (11) = 37 symbols**, enough to run
"create context → allocate buffers → load kernel → launch → synchronize → read back".

---

## 13. To be validated / decisions needed

1. **Symbols with unverified signatures** (marked *(signature TBV)* above): ~14 have only call
   sites and no one-line declaration comment; their prototypes must be inferred from the
   `external_call` argument types. This does not affect the tiering, only the prototypes at
   implementation time.
2. **Current behavior of `grid_dim`**: REJECT (error out), or accept `gridDim.x=N` as "the scalar
   frontend re-executes N times"? Depends on whether the scalar frontend can run an outer loop
   with an incrementing index.
3. **`getAttribute` attribute-code table**: which of the GPU attribute enums (max threads per
   block, …) are meaningful for T1; a T1-side mapping needs to be defined.
4. **Autotune and cycle feedback**: the evaluator of `kgen.param.evaluate` is user-written Mojo
   code that receives the candidate functions' **addresses** and returns the chosen index. If the
   evaluator does not call the candidates on the host but instead ships them into the simulator
   via the §8 event interface and reads back cycle counts, the mechanism is self-consistent.
   **A minimal experiment is needed**: when the candidates are JITed by `ExecutionEngine`, does
   the mere presence of RVV instructions crash on the host (even if never executed)? Cheap to
   check, high value; do it before implementing.
