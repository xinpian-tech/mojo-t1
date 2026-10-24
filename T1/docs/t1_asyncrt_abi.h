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
 * t1_asyncrt_abi.h — AsyncRT C ABI surface required to make T1 a first-class
 *                    Mojo `DeviceContext` backend.
 *
 * Implement these symbols in libT1RT.so and link it in place of libmax.so.
 * Mojo's `max.gpu.host.DeviceContext` then works unmodified against T1.
 *
 * PROVENANCE
 *   Every signature below is transcribed from the C declaration comments that
 *   sit above each `external_call` in the open-source Mojo bindings:
 *     max/mojo/max/gpu/host/device_context.mojo
 *   The closed-source implementation lives in `MojoBindings.cpp` (referenced
 *   by name in a comment at device_context.mojo:3856). Line references below
 *   point at the Mojo binding that calls each symbol.
 *
 * ERROR CONVENTION
 *   Functions returning `const char *` return NULL (or "") on success and a
 *   NUL-terminated error message on failure. The caller frees it via
 *   AsyncRT_DeviceContext_strfree. This makes "not implemented" trivial to
 *   express, so the ~95 symbols NOT in this header can each be a one-liner
 *   returning "unsupported on T1".
 *
 * OWNERSHIP
 *   Handles are opaque. Mojo declares them as empty structs
 *   (device_context.mojo:103/107/111), so they may point at anything you like.
 *
 * STRING OUT-PARAMS
 *   A few entry points return a string through an `llvm::StringRef` out-param,
 *   which is layout-compatible with `struct { const char *data; size_t len; }`
 *   and with Mojo's `StaticString`. Those are marked below.
\*===--------------------------------------------------------------------===*/

#ifndef T1_ASYNCRT_ABI_H
#define T1_ASYNCRT_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*--- Opaque handles -------------------------------------------------------*/

typedef struct DeviceContext  DeviceContext;
typedef struct DeviceBuffer   DeviceBuffer;
typedef struct DeviceFunction DeviceFunction;

/* Layout-compatible with llvm::StringRef and Mojo's StaticString. */
typedef struct T1StringRef {
  const char *data;
  size_t      len;
} T1StringRef;

/*==========================================================================*\
 * 1. Context lifecycle
\*==========================================================================*/

/* device_context.mojo:3916 — `api` is the device-kind selector; return
 * "t1" from AsyncRT_DeviceContext_deviceApi so the Mojo-side dispatch in
 * DeviceFunction._call_with_pack (device_context.mojo:3156) can branch on it.
 * Mojo calls this from `DeviceContext(device_id=0, api="t1")`. */
const char *AsyncRT_DeviceContext_create(const DeviceContext **result,
                                         const char *api, int id);

/* device_context.mojo:3937 / :3993 — refcounting. */
void AsyncRT_DeviceContext_retain(const DeviceContext *ctx);
void AsyncRT_DeviceContext_release(const DeviceContext *ctx);

/* device_context.mojo:5982 — block until all enqueued work has retired.
 * On T1 this drains the doorbell queue and waits for the last job. */
const char *AsyncRT_DeviceContext_synchronize(const DeviceContext *ctx);

/* device_context.mojo:206 — frees strings handed back across this ABI. */
void AsyncRT_DeviceContext_strfree(const char *ptr);

/*==========================================================================*\
 * 2. Device identity
\*==========================================================================*/

/* device_context.mojo:4085 — out-param is an llvm::StringRef. Return "t1". */
void AsyncRT_DeviceContext_deviceApi(T1StringRef *result,
                                     const DeviceContext *ctx);

/* device_context.mojo:6312 — out-param is an llvm::StringRef.
 * Suggested: the T1 configuration string, e.g. "t1-vlen4096-dlen256". */
void AsyncRT_DeviceContext_archName(T1StringRef *result,
                                    const DeviceContext *ctx);

/* device_context.mojo:4043 — human-readable name; caller frees via strfree. */
const char *AsyncRT_DeviceContext_deviceName(const DeviceContext *ctx);

/* device_context.mojo:6258 */
int64_t AsyncRT_DeviceContext_id(const DeviceContext *ctx);

/* device_context.mojo:6562 — `kind` matches the `api` string ("t1"). */
int32_t *AsyncRT_DeviceContext_numberOfDevices(const char *kind);

/*==========================================================================*\
 * 3. Memory
 *
 * Both allocators hand back TWO things: an owning DeviceBuffer handle and the
 * raw address the kernel will see (`device_ptr`). On T1 the raw address is a
 * physical address inside the window that the x86 host and the accelerator
 * both map. NOTE: T1 is RV32 — device addresses are 32-bit while these host
 * signatures carry 64-bit pointers. Keep the window below 4 GiB (or identity
 * mapped) so the low 32 bits of a host pointer ARE the device address; see
 * the "64→32 slot convention" in minimal-kernel-plan.md §C1.
 *
 * `createBuffer_async` vs `createHostBuffer` is the accelerator-resident vs
 * host-visible distinction (Metal uses the same split for unified memory).
 * On T1 both come from the shared window; keep them distinct anyway so the
 * residency tracker knows which ranges need coherence maintenance at a
 * doorbell boundary and which do not.
\*==========================================================================*/

/* device_context.mojo:1488 */
const char *AsyncRT_DeviceContext_createBuffer_async(const DeviceBuffer **result,
                                                     void **device_ptr,
                                                     const DeviceContext *ctx,
                                                     size_t len,
                                                     size_t elem_size);

/* device_context.mojo:374 */
const char *AsyncRT_DeviceContext_createHostBuffer(const DeviceBuffer **result,
                                                   void **device_ptr,
                                                   const DeviceContext *ctx,
                                                   size_t len,
                                                   size_t elem_size);

/* device_context.mojo:465 */
void AsyncRT_DeviceBuffer_release(const DeviceBuffer *buffer);

/* device_context.mojo:5266 / :5333 — note there is no length argument: the
 * transfer size is carried by the DeviceBuffer. Implement these as coherence
 * maintenance (clean / invalidate over the buffer's range) rather than as
 * memcpy whenever the window is genuinely shared; that is what makes
 * DeviceBuffer.map_to_host() cheap without touching any Mojo code. */
const char *AsyncRT_DeviceContext_HtoD_async(const DeviceContext *ctx,
                                             const DeviceBuffer *dst,
                                             const void *src);
const char *AsyncRT_DeviceContext_DtoH_async(const DeviceContext *ctx,
                                             void *dst,
                                             const DeviceBuffer *src);

/*==========================================================================*\
 * 4. Code loading
\*==========================================================================*/

/* device_context.mojo:3673 —
 *   `data`/`dataLen` is whatever CompiledFunctionInfo.asm holds; with
 *   emission_kind="object" that is a relocatable ELF, and with a T1
 *   TargetBackend::createArchive override it can be a flat binary instead.
 *   `functionName` is the mangled entry symbol.
 *   `maxDynamicSharedBytes` is a GPU concept: reject anything but -1/0.
 *   `debugLevel` / `optimizationLevel` may be recorded for diagnostics.
 *
 * T1 implementation: place the code in the instruction window
 * (transport write_mem) and record the resolved entry address. */
const char *AsyncRT_DeviceContext_loadFunction(const DeviceFunction **result,
                                               const DeviceContext *ctx,
                                               const char *moduleName,
                                               const char *functionName,
                                               const char *data,
                                               size_t dataLen,
                                               int32_t maxDynamicSharedBytes,
                                               const char *debugLevel,
                                               int32_t optimizationLevel);

/* device_context.mojo:2835 */
void AsyncRT_DeviceFunction_release(const DeviceFunction *ctx);

/*==========================================================================*\
 * 5. Launch
 *
 * `args` is an array of `argCount` pointers to argument VALUES, with
 * `argSizes[i]` giving each value's width. Explicit kernel arguments come
 * first, then the closure captures written by the compiler-generated
 * `populate` function; zero-sized captures have already been compacted out
 * on the Mojo side (device_context.mojo:3145).
 *
 * gridDim/blockDim are the CUDA SPMD launch geometry: "run this scalar
 * program gridX*gridY*gridZ blocks of blockX*blockY*blockZ threads". T1 has
 * no thread grid — its parallelism is inside one instruction (VLEN/lanes) —
 * so REJECT anything other than all-ones rather than silently ignoring it.
 * A kernel written for gridDim=1024 assumes 1024 copies run; running one copy
 * would silently produce wrong results, which is the worst possible failure
 * mode to debug in RTL simulation.
 *
 * DECIDED: milestone 1 rejects any non-all-ones geometry. The planned future
 * meaning (see asyncrt-abi-inventory.md §1, Cray-MSP analogy) is
 *   blockDim -> multiple threads inside one T1, gridDim -> multiple T1s on the
 * bus — so carry all six components through the runtime's internal structures
 * from day one even though only 1,1,1 / 1,1,1 is accepted now.
\*==========================================================================*/

typedef struct LaunchAttribute LaunchAttribute; /* opaque; expect count 0 */

/* device_context.mojo:3863 — every dimension/count is uint32_t. */
const char *AsyncRT_DeviceContext_enqueueFunctionDirect(
    const DeviceContext *ctx,
    const DeviceFunction *func,
    uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ,
    uint32_t sharedMemBytes,
    LaunchAttribute *attributes,
    uint32_t numAttributes,
    void **args,
    uint32_t argCount,
    uint64_t *argSizes);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* T1_ASYNCRT_ABI_H */
