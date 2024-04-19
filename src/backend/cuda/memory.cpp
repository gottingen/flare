// Copyright 2023 The EA Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <memory.hpp>

#include <Event.hpp>
#include <common/Logger.hpp>
#include <common/MemoryManagerBase.hpp>
#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <common/util.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <err_cuda.hpp>
#include <platform.hpp>
#include <collie/log/logging.h>
#include <types.hpp>
#include <fly/dim4.hpp>

#include <cstdlib>
#include <mutex>

using fly::dim4;
using flare::common::bytesToString;
using flare::common::half;

using std::move;

namespace flare {
namespace cuda {
float getMemoryPressure() { return memoryManager().getMemoryPressure(); }
float getMemoryPressureThreshold() {
    return memoryManager().getMemoryPressureThreshold();
}

bool jitTreeExceedsMemoryPressure(size_t bytes) {
    return memoryManager().jitTreeExceedsMemoryPressure(bytes);
}

void setMemStepSize(size_t step_bytes) {
    memoryManager().setMemStepSize(step_bytes);
}

size_t getMemStepSize() { return memoryManager().getMemStepSize(); }

void signalMemoryCleanup() { memoryManager().signalMemoryCleanup(); }

void shutdownMemoryManager() { memoryManager().shutdown(); }

void shutdownPinnedMemoryManager() { pinnedMemoryManager().shutdown(); }

void printMemInfo(const char *msg, const int device) {
    memoryManager().printInfo(msg, device);
}

template<typename T>
uptr<T> memAlloc(const size_t &elements) {
    // TODO: make memAlloc aware of array shapes
    dim4 dims(elements);
    void *ptr = memoryManager().alloc(false, 1, dims.get(), sizeof(T));
    return uptr<T>(static_cast<T *>(ptr), memFree);
}

void *memAllocUser(const size_t &bytes) {
    dim4 dims(bytes);
    void *ptr = memoryManager().alloc(true, 1, dims.get(), 1);
    return ptr;
}

void memFree(void *ptr) { memoryManager().unlock(ptr, false); }

void memFreeUser(void *ptr) { memoryManager().unlock(ptr, true); }

void memLock(const void *ptr) {
    memoryManager().userLock(const_cast<void *>(ptr));
}

void memUnlock(const void *ptr) {
    memoryManager().userUnlock(const_cast<void *>(ptr));
}

bool isLocked(const void *ptr) {
    return memoryManager().isUserLocked(const_cast<void *>(ptr));
}

void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                      size_t *lock_bytes, size_t *lock_buffers) {
    memoryManager().usageInfo(alloc_bytes, alloc_buffers, lock_bytes,
                              lock_buffers);
}

template<typename T>
T *pinnedAlloc(const size_t &elements) {
    // TODO: make pinnedAlloc aware of array shapes
    dim4 dims(elements);
    void *ptr = pinnedMemoryManager().alloc(false, 1, dims.get(), sizeof(T));
    return static_cast<T *>(ptr);
}

void pinnedFree(void *ptr) { pinnedMemoryManager().unlock(ptr, false); }

#define INSTANTIATE(T)                                 \
    template uptr<T> memAlloc(const size_t &elements); \
    template T *pinnedAlloc(const size_t &elements);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(char)
INSTANTIATE(uchar)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)

template<>
void *pinnedAlloc<void>(const size_t &elements) {
    // TODO: make pinnedAlloc aware of array shapes
    dim4 dims(elements);
    void *ptr = pinnedMemoryManager().alloc(false, 1, dims.get(), 1);
    return ptr;
}

Allocator::Allocator() { logger = common::loggerFactory("mem"); }

void Allocator::shutdown() {
    for (int n = 0; n < getDeviceCount(); n++) {
        try {
            setDevice(n);
            shutdownMemoryManager();
        } catch (const AfError &err) {
            continue;  // Do not throw any errors while shutting down
        }
    }
}

int Allocator::getActiveDeviceId() { return cuda::getActiveDeviceId(); }

size_t Allocator::getMaxMemorySize(int id) { return getDeviceMemorySize(id); }

void *Allocator::nativeAlloc(const size_t bytes) {
    void *ptr = NULL;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    FLY_TRACE("nativeAlloc: {:>7} {}", bytesToString(bytes), ptr);
    return ptr;
}

void Allocator::nativeFree(void *ptr) {
    FLY_TRACE("nativeFree:          {}", ptr);
    cudaError_t err = cudaFree(ptr);
    if (err != cudaErrorCudartUnloading) { CUDA_CHECK(err); }
}

AllocatorPinned::AllocatorPinned() { logger = common::loggerFactory("mem"); }

void AllocatorPinned::shutdown() { shutdownPinnedMemoryManager(); }

int AllocatorPinned::getActiveDeviceId() {
    return 0;  // pinned uses a single vector
}

size_t AllocatorPinned::getMaxMemorySize(int id) {
    UNUSED(id);
    return getHostMemorySize();
}

void *AllocatorPinned::nativeAlloc(const size_t bytes) {
    void *ptr;
    CUDA_CHECK(cudaMallocHost(&ptr, bytes));
    FLY_TRACE("Pinned::nativeAlloc: {:>7} {}", bytesToString(bytes), ptr);
    return ptr;
}

void AllocatorPinned::nativeFree(void *ptr) {
    FLY_TRACE("Pinned::nativeFree:          {}", ptr);
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaErrorCudartUnloading) { CUDA_CHECK(err); }
}
}  // namespace cuda
}  // namespace flare
