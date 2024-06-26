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

#include <common/DefaultMemoryManager.hpp>
#include <common/Logger.hpp>
#include <common/half.hpp>
#include <err_cpu.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <collie/log/logging.h>
#include <types.hpp>
#include <fly/dim4.hpp>

#include <utility>

using fly::dim4;
using flare::common::bytesToString;
using flare::common::half;
using std::function;
using std::move;
using std::unique_ptr;

namespace flare {
namespace cpu {
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

void printMemInfo(const char *msg, const int device) {
    memoryManager().printInfo(msg, device);
}

template<typename T>
unique_ptr<T[], function<void(void *)>> memAlloc(const size_t &elements) {
    // TODO: make memAlloc aware of array shapes
    dim4 dims(elements);
    T *ptr = static_cast<T *>(
        memoryManager().alloc(false, 1, dims.get(), sizeof(T)));
    return unique_ptr<T[], function<void(void *)>>(ptr, memFree);
}

void *memAllocUser(const size_t &bytes) {
    dim4 dims(bytes);
    void *ptr = memoryManager().alloc(true, 1, dims.get(), 1);
    return ptr;
}

void memFree(void *ptr) { return memoryManager().unlock(ptr, false); }

void memFreeUser(void *ptr) { memoryManager().unlock(ptr, true); }

void memLock(const void *ptr) { memoryManager().userLock(ptr); }

bool isLocked(const void *ptr) { return memoryManager().isUserLocked(ptr); }

void memUnlock(const void *ptr) { memoryManager().userUnlock(ptr); }

void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                      size_t *lock_bytes, size_t *lock_buffers) {
    memoryManager().usageInfo(alloc_bytes, alloc_buffers, lock_bytes,
                              lock_buffers);
}

template<typename T>
T *pinnedAlloc(const size_t &elements) {
    // TODO: make pinnedAlloc aware of array shapes
    dim4 dims(elements);
    void *ptr = memoryManager().alloc(false, 1, dims.get(), sizeof(T));
    return static_cast<T *>(ptr);
}

void pinnedFree(void *ptr) { memoryManager().unlock(ptr, false); }

#define INSTANTIATE(T)                                                   \
    template std::unique_ptr<T[], std::function<void(void *)>> memAlloc( \
        const size_t &elements);                                         \
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
INSTANTIATE(ushort)
INSTANTIATE(short)
INSTANTIATE(half)

template<>
void *pinnedAlloc<void>(const size_t &elements) {
    // TODO: make pinnedAlloc aware of array shapes
    dim4 dims(elements);
    void *ptr = memoryManager().alloc(false, 1, dims.get(), 1);
    return ptr;
}

Allocator::Allocator() { logger = common::loggerFactory("mem"); }

void Allocator::shutdown() {
    for (int n = 0; n < cpu::getDeviceCount(); n++) {
        try {
            cpu::setDevice(n);
            shutdownMemoryManager();
        } catch (const AfError &err) {
            continue;  // Do not throw any errors while shutting down
        }
    }
}

int Allocator::getActiveDeviceId() {
    return static_cast<int>(cpu::getActiveDeviceId());
}

size_t Allocator::getMaxMemorySize(int id) {
    return cpu::getDeviceMemorySize(id);
}

void *Allocator::nativeAlloc(const size_t bytes) {
    void *ptr = malloc(bytes);  // NOLINT(hicpp-no-malloc)
    FLY_TRACE("nativeAlloc: {:>7} {}", bytesToString(bytes), ptr);
    if (!ptr) { FLY_ERROR("Unable to allocate memory", FLY_ERR_NO_MEM); }
    return ptr;
}

void Allocator::nativeFree(void *ptr) {
    FLY_TRACE("nativeFree: {: >8} {}", " ", ptr);
    // Make sure this pointer is not being used on the queue before freeing the
    // memory.
    getQueue().sync();
    free(ptr);  // NOLINT(hicpp-no-malloc)
}
}  // namespace cpu
}  // namespace flare
