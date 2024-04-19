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
#pragma once

#include <common/AllocatorInterface.hpp>

#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace cl {
class Buffer;  // Forward declaration of cl::Buffer from CL/cl2.hpp
}

namespace flare {
namespace opencl {
cl::Buffer *bufferAlloc(const size_t &bytes);
void bufferFree(cl::Buffer *buf);

using bufptr = std::unique_ptr<cl::Buffer, std::function<void(cl::Buffer *)>>;

template<typename T>
bufptr memAlloc(const size_t &elements);
void *memAllocUser(const size_t &bytes);

// Need these as 2 separate function and not a default argument
// This is because it is used as the deleter in shared pointer
// which cannot support default arguments
void memFree(cl::Buffer *ptr);
void memFree(cl_mem ptr);
void memFreeUser(void *ptr);

void memLock(const cl::Buffer *ptr);
void memUnlock(const cl::Buffer *ptr);
bool isLocked(const void *ptr);

template<typename T>
T *pinnedAlloc(const size_t &elements);
template<typename T>
void pinnedFree(T *ptr);

void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                      size_t *lock_bytes, size_t *lock_buffers);
void signalMemoryCleanup();
void shutdownMemoryManager();
void pinnedGarbageCollect();

void printMemInfo(const char *msg, const int device);

float getMemoryPressure();
float getMemoryPressureThreshold();
bool jitTreeExceedsMemoryPressure(size_t bytes);
void setMemStepSize(size_t step_bytes);
size_t getMemStepSize(void);

class Allocator final : public common::AllocatorInterface {
   public:
    Allocator();
    ~Allocator() = default;
    void shutdown() override;
    int getActiveDeviceId() override;
    size_t getMaxMemorySize(int id) override;
    void *nativeAlloc(const size_t bytes) override;
    void nativeFree(void *ptr) override;
};

class AllocatorPinned final : public common::AllocatorInterface {
   public:
    AllocatorPinned();
    ~AllocatorPinned() = default;
    void shutdown() override;
    int getActiveDeviceId() override;
    size_t getMaxMemorySize(int id) override;
    void *nativeAlloc(const size_t bytes) override;
    void nativeFree(void *ptr) override;

   private:
    std::vector<std::map<void *, cl::Buffer *>> pinnedMaps;
};

}  // namespace opencl
}  // namespace flare
