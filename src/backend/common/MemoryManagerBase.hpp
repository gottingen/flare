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
#include <fly/defines.h>

#include <cstddef>
#include <memory>

namespace clog {
class logger;
}

namespace flare {
namespace common {
/**
 * A internal base interface for a memory manager which is exposed to FLY
 * internals. Externally, both the default FLY memory manager implementation and
 * custom memory manager implementations are wrapped in a derived implementation
 * of this interface.
 */
class MemoryManagerBase {
   public:
    MemoryManagerBase()                                     = default;
    MemoryManagerBase &operator=(const MemoryManagerBase &) = delete;
    MemoryManagerBase(const MemoryManagerBase &)            = delete;
    virtual ~MemoryManagerBase() {}
    // Shuts down the allocator interface which calls shutdown on the subclassed
    // memory manager with device-specific context
    virtual void shutdownAllocator() {
        if (nmi_) nmi_->shutdown();
    }
    virtual void initialize()                                        = 0;
    virtual void shutdown()                                          = 0;
    virtual void *alloc(bool user_lock, const unsigned ndims, dim_t *dims,
                        const unsigned element_size)                 = 0;
    virtual size_t allocated(void *ptr)                              = 0;
    virtual void unlock(void *ptr, bool user_unlock)                 = 0;
    virtual void signalMemoryCleanup()                               = 0;
    virtual void printInfo(const char *msg, const int device)        = 0;
    virtual void usageInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                           size_t *lock_bytes, size_t *lock_buffers) = 0;
    virtual void userLock(const void *ptr)                           = 0;
    virtual void userUnlock(const void *ptr)                         = 0;
    virtual bool isUserLocked(const void *ptr)                       = 0;
    virtual size_t getMemStepSize()                                  = 0;
    virtual void setMemStepSize(size_t new_step_size)                = 0;

    int getActiveDeviceId() { return nmi_->getActiveDeviceId(); }
    size_t getMaxMemorySize(int id) { return nmi_->getMaxMemorySize(id); }
    void *nativeAlloc(const size_t bytes) { return nmi_->nativeAlloc(bytes); }
    void nativeFree(void *ptr) { nmi_->nativeFree(ptr); }
    virtual clog::logger *getLogger() final { return nmi_->getLogger(); }
    virtual void setAllocator(std::unique_ptr<AllocatorInterface> nmi) {
        nmi_ = std::move(nmi);
    }

    // Memory pressure functions
    void setMemoryPressureThreshold(float pressure) {
        memoryPressureThreshold_ = pressure;
    }
    float getMemoryPressureThreshold() const {
        return memoryPressureThreshold_;
    }
    virtual float getMemoryPressure()                       = 0;
    virtual bool jitTreeExceedsMemoryPressure(size_t bytes) = 0;

   private:
    // A threshold at or above which JIT evaluations will be triggered due to
    // memory pressure. Settable via a call to setMemoryPressureThreshold
    float memoryPressureThreshold_{1.0};
    // A backend-specific memory manager, containing backend-specific
    // methods that call native memory manipulation functions in a device
    // API. We need to wrap these since they are opaquely called by the
    // memory manager.
    std::unique_ptr<AllocatorInterface> nmi_;
};

}  // namespace common
}  // namespace flare
