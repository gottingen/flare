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

#include <cstddef>
#include <memory>

namespace clog {
class logger;
}
namespace flare {
namespace common {

/**
 * An interface that provides backend-specific memory management functions,
 * typically calling a dedicated backend-specific native API. Stored, wrapped,
 * and called by a MemoryManagerBase, from which calls to its interface are
 * delegated.
 */
class AllocatorInterface {
   public:
    AllocatorInterface() = default;
    virtual ~AllocatorInterface() {}
    virtual void shutdown()                       = 0;
    virtual int getActiveDeviceId()               = 0;
    virtual size_t getMaxMemorySize(int id)       = 0;
    virtual void *nativeAlloc(const size_t bytes) = 0;
    virtual void nativeFree(void *ptr)            = 0;
    virtual clog::logger *getLogger() final { return this->logger.get(); }

   protected:
    std::shared_ptr<clog::logger> logger;
};

}  // namespace common
}  // namespace flare
