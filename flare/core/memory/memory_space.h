// Copyright 2023 The Elastic-AI Authors.
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

/** @file memory_space.hpp
 *
 *  Operations common to memory space instances, or at least default
 *  implementations thereof.
 */

#ifndef FLARE_CORE_MEMORY_MEMORY_SPACE_H_
#define FLARE_CORE_MEMORY_MEMORY_SPACE_H_

#include <flare/core/defines.h>
#include <flare/core/memory/shared_alloc.h>
#include <flare/core/common/error.h>

#include <string>

namespace flare::detail {

    // Defined in implementation file to avoid having to include iostream
    void safe_throw_allocation_with_header_failure(
            std::string const &space_name, std::string const &label,
            flare::experimental::RawMemoryAllocationFailure const &failure);

    template<class MemorySpace>
    SharedAllocationHeader *checked_allocation_with_header(MemorySpace const &space,
                                                           std::string const &label,
                                                           size_t alloc_size) {
        try {
            return reinterpret_cast<SharedAllocationHeader *>(space.allocate(
                    label.c_str(), alloc_size + sizeof(SharedAllocationHeader),
                    alloc_size));
        } catch (flare::experimental::RawMemoryAllocationFailure const &failure) {
            safe_throw_allocation_with_header_failure(space.name(), label, failure);
        }
        return nullptr;  // unreachable
    }

    template<class ExecutionSpace, class MemorySpace>
    SharedAllocationHeader *checked_allocation_with_header(
            ExecutionSpace const &exec_space, MemorySpace const &space,
            std::string const &label, size_t alloc_size) {
        try {
            return reinterpret_cast<SharedAllocationHeader *>(space.allocate(
                    exec_space, label.c_str(), alloc_size + sizeof(SharedAllocationHeader),
                    alloc_size));
        } catch (flare::experimental::RawMemoryAllocationFailure const &failure) {
            safe_throw_allocation_with_header_failure(space.name(), label, failure);
        }
        return nullptr;  // unreachable
    }

}  // end namespace flare::detail

#endif  // FLARE_CORE_MEMORY_MEMORY_SPACE_H_
