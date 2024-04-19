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

#include <memory.hpp>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_vector.h>

// Below Class definition is found at the following URL
// http://stackoverflow.com/questions/9007343/mix-custom-memory-managment-and-thrust-in-cuda

namespace flare {
namespace cuda {

template<typename T>
struct ThrustAllocator : thrust::device_malloc_allocator<T> {
    // shorthand for the name of the base class
    typedef thrust::device_malloc_allocator<T> super_t;

    // get access to some of the base class's typedefs
    // note that because we inherited from device_malloc_allocator,
    // pointer is actually thrust::device_ptr<T>
    typedef typename super_t::pointer pointer;

    typedef typename super_t::size_type size_type;

    pointer allocate(size_type elements) {
        return thrust::device_ptr<T>(
            memAlloc<T>(elements)
                .release());  // delegate to Flare allocator
    }

    void deallocate(pointer p, size_type n) {
        UNUSED(n);
        memFree(p.get());  // delegate to Flare allocator
    }
};
}  // namespace cuda
}  // namespace flare
