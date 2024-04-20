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

#ifdef __CUDACC_RTC__

namespace flare {
namespace cuda {
template<typename T>
struct SharedMemory {
    __DH__ T* getPointer() {
        extern __shared__ T ptr[];
        return ptr;
    }
};
}  // namespace cuda
}  // namespace flare

#else

namespace flare {
namespace cuda {
namespace kernel {

template<typename T>
struct SharedMemory {
    // return a pointer to the runtime-sized shared memory array.
    __device__ T* getPointer();
};

#define SPECIALIZE(T)                         \
    template<>                                \
    struct SharedMemory<T> {                  \
        __device__ T* getPointer() {          \
            extern __shared__ T ptr_##T##_[]; \
            return ptr_##T##_;                \
        }                                     \
    };

SPECIALIZE(float)
SPECIALIZE(cfloat)
SPECIALIZE(double)
SPECIALIZE(cdouble)
SPECIALIZE(char)
SPECIALIZE(int)
SPECIALIZE(uint)
SPECIALIZE(short)
SPECIALIZE(ushort)
SPECIALIZE(uchar)
SPECIALIZE(intl)
SPECIALIZE(uintl)

#undef SPECIALIZE

}  // namespace kernel
}  // namespace cuda
}  // namespace flare

#endif
