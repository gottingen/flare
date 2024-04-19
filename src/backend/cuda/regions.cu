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

#include <Array.hpp>
#include <err_cuda.hpp>
#include <kernel/regions.hpp>
#include <regions.hpp>
#include <fly/dim4.hpp>

using fly::dim4;

namespace flare {
namespace cuda {

template<typename T>
Array<T> regions(const Array<char> &in, fly_connectivity connectivity) {
    const dim4 dims = in.dims();

    Array<T> out = createEmptyArray<T>(dims);

    // Create bindless texture object for the equiv map.
    cudaTextureObject_t tex = 0;

    // Use texture objects with compute 3.0 or higher
    if (!std::is_same<T, double>::value) {
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType           = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = out.get();

        if (std::is_signed<T>::value)
            resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
        else if (std::is_unsigned<T>::value)
            resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
        else
            resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;

        resDesc.res.linear.desc.x      = sizeof(T) * 8;  // bits per channel
        resDesc.res.linear.sizeInBytes = dims[0] * dims[1] * sizeof(T);
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    }

    switch (connectivity) {
        case FLY_CONNECTIVITY_4: ::regions<T, false, 2>(out, in, tex); break;
        case FLY_CONNECTIVITY_8_4: ::regions<T, true, 2>(out, in, tex); break;
    }

    // Iterative procedure(while loop) in kernel::regions
    // does stream synchronization towards loop end. So, it is
    // safe to destroy the texture object
    CUDA_CHECK(cudaDestroyTextureObject(tex));

    return out;
}

#define INSTANTIATE(T)                                  \
    template Array<T> regions<T>(const Array<char> &in, \
                                 fly_connectivity connectivity);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cuda
}  // namespace flare
