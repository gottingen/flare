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
#include <kernel/morph.hpp>
#include <morph.hpp>
#include <fly/dim4.hpp>

using fly::dim4;

namespace flare {
namespace cuda {

template<typename T>
Array<T> morph(const Array<T> &in, const Array<T> &mask, bool isDilation) {
    const dim4 mdims = mask.dims();
    if (mdims[0] != mdims[1]) {
        CUDA_NOT_SUPPORTED("Rectangular masks are not supported");
    }
    if (mdims[0] > 19) {
        CUDA_NOT_SUPPORTED("Kernels > 19x19 are not supported");
    }
    Array<T> out = createEmptyArray<T>(in.dims());
    kernel::morph<T>(out, in, mask, isDilation);
    return out;
}

template<typename T>
Array<T> morph3d(const Array<T> &in, const Array<T> &mask, bool isDilation) {
    const dim4 mdims = mask.dims();
    if (mdims[0] != mdims[1] || mdims[0] != mdims[2]) {
        CUDA_NOT_SUPPORTED("Only cubic masks are supported");
    }
    if (mdims[0] > 7) { CUDA_NOT_SUPPORTED("Kernels > 7x7x7 not supported"); }
    Array<T> out = createEmptyArray<T>(in.dims());
    kernel::morph3d<T>(out, in, mask, isDilation);
    return out;
}

#define INSTANTIATE(T)                                                    \
    template Array<T> morph<T>(const Array<T> &, const Array<T> &, bool); \
    template Array<T> morph3d<T>(const Array<T> &, const Array<T> &, bool);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cuda
}  // namespace flare
