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

#include <moments.hpp>

#include <Array.hpp>
#include <debug_cuda.hpp>
#include <err_cuda.hpp>
#include <kernel/moments.hpp>

namespace flare {
namespace cuda {

static inline unsigned bitCount(unsigned v) {
    v = v - ((v >> 1U) & 0x55555555U);
    v = (v & 0x33333333U) + ((v >> 2U) & 0x33333333U);
    return (((v + (v >> 4U)) & 0xF0F0F0FU) * 0x1010101U) >> 24U;
}

using fly::dim4;

template<typename T>
Array<float> moments(const Array<T> &in, const fly_moment_type moment) {
    in.eval();
    dim4 odims, idims = in.dims();
    dim_t moments_dim = bitCount(moment);

    odims[0] = moments_dim;
    odims[1] = 1;
    odims[2] = idims[2];
    odims[3] = idims[3];

    Array<float> out = createValueArray<float>(odims, 0.f);
    out.eval();

    kernel::moments<T>(out, in, moment);
    return out;
}

#define INSTANTIATE(T)                                   \
    template Array<float> moments<T>(const Array<T> &in, \
                                     const fly_moment_type moment);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace cuda
}  // namespace flare
