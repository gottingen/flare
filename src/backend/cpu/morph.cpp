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
#include <copy.hpp>
#include <kernel/morph.hpp>
#include <morph.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <fly/dim4.hpp>
#include <algorithm>

using fly::dim4;

namespace flare {
namespace cpu {
template<typename T>
Array<T> morph(const Array<T> &in, const Array<T> &mask, bool isDilation) {
    fly::borderType padType = isDilation ? FLY_PAD_ZERO : FLY_PAD_CLAMP_TO_EDGE;
    const fly::dim4 &idims  = in.dims();
    const fly::dim4 &mdims  = mask.dims();

    const fly::dim4 lpad(mdims[0] / 2, mdims[1] / 2, 0, 0);
    const fly::dim4 &upad(lpad);
    const fly::dim4 odims(lpad[0] + idims[0] + upad[0],
                         lpad[1] + idims[1] + upad[1], idims[2], idims[3]);

    auto out = createEmptyArray<T>(odims);
    auto inp = padArrayBorders(in, lpad, upad, padType);

    if (isDilation) {
        getQueue().enqueue(kernel::morph<T, true>, out, inp, mask);
    } else {
        getQueue().enqueue(kernel::morph<T, false>, out, inp, mask);
    }

    std::vector<fly_seq> idxs(4, fly_span);
    idxs[0] = fly_seq{double(lpad[0]), double(lpad[0] + idims[0] - 1), 1.0};
    idxs[1] = fly_seq{double(lpad[1]), double(lpad[1] + idims[1] - 1), 1.0};

    return createSubArray(out, idxs);
}

template<typename T>
Array<T> morph3d(const Array<T> &in, const Array<T> &mask, bool isDilation) {
    Array<T> out = createEmptyArray<T>(in.dims());
    if (isDilation) {
        getQueue().enqueue(kernel::morph3d<T, true>, out, in, mask);
    } else {
        getQueue().enqueue(kernel::morph3d<T, false>, out, in, mask);
    }
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
INSTANTIATE(ushort)
INSTANTIATE(short)
}  // namespace cpu
}  // namespace flare
