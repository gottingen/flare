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
#include <common/half.hpp>
#include <histogram.hpp>
#include <kernel/histogram.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <fly/dim4.hpp>

using fly::dim4;
using flare::common::half;

namespace flare {
namespace cpu {

template<typename T>
Array<uint> histogram(const Array<T> &in, const unsigned &nbins,
                      const double &minval, const double &maxval,
                      const bool isLinear) {
    const dim4 &inDims = in.dims();
    dim4 outDims       = dim4(nbins, 1, inDims[2], inDims[3]);
    Array<uint> out    = createValueArray<uint>(outDims, uint(0));
    if (isLinear) {
        getQueue().enqueue(kernel::histogram<T, true>, out, in, nbins, minval,
                           maxval);
    } else {
        getQueue().enqueue(kernel::histogram<T, false>, out, in, nbins, minval,
                           maxval);
    }
    return out;
}

#define INSTANTIATE(T)                                                    \
    template Array<uint> histogram<T>(const Array<T> &, const unsigned &, \
                                      const double &, const double &,     \
                                      const bool);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(half)

}  // namespace cpu
}  // namespace flare
