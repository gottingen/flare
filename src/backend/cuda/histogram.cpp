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
#include <err_cuda.hpp>
#include <histogram.hpp>
#include <kernel/histogram.hpp>
#include <fly/dim4.hpp>

using fly::dim4;
using flare::common::half;

namespace flare {
namespace cuda {

template<typename T>
Array<uint> histogram(const Array<T> &in, const unsigned &nbins,
                      const double &minval, const double &maxval,
                      const bool isLinear) {
    const dim4 &dims = in.dims();
    dim4 outDims     = dim4(nbins, 1, dims[2], dims[3]);
    Array<uint> out  = createValueArray<uint>(outDims, uint(0));
    kernel::histogram<T>(out, in, nbins, minval, maxval, isLinear);
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

}  // namespace cuda
}  // namespace flare
