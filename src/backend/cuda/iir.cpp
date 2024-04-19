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
#include <arith.hpp>
#include <convolve.hpp>
#include <err_cuda.hpp>
#include <iir.hpp>
#include <kernel/iir.hpp>
#include <math.hpp>
#include <fly/dim4.hpp>

using fly::dim4;

namespace flare {
namespace cuda {
template<typename T>
Array<T> iir(const Array<T> &b, const Array<T> &a, const Array<T> &x) {
    FLY_BATCH_KIND type = x.ndims() == 1 ? FLY_BATCH_NONE : FLY_BATCH_SAME;
    if (x.ndims() != b.ndims()) {
        type = (x.ndims() < b.ndims()) ? FLY_BATCH_RHS : FLY_BATCH_LHS;
    }

    // Extract the first N elements
    Array<T> c = convolve<T, T>(x, b, type, 1, true);
    dim4 cdims = c.dims();
    cdims[0]   = x.dims()[0];
    c.resetDims(cdims);

    int num_a = a.dims()[0];

    if (num_a == 1) { return c; }

    dim4 ydims = c.dims();
    Array<T> y = createEmptyArray<T>(ydims);

    if (a.ndims() > 1) {
        kernel::iir<T, true>(y, c, a);
    } else {
        kernel::iir<T, false>(y, c, a);
    }
    return y;
}

#define INSTANTIATE(T)                                          \
    template Array<T> iir(const Array<T> &b, const Array<T> &a, \
                          const Array<T> &x);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
}  // namespace cuda
}  // namespace flare
