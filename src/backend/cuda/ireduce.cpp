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
#include <ireduce.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <fly/dim4.hpp>

#undef _GLIBCXX_USE_INT128
#include <err_cuda.hpp>
#include <kernel/ireduce.hpp>

#include <complex>

using fly::dim4;
using flare::common::half;

namespace flare {
namespace cuda {

template<fly_op_t op, typename T>
void ireduce(Array<T> &out, Array<uint> &loc, const Array<T> &in,
             const int dim) {
    Array<uint> rlen = createEmptyArray<uint>(fly::dim4(0));
    kernel::ireduce<T, op>(out, loc.get(), in, dim, rlen);
}

template<fly_op_t op, typename T>
void rreduce(Array<T> &out, Array<uint> &loc, const Array<T> &in, const int dim,
             const Array<uint> &rlen) {
    kernel::ireduce<T, op>(out, loc.get(), in, dim, rlen);
}

template<fly_op_t op, typename T>
T ireduce_all(unsigned *loc, const Array<T> &in) {
    return kernel::ireduce_all<T, op>(loc, in);
}

#define INSTANTIATE(ROp, T)                                           \
    template void ireduce<ROp, T>(Array<T> & out, Array<uint> & loc,  \
                                  const Array<T> &in, const int dim); \
    template void rreduce<ROp, T>(Array<T> & out, Array<uint> & loc,  \
                                  const Array<T> &in, const int dim,  \
                                  const Array<uint> &rlen);           \
    template T ireduce_all<ROp, T>(unsigned *loc, const Array<T> &in);

// min
INSTANTIATE(fly_min_t, float)
INSTANTIATE(fly_min_t, double)
INSTANTIATE(fly_min_t, cfloat)
INSTANTIATE(fly_min_t, cdouble)
INSTANTIATE(fly_min_t, int)
INSTANTIATE(fly_min_t, uint)
INSTANTIATE(fly_min_t, intl)
INSTANTIATE(fly_min_t, uintl)
INSTANTIATE(fly_min_t, short)
INSTANTIATE(fly_min_t, ushort)
INSTANTIATE(fly_min_t, char)
INSTANTIATE(fly_min_t, uchar)
INSTANTIATE(fly_min_t, half)

// max
INSTANTIATE(fly_max_t, float)
INSTANTIATE(fly_max_t, double)
INSTANTIATE(fly_max_t, cfloat)
INSTANTIATE(fly_max_t, cdouble)
INSTANTIATE(fly_max_t, int)
INSTANTIATE(fly_max_t, uint)
INSTANTIATE(fly_max_t, intl)
INSTANTIATE(fly_max_t, uintl)
INSTANTIATE(fly_max_t, short)
INSTANTIATE(fly_max_t, ushort)
INSTANTIATE(fly_max_t, char)
INSTANTIATE(fly_max_t, uchar)
INSTANTIATE(fly_max_t, half)
}  // namespace cuda
}  // namespace flare
