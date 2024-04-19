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
#include <err_opencl.hpp>
#include <scan.hpp>
#include <fly/dim4.hpp>
#include <complex>

#include <kernel/scan_dim_by_key.hpp>
#include <kernel/scan_first_by_key.hpp>

namespace flare {
namespace opencl {
template<fly_op_t op, typename Ti, typename Tk, typename To>
Array<To> scan(const Array<Tk>& key, const Array<Ti>& in, const int dim,
               bool inclusive_scan) {
    Array<To> out = createEmptyArray<To>(in.dims());

    Param Out = out;
    Param Key = key;
    Param In  = in;

    if (dim == 0) {
        kernel::scanFirstByKey<Ti, Tk, To, op>(Out, In, Key, inclusive_scan);
    } else {
        kernel::scanDimByKey<Ti, Tk, To, op>(Out, In, Key, dim, inclusive_scan);
    }
    return out;
}

#define INSTANTIATE_SCAN_BY_KEY(ROp, Ti, Tk, To)                  \
    template Array<To> scan<ROp, Ti, Tk, To>(                     \
        const Array<Tk>& key, const Array<Ti>& in, const int dim, \
        bool inclusive_scan);

#define INSTANTIATE_SCAN_BY_KEY_ALL(ROp, Tk)           \
    INSTANTIATE_SCAN_BY_KEY(ROp, float, Tk, float)     \
    INSTANTIATE_SCAN_BY_KEY(ROp, double, Tk, double)   \
    INSTANTIATE_SCAN_BY_KEY(ROp, cfloat, Tk, cfloat)   \
    INSTANTIATE_SCAN_BY_KEY(ROp, cdouble, Tk, cdouble) \
    INSTANTIATE_SCAN_BY_KEY(ROp, int, Tk, int)         \
    INSTANTIATE_SCAN_BY_KEY(ROp, uint, Tk, uint)       \
    INSTANTIATE_SCAN_BY_KEY(ROp, intl, Tk, intl)       \
    INSTANTIATE_SCAN_BY_KEY(ROp, uintl, Tk, uintl)

#define INSTANTIATE_SCAN_BY_KEY_OP(ROp)    \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, int)  \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, uint) \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, intl) \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, uintl)

INSTANTIATE_SCAN_BY_KEY_OP(fly_add_t)
INSTANTIATE_SCAN_BY_KEY_OP(fly_mul_t)
INSTANTIATE_SCAN_BY_KEY_OP(fly_min_t)
INSTANTIATE_SCAN_BY_KEY_OP(fly_max_t)
}  // namespace opencl
}  // namespace flare
