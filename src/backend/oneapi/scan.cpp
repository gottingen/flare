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

#include <err_oneapi.hpp>
#include <scan.hpp>

#include <kernel/scan_dim.hpp>
#include <kernel/scan_first.hpp>

namespace flare {
namespace oneapi {
template<fly_op_t op, typename Ti, typename To>
Array<To> scan(const Array<Ti>& in, const int dim, bool inclusiveScan) {
    Array<To> out = createEmptyArray<To>(in.dims());

    Param<To> Out = out;
    Param<Ti> In  = in;

    switch (dim) {
        case 0: kernel::scan_first<Ti, To, op>(Out, In, inclusiveScan); break;
        case 1: kernel::scan_dim<Ti, To, op, 1>(Out, In, inclusiveScan); break;
        case 2: kernel::scan_dim<Ti, To, op, 2>(Out, In, inclusiveScan); break;
        case 3: kernel::scan_dim<Ti, To, op, 3>(Out, In, inclusiveScan); break;
    }

    return out;
}

#define INSTANTIATE_SCAN(ROp, Ti, To) \
    template Array<To> scan<ROp, Ti, To>(const Array<Ti>&, const int, bool);

#define INSTANTIATE_SCAN_ALL(ROp)           \
    INSTANTIATE_SCAN(ROp, float, float)     \
    INSTANTIATE_SCAN(ROp, double, double)   \
    INSTANTIATE_SCAN(ROp, cfloat, cfloat)   \
    INSTANTIATE_SCAN(ROp, cdouble, cdouble) \
    INSTANTIATE_SCAN(ROp, int, int)         \
    INSTANTIATE_SCAN(ROp, uint, uint)       \
    INSTANTIATE_SCAN(ROp, intl, intl)       \
    INSTANTIATE_SCAN(ROp, uintl, uintl)     \
    INSTANTIATE_SCAN(ROp, char, uint)       \
    INSTANTIATE_SCAN(ROp, uchar, uint)      \
    INSTANTIATE_SCAN(ROp, short, int)       \
    INSTANTIATE_SCAN(ROp, ushort, uint)

INSTANTIATE_SCAN(fly_notzero_t, char, uint)
INSTANTIATE_SCAN_ALL(fly_add_t)
INSTANTIATE_SCAN_ALL(fly_mul_t)
INSTANTIATE_SCAN_ALL(fly_min_t)
INSTANTIATE_SCAN_ALL(fly_max_t)
}  // namespace oneapi
}  // namespace flare
