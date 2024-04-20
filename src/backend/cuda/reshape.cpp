
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

#include <copy.hpp>

#include <common/half.hpp>
#include <kernel/memcopy.hpp>

using flare::common::half;

namespace flare {
namespace cuda {

template<typename inType, typename outType>
Array<outType> reshape(const Array<inType> &in, const dim4 &outDims,
                       outType defaultValue, double scale) {
    Array<outType> out = createEmptyArray<outType>(outDims);
    if (out.elements() > 0) {
        kernel::copy<inType, outType>(out, in, in.ndims(), defaultValue, scale);
    }
    return out;
}

#define INSTANTIATE(SRC_T)                                                    \
    template Array<float> reshape<SRC_T, float>(Array<SRC_T> const &,         \
                                                dim4 const &, float, double); \
    template Array<double> reshape<SRC_T, double>(                            \
        Array<SRC_T> const &, dim4 const &, double, double);                  \
    template Array<cfloat> reshape<SRC_T, cfloat>(                            \
        Array<SRC_T> const &, dim4 const &, cfloat, double);                  \
    template Array<cdouble> reshape<SRC_T, cdouble>(                          \
        Array<SRC_T> const &, dim4 const &, cdouble, double);                 \
    template Array<int> reshape<SRC_T, int>(Array<SRC_T> const &,             \
                                            dim4 const &, int, double);       \
    template Array<uint> reshape<SRC_T, uint>(Array<SRC_T> const &,           \
                                              dim4 const &, uint, double);    \
    template Array<intl> reshape<SRC_T, intl>(Array<SRC_T> const &,           \
                                              dim4 const &, intl, double);    \
    template Array<uintl> reshape<SRC_T, uintl>(Array<SRC_T> const &,         \
                                                dim4 const &, uintl, double); \
    template Array<short> reshape<SRC_T, short>(Array<SRC_T> const &,         \
                                                dim4 const &, short, double); \
    template Array<ushort> reshape<SRC_T, ushort>(                            \
        Array<SRC_T> const &, dim4 const &, ushort, double);                  \
    template Array<uchar> reshape<SRC_T, uchar>(Array<SRC_T> const &,         \
                                                dim4 const &, uchar, double); \
    template Array<char> reshape<SRC_T, char>(Array<SRC_T> const &,           \
                                              dim4 const &, char, double);    \
    template Array<half> reshape<SRC_T, half>(Array<SRC_T> const &,           \
                                              dim4 const &, half, double);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(half)

#define INSTANTIATE_COMPLEX(SRC_T)                           \
    template Array<cfloat> reshape<SRC_T, cfloat>(           \
        Array<SRC_T> const &, dim4 const &, cfloat, double); \
    template Array<cdouble> reshape<SRC_T, cdouble>(         \
        Array<SRC_T> const &, dim4 const &, cdouble, double);

INSTANTIATE_COMPLEX(cfloat)
INSTANTIATE_COMPLEX(cdouble)

}  // namespace cuda
}  // namespace flare