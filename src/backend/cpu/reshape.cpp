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
#include <kernel/copy.hpp>
#include <platform.hpp>
#include <queue.hpp>

namespace flare {
namespace cpu {
template<typename T>
void multiply_inplace(Array<T> &in, double val) {
    getQueue().enqueue(kernel::copyElemwise<T, T>, in, in, static_cast<T>(0),
                       val);
}

template<typename inType, typename outType>
Array<outType> reshape(const Array<inType> &in, const dim4 &outDims,
                       outType defaultValue, double scale) {
    Array<outType> out = createValueArray(outDims, defaultValue);
    getQueue().enqueue(kernel::copyElemwise<outType, inType>, out, in,
                       defaultValue, scale);
    return out;
}

#define INSTANTIATE(T) \
    template void multiply_inplace<T>(Array<T> & in, double norm);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(short)
INSTANTIATE(ushort)

#define INSTANTIATE_PAD_ARRAY(SRC_T)                                          \
    template Array<float> reshape<SRC_T, float>(const Array<SRC_T> &,         \
                                                const dim4 &, float, double); \
    template Array<double> reshape<SRC_T, double>(                            \
        const Array<SRC_T> &, const dim4 &, double, double);                  \
    template Array<cfloat> reshape<SRC_T, cfloat>(                            \
        const Array<SRC_T> &, const dim4 &, cfloat, double);                  \
    template Array<cdouble> reshape<SRC_T, cdouble>(                          \
        const Array<SRC_T> &, const dim4 &, cdouble, double);                 \
    template Array<int> reshape<SRC_T, int>(const Array<SRC_T> &,             \
                                            const dim4 &, int, double);       \
    template Array<uint> reshape<SRC_T, uint>(const Array<SRC_T> &,           \
                                              const dim4 &, uint, double);    \
    template Array<intl> reshape<SRC_T, intl>(const Array<SRC_T> &,           \
                                              const dim4 &, intl, double);    \
    template Array<uintl> reshape<SRC_T, uintl>(const Array<SRC_T> &,         \
                                                const dim4 &, uintl, double); \
    template Array<short> reshape<SRC_T, short>(const Array<SRC_T> &,         \
                                                const dim4 &, short, double); \
    template Array<ushort> reshape<SRC_T, ushort>(                            \
        const Array<SRC_T> &, const dim4 &, ushort, double);                  \
    template Array<uchar> reshape<SRC_T, uchar>(const Array<SRC_T> &,         \
                                                const dim4 &, uchar, double); \
    template Array<char> reshape<SRC_T, char>(const Array<SRC_T> &,           \
                                              const dim4 &, char, double);

INSTANTIATE_PAD_ARRAY(float)
INSTANTIATE_PAD_ARRAY(double)
INSTANTIATE_PAD_ARRAY(int)
INSTANTIATE_PAD_ARRAY(uint)
INSTANTIATE_PAD_ARRAY(intl)
INSTANTIATE_PAD_ARRAY(uintl)
INSTANTIATE_PAD_ARRAY(uchar)
INSTANTIATE_PAD_ARRAY(char)
INSTANTIATE_PAD_ARRAY(ushort)
INSTANTIATE_PAD_ARRAY(short)
INSTANTIATE_PAD_ARRAY(flare::common::half)

#define INSTANTIATE_PAD_ARRAY_COMPLEX(SRC_T)                 \
    template Array<cfloat> reshape<SRC_T, cfloat>(           \
        const Array<SRC_T> &, const dim4 &, cfloat, double); \
    template Array<cdouble> reshape<SRC_T, cdouble>(         \
        const Array<SRC_T> &, const dim4 &, cdouble, double);

INSTANTIATE_PAD_ARRAY_COMPLEX(cfloat)
INSTANTIATE_PAD_ARRAY_COMPLEX(cdouble)
}  // namespace cpu
}  // namespace flare
