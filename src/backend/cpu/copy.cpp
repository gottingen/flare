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
#include <common/ArrayInfo.hpp>
#include <common/complex.hpp>
#include <common/half.hpp>
#include <copy.hpp>
#include <err_cpu.hpp>
#include <kernel/copy.hpp>
#include <platform.hpp>
#include <queue.hpp>

#include <fly/defines.h>
#include <fly/dim4.hpp>

#include <cstdio>
#include <cstring>

using flare::common::half;  // NOLINT(misc-unused-using-decls) bug in
                                // clang-tidy
using flare::common::is_complex;

namespace flare {
namespace cpu {

template<typename T>
void copyData(T *to, const Array<T> &from) {
    if (from.elements() == 0) { return; }

    from.eval();
    // Ensure all operations on 'from' are complete before copying data to host.
    getQueue().sync();
    if (from.isLinear()) {
        // FIXME: Check for errors / exceptions
        memcpy(to, from.get(), from.elements() * sizeof(T));
    } else {
        dim4 ostrides = calcStrides(from.dims());
        kernel::stridedCopy<T>(to, ostrides, from.get(), from.dims(),
                               from.strides(), from.ndims() - 1);
    }
}

template<typename T>
Array<T> copyArray(const Array<T> &A) {
    Array<T> out = createEmptyArray<T>(A.dims());
    if (A.elements() > 0) { getQueue().enqueue(kernel::copy<T, T>, out, A); }
    return out;
}

template<typename inType, typename outType>
void copyArray(Array<outType> &out, Array<inType> const &in) {
    static_assert(
        !(is_complex<inType>::value && !is_complex<outType>::value),
        "Cannot copy from complex Array<T> to a non complex Array<T>");
    getQueue().enqueue(kernel::copy<outType, inType>, out, in);
}

#define INSTANTIATE(T)                                         \
    template void copyData<T>(T * data, const Array<T> &from); \
    template Array<T> copyArray<T>(const Array<T> &A);

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
INSTANTIATE(half)

#define INSTANTIATE_COPY_ARRAY(SRC_T)                                 \
    template void copyArray<SRC_T, float>(Array<float> & dst,         \
                                          Array<SRC_T> const &src);   \
    template void copyArray<SRC_T, double>(Array<double> & dst,       \
                                           Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, cfloat>(Array<cfloat> & dst,       \
                                           Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, cdouble>(Array<cdouble> & dst,     \
                                            Array<SRC_T> const &src); \
    template void copyArray<SRC_T, int>(Array<int> & dst,             \
                                        Array<SRC_T> const &src);     \
    template void copyArray<SRC_T, uint>(Array<uint> & dst,           \
                                         Array<SRC_T> const &src);    \
    template void copyArray<SRC_T, intl>(Array<intl> & dst,           \
                                         Array<SRC_T> const &src);    \
    template void copyArray<SRC_T, uintl>(Array<uintl> & dst,         \
                                          Array<SRC_T> const &src);   \
    template void copyArray<SRC_T, short>(Array<short> & dst,         \
                                          Array<SRC_T> const &src);   \
    template void copyArray<SRC_T, ushort>(Array<ushort> & dst,       \
                                           Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, uchar>(Array<uchar> & dst,         \
                                          Array<SRC_T> const &src);   \
    template void copyArray<SRC_T, char>(Array<char> & dst,           \
                                         Array<SRC_T> const &src);    \
    template void copyArray<SRC_T, half>(Array<half> & dst,           \
                                         Array<SRC_T> const &src);

INSTANTIATE_COPY_ARRAY(float)
INSTANTIATE_COPY_ARRAY(double)
INSTANTIATE_COPY_ARRAY(int)
INSTANTIATE_COPY_ARRAY(uint)
INSTANTIATE_COPY_ARRAY(intl)
INSTANTIATE_COPY_ARRAY(uintl)
INSTANTIATE_COPY_ARRAY(uchar)
INSTANTIATE_COPY_ARRAY(char)
INSTANTIATE_COPY_ARRAY(ushort)
INSTANTIATE_COPY_ARRAY(short)
INSTANTIATE_COPY_ARRAY(half)

#define INSTANTIATE_COPY_ARRAY_COMPLEX(SRC_T)                        \
    template void copyArray<SRC_T, cfloat>(Array<cfloat> & dst,      \
                                           Array<SRC_T> const &src); \
    template void copyArray<SRC_T, cdouble>(Array<cdouble> & dst,    \
                                            Array<SRC_T> const &src);

INSTANTIATE_COPY_ARRAY_COMPLEX(cfloat)
INSTANTIATE_COPY_ARRAY_COMPLEX(cdouble)

template<typename T>
T getScalar(const Array<T> &in) {
    in.eval();
    getQueue().sync();
    return in.get()[0];
}

#define INSTANTIATE_GETSCALAR(T) template T getScalar(const Array<T> &in);

INSTANTIATE_GETSCALAR(float)
INSTANTIATE_GETSCALAR(double)
INSTANTIATE_GETSCALAR(cfloat)
INSTANTIATE_GETSCALAR(cdouble)
INSTANTIATE_GETSCALAR(int)
INSTANTIATE_GETSCALAR(uint)
INSTANTIATE_GETSCALAR(uchar)
INSTANTIATE_GETSCALAR(char)
INSTANTIATE_GETSCALAR(intl)
INSTANTIATE_GETSCALAR(uintl)
INSTANTIATE_GETSCALAR(short)
INSTANTIATE_GETSCALAR(ushort)
INSTANTIATE_GETSCALAR(half)
}  // namespace cpu
}  // namespace flare
