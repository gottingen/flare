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

#include <qr.hpp>

#include <err_cpu.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <copy.hpp>
#include <lapack_helper.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <triangle.hpp>
#include <fly/dim4.hpp>

using fly::dim4;

namespace flare {
namespace cpu {

template<typename T>
using geqrf_func_def = int (*)(ORDER_TYPE, int, int, T *, int, T *);

template<typename T>
using gqr_func_def = int (*)(ORDER_TYPE, int, int, int, T *, int, const T *);

#define QR_FUNC_DEF(FUNC) \
    template<typename T>  \
    FUNC##_func_def<T> FUNC##_func();

#define QR_FUNC(FUNC, TYPE, PREFIX)             \
    template<>                                  \
    FUNC##_func_def<TYPE> FUNC##_func<TYPE>() { \
        return &LAPACK_NAME(PREFIX##FUNC);      \
    }

QR_FUNC_DEF(geqrf)
QR_FUNC(geqrf, float, s)
QR_FUNC(geqrf, double, d)
QR_FUNC(geqrf, cfloat, c)
QR_FUNC(geqrf, cdouble, z)

#define GQR_FUNC_DEF(FUNC) \
    template<typename T>   \
    FUNC##_func_def<T> FUNC##_func();

#define GQR_FUNC(FUNC, TYPE, PREFIX)            \
    template<>                                  \
    FUNC##_func_def<TYPE> FUNC##_func<TYPE>() { \
        return &LAPACK_NAME(PREFIX);            \
    }

GQR_FUNC_DEF(gqr)
GQR_FUNC(gqr, float, sorgqr)
GQR_FUNC(gqr, double, dorgqr)
GQR_FUNC(gqr, cfloat, cungqr)
GQR_FUNC(gqr, cdouble, zungqr)

template<typename T>
void qr(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in) {
    dim4 iDims = in.dims();
    int M      = iDims[0];
    int N      = iDims[1];

    const dim4 NullShape(0, 0, 0, 0);

    dim4 endPadding(M - iDims[0], max(M, N) - iDims[1], 0, 0);
    q = (endPadding == NullShape
             ? copyArray(in)
             : padArrayBorders(in, NullShape, endPadding, FLY_PAD_ZERO));
    q.resetDims(iDims);
    t = qr_inplace(q);

    // SPLIT into q and r
    dim4 rdims(M, N);
    r = createEmptyArray<T>(rdims);

    triangle<T>(r, q, true, false);

    auto func = [=](Param<T> q, Param<T> t, int M, int N) {
        gqr_func<T>()(FLY_LAPACK_COL_MAJOR, M, M, min(M, N), q.get(),
                      q.strides(1), t.get());
    };
    q.resetDims(dim4(M, M));
    getQueue().enqueue(func, q, t, M, N);
}

template<typename T>
Array<T> qr_inplace(Array<T> &in) {
    dim4 iDims = in.dims();
    int M      = iDims[0];
    int N      = iDims[1];
    Array<T> t = createEmptyArray<T>(fly::dim4(min(M, N), 1, 1, 1));

    auto func = [=](Param<T> in, Param<T> t, int M, int N) {
        geqrf_func<T>()(FLY_LAPACK_COL_MAJOR, M, N, in.get(), in.strides(1),
                        t.get());
    };
    getQueue().enqueue(func, in, t, M, N);

    return t;
}

}  // namespace cpu
}  // namespace flare

#else  // WITH_LINEAR_ALGEBRA

namespace flare {
namespace cpu {

template<typename T>
void qr(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in) {
    FLY_ERROR("Linear Algebra is disabled on CPU", FLY_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<T> qr_inplace(Array<T> &in) {
    FLY_ERROR("Linear Algebra is disabled on CPU", FLY_ERR_NOT_CONFIGURED);
}

}  // namespace cpu
}  // namespace flare

#endif  // WITH_LINEAR_ALGEBRA

namespace flare {
namespace cpu {

#define INSTANTIATE_QR(T)                                         \
    template Array<T> qr_inplace<T>(Array<T> & in);               \
    template void qr<T>(Array<T> & q, Array<T> & r, Array<T> & t, \
                        const Array<T> &in);

INSTANTIATE_QR(float)
INSTANTIATE_QR(cfloat)
INSTANTIATE_QR(double)
INSTANTIATE_QR(cdouble)

}  // namespace cpu
}  // namespace flare
