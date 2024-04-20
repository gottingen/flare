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

#include <common/err_common.hpp>
#include <inverse.hpp>

#if defined(WITH_LINEAR_ALGEBRA)

#include <err_cpu.hpp>
#include <handle.hpp>
#include <range.hpp>
#include <fly/dim4.hpp>
#include <cassert>

#include <identity.hpp>
#include <lapack_helper.hpp>
#include <lu.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <solve.hpp>

namespace flare {
namespace cpu {

template<typename T>
using getri_func_def = int (*)(ORDER_TYPE, int, T *, int, const int *);

#define INV_FUNC_DEF(FUNC) \
    template<typename T>   \
    FUNC##_func_def<T> FUNC##_func();

#define INV_FUNC(FUNC, TYPE, PREFIX)            \
    template<>                                  \
    FUNC##_func_def<TYPE> FUNC##_func<TYPE>() { \
        return &LAPACK_NAME(PREFIX##FUNC);      \
    }

INV_FUNC_DEF(getri)
INV_FUNC(getri, float, s)
INV_FUNC(getri, double, d)
INV_FUNC(getri, cfloat, c)
INV_FUNC(getri, cdouble, z)

template<typename T>
Array<T> inverse(const Array<T> &in) {
    int M = in.dims()[0];
    int N = in.dims()[1];

    if (M != N) {
        Array<T> I = identity<T>(in.dims());
        return solve(in, I);
    }

    Array<T> A       = copyArray<T>(in);
    Array<int> pivot = lu_inplace<T>(A, false);

    auto func = [=](Param<T> A, Param<int> pivot, int M) {
        getri_func<T>()(FLY_LAPACK_COL_MAJOR, M, A.get(), A.strides(1),
                        pivot.get());
    };
    getQueue().enqueue(func, A, pivot, M);

    return A;
}

#define INSTANTIATE(T) template Array<T> inverse<T>(const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)

}  // namespace cpu
}  // namespace flare

#else  // WITH_LINEAR_ALGEBRA

namespace flare {
namespace cpu {

template<typename T>
Array<T> inverse(const Array<T> &in) {
    FLY_ERROR("Linear Algebra is disabled on CPU", FLY_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE(T) template Array<T> inverse<T>(const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)

}  // namespace cpu
}  // namespace flare

#endif  // WITH_LINEAR_ALGEBRA
