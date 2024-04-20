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
#include <common/err_common.hpp>
#include <err_cpu.hpp>
#include <svd.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <copy.hpp>
#include <lapack_helper.hpp>
#include <platform.hpp>
#include <queue.hpp>

namespace flare {
namespace cpu {

#define SVD_FUNC_DEF(FUNC)            \
    template<typename T, typename Tr> \
    svd_func_def<T, Tr> svd_func();

#define SVD_FUNC(FUNC, T, Tr, PREFIX)       \
    template<>                              \
    svd_func_def<T, Tr> svd_func<T, Tr>() { \
        return &LAPACK_NAME(PREFIX##FUNC);  \
    }

#if defined(USE_MKL) || defined(__APPLE__)

template<typename T, typename Tr>
using svd_func_def = int (*)(ORDER_TYPE, char jobz, int m, int n, T *in,
                             int ldin, Tr *s, T *u, int ldu, T *vt, int ldvt);

SVD_FUNC_DEF(gesdd)
SVD_FUNC(gesdd, float, float, s)
SVD_FUNC(gesdd, double, double, d)
SVD_FUNC(gesdd, cfloat, float, c)
SVD_FUNC(gesdd, cdouble, double, z)

#else  // Atlas causes memory freeing issues with using gesdd

template<typename T, typename Tr>
using svd_func_def = int (*)(ORDER_TYPE, char jobu, char jobvt, int m, int n,
                             T *in, int ldin, Tr *s, T *u, int ldu, T *vt,
                             int ldvt, Tr *superb);

SVD_FUNC_DEF(gesvd)
SVD_FUNC(gesvd, float, float, s)
SVD_FUNC(gesvd, double, double, d)
SVD_FUNC(gesvd, cfloat, float, c)
SVD_FUNC(gesvd, cdouble, double, z)

#endif

template<typename T, typename Tr>
void svdInPlace(Array<Tr> &s, Array<T> &u, Array<T> &vt, Array<T> &in) {
    auto func = [=](Param<Tr> s, Param<T> u, Param<T> vt, Param<T> in) {
        dim4 iDims = in.dims();
        int M      = iDims[0];
        int N      = iDims[1];

#if defined(USE_MKL) || defined(__APPLE__)
        svd_func<T, Tr>()(FLY_LAPACK_COL_MAJOR, 'A', M, N, in.get(),
                          in.strides(1), s.get(), u.get(), u.strides(1),
                          vt.get(), vt.strides(1));
#else
        std::vector<Tr> superb(std::min(M, N));
        svd_func<T, Tr>()(FLY_LAPACK_COL_MAJOR, 'A', 'A', M, N, in.get(),
                          in.strides(1), s.get(), u.get(), u.strides(1),
                          vt.get(), vt.strides(1), &superb[0]);
#endif
    };
    getQueue().enqueue(func, s, u, vt, in);
}

template<typename T, typename Tr>
void svd(Array<Tr> &s, Array<T> &u, Array<T> &vt, const Array<T> &in) {
    Array<T> in_copy = copyArray<T>(in);
    svdInPlace(s, u, vt, in_copy);
}

}  // namespace cpu
}  // namespace flare

#else  // WITH_LINEAR_ALGEBRA

namespace flare {
namespace cpu {

template<typename T, typename Tr>
void svd(Array<Tr> &s, Array<T> &u, Array<T> &vt, const Array<T> &in) {
    FLY_ERROR("Linear Algebra is disabled on CPU", FLY_ERR_NOT_CONFIGURED);
}

template<typename T, typename Tr>
void svdInPlace(Array<Tr> &s, Array<T> &u, Array<T> &vt, Array<T> &in) {
    FLY_ERROR("Linear Algebra is disabled on CPU", FLY_ERR_NOT_CONFIGURED);
}

}  // namespace cpu
}  // namespace flare

#endif  // WITH_LINEAR_ALGEBRA

namespace flare {
namespace cpu {

#define INSTANTIATE_SVD(T, Tr)                                           \
    template void svd<T, Tr>(Array<Tr> & s, Array<T> & u, Array<T> & vt, \
                             const Array<T> &in);                        \
    template void svdInPlace<T, Tr>(Array<Tr> & s, Array<T> & u,         \
                                    Array<T> & vt, Array<T> & in);

INSTANTIATE_SVD(float, float)
INSTANTIATE_SVD(double, double)
INSTANTIATE_SVD(cfloat, float)
INSTANTIATE_SVD(cdouble, double)

}  // namespace cpu
}  // namespace flare
