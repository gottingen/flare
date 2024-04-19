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

#include <blas.hpp>
#include <cholesky.hpp>
#include <copy.hpp>
#include <err_opencl.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <cpu/cpu_cholesky.hpp>
#include <magma/magma.h>
#include <triangle.hpp>

namespace flare {
namespace opencl {

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper) {
    if (OpenCLCPUOffload()) { return cpu::cholesky_inplace(in, is_upper); }

    dim4 iDims = in.dims();
    int N      = iDims[0];

    magma_uplo_t uplo = is_upper ? MagmaUpper : MagmaLower;

    int info           = 0;
    cl::Buffer *in_buf = in.get();
    magma_potrf_gpu<T>(uplo, N, (*in_buf)(), in.getOffset(), in.strides()[1],
                       getQueue()(), &info);
    return info;
}

template<typename T>
Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper) {
    if (OpenCLCPUOffload()) { return cpu::cholesky(info, in, is_upper); }

    Array<T> out = copyArray<T>(in);
    *info        = cholesky_inplace(out, is_upper);

    triangle<T>(out, out, is_upper, false);

    return out;
}

#define INSTANTIATE_CH(T)                                                 \
    template int cholesky_inplace<T>(Array<T> & in, const bool is_upper); \
    template Array<T> cholesky<T>(int *info, const Array<T> &in,          \
                                  const bool is_upper);

INSTANTIATE_CH(float)
INSTANTIATE_CH(cfloat)
INSTANTIATE_CH(double)
INSTANTIATE_CH(cdouble)

}  // namespace opencl
}  // namespace flare

#else  // WITH_LINEAR_ALGEBRA

namespace flare {
namespace opencl {

template<typename T>
Array<T> cholesky(int *info, const Array<T> &in, const bool is_upper) {
    FLY_ERROR("Linear Algebra is disabled on OpenCL", FLY_ERR_NOT_CONFIGURED);
}

template<typename T>
int cholesky_inplace(Array<T> &in, const bool is_upper) {
    FLY_ERROR("Linear Algebra is disabled on OpenCL", FLY_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE_CH(T)                                                 \
    template int cholesky_inplace<T>(Array<T> & in, const bool is_upper); \
    template Array<T> cholesky<T>(int *info, const Array<T> &in,          \
                                  const bool is_upper);

INSTANTIATE_CH(float)
INSTANTIATE_CH(cfloat)
INSTANTIATE_CH(double)
INSTANTIATE_CH(cdouble)

}  // namespace opencl
}  // namespace flare

#endif  // WITH_LINEAR_ALGEBRA
