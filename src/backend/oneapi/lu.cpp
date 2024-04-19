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
#include <lu.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <blas.hpp>
#include <copy.hpp>
#include <kernel/lu_split.hpp>
#include <memory.hpp>
#include <oneapi/mkl/lapack.hpp>
#include <platform.hpp>

namespace flare {
namespace oneapi {

Array<int> convertPivot(sycl::buffer<int64_t> &pivot, int in_sz, int out_sz,
                        bool convert_pivot) {
    std::vector<int> d_po(out_sz);
    for (int i = 0; i < out_sz; i++) { d_po[i] = i; }

    auto d_pi = pivot.get_host_access();

    if (convert_pivot) {
        for (int j = 0; j < in_sz; j++) {
            // 1 indexed in pivot
            std::swap(d_po[j], d_po[d_pi[j] - 1]);
        }

        Array<int> res = createHostDataArray(dim4(out_sz), &d_po[0]);
        return res;
    } else {
        d_po.resize(in_sz);
        for (int j = 0; j < in_sz; j++) { d_po[j] = static_cast<int>(d_pi[j]); }
    }
    Array<int> res = createHostDataArray(dim4(in_sz), &d_po[0]);
    return res;
}

template<typename T>
void lu(Array<T> &lower, Array<T> &upper, Array<int> &pivot,
        const Array<T> &in) {
    dim4 iDims = in.dims();
    int M      = iDims[0];
    int N      = iDims[1];
    int MN     = std::min(M, N);

    Array<T> in_copy = copyArray<T>(in);
    pivot            = lu_inplace(in_copy);

    // SPLIT into lower and upper
    dim4 ldims(M, MN);
    dim4 udims(MN, N);
    lower = createEmptyArray<T>(ldims);
    upper = createEmptyArray<T>(udims);
    kernel::lu_split<T>(lower, upper, in_copy);
}

template<typename T>
Array<int> lu_inplace(Array<T> &in, const bool convert_pivot) {
    dim4 iDims    = in.dims();
    dim4 iStrides = in.strides();
    int64_t M     = iDims[0];
    int64_t N     = iDims[1];
    int64_t MN    = std::min(M, N);
    int64_t LDA   = iStrides[1];

    std::int64_t scratchpad_size =
        ::oneapi::mkl::lapack::getrf_scratchpad_size<T>(getQueue(), M, N, LDA);

    auto ipiv       = memAlloc<int64_t>(MN);
    auto scratchpad = memAlloc<compute_t<T>>(scratchpad_size);

    sycl::buffer<compute_t<T>> in_buffer =
        in.template getBufferWithOffset<compute_t<T>>();
    ::oneapi::mkl::lapack::getrf(getQueue(), M, N, in_buffer, LDA, *ipiv,
                                 *scratchpad, scratchpad->size());

    Array<int> pivot = convertPivot(*ipiv, MN, M, convert_pivot);
    return pivot;
}

bool isLAPACKAvailable() { return true; }

#define INSTANTIATE_LU(T)                                        \
    template Array<int> lu_inplace<T>(Array<T> & in,             \
                                      const bool convert_pivot); \
    template void lu<T>(Array<T> & lower, Array<T> & upper,      \
                        Array<int> & pivot, const Array<T> &in);

INSTANTIATE_LU(float)
INSTANTIATE_LU(cfloat)
INSTANTIATE_LU(double)
INSTANTIATE_LU(cdouble)

}  // namespace oneapi
}  // namespace flare

#else  // WITH_LINEAR_ALGEBRA

namespace flare {
namespace oneapi {

template<typename T>
void lu(Array<T> &lower, Array<T> &upper, Array<int> &pivot,
        const Array<T> &in) {
    FLY_ERROR("Linear Algebra is disabled on OneAPI backend",
             FLY_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<int> lu_inplace(Array<T> &in, const bool convert_pivot) {
    FLY_ERROR("Linear Algebra is disabled on OneAPI backend",
             FLY_ERR_NOT_CONFIGURED);
}

bool isLAPACKAvailable() { return false; }

#define INSTANTIATE_LU(T)                                        \
    template Array<int> lu_inplace<T>(Array<T> & in,             \
                                      const bool convert_pivot); \
    template void lu<T>(Array<T> & lower, Array<T> & upper,      \
                        Array<int> & pivot, const Array<T> &in);

INSTANTIATE_LU(float)
INSTANTIATE_LU(cfloat)
INSTANTIATE_LU(double)
INSTANTIATE_LU(cdouble)

}  // namespace oneapi
}  // namespace flare

#endif  // WITH_LINEAR_ALGEBRA
