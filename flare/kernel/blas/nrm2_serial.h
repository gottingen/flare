// Copyright 2023 The Elastic-AI Authors.
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
//
// Created by jeff on 23-10-8.
//

#ifndef FLARE_KERNEL_BLAS_NRM2_SERIAL_H_
#define FLARE_KERNEL_BLAS_NRM2_SERIAL_H_

#include <flare/core.h>
#include <flare/kernel/common/inner_product_space_traits.h>

namespace flare::blas::detail {

    ///
    /// Serial Internal Impl
    /// ====================
    template <typename ValueType>
    FLARE_INLINE_FUNCTION static
    typename flare::detail::InnerProductSpaceTraits<ValueType>::mag_type
    serial_nrm2(const int m, const ValueType *FLARE_RESTRICT X,
                const int xs0) {
        using IPT       = flare::detail::InnerProductSpaceTraits<ValueType>;
        using norm_type = typename IPT::mag_type;

        norm_type nrm = flare::ArithTraits<norm_type>::zero();

#if defined(FLARE_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for (int i = 0; i < m; ++i)
            nrm += IPT::norm(IPT::dot(X[i * xs0], X[i * xs0]));

        return flare::ArithTraits<norm_type>::sqrt(nrm);
    }

    template <typename ValueType>
    FLARE_INLINE_FUNCTION static void serial_nrm2(
            const int m, const int n, const ValueType *FLARE_RESTRICT X, const int xs0,
            const int xs1,
            typename flare::detail::InnerProductSpaceTraits<ValueType>::mag_type
            *FLARE_RESTRICT R,
            const int ys0) {
        for (int vecIdx = 0; vecIdx < n; ++vecIdx)
            R[vecIdx * ys0] = serial_nrm2(m, X + vecIdx * xs1, xs0);

        return;
    }

}  // namespace flare::blas::detail

#endif  // FLARE_KERNEL_BLAS_NRM2_SERIAL_H_
