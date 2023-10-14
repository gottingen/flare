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

#ifndef FLARE_KERNEL_BLAS_SET_H_
#define FLARE_KERNEL_BLAS_SET_H_

#include <flare/kernel/blas/set_impl.h>

namespace flare::blas {

    ///
    /// Serial Set
    ///

    struct SerialSet {
        template <typename ScalarType, typename ATensorType>
        FLARE_INLINE_FUNCTION static int invoke(const ScalarType alpha,
                                                 const ATensorType &A) {
            return flare::blas::detail::SerialSetInternal::invoke(
                    A.extent(0), A.extent(1), alpha, A.data(), A.stride_0(), A.stride_1());
        }
    };

    ///
    /// Team Set
    ///

    template <typename MemberType>
    struct TeamSet {
        template <typename ScalarType, typename ATensorType>
        FLARE_INLINE_FUNCTION static int invoke(const MemberType &member,
                                                 const ScalarType alpha,
                                                 const ATensorType &A) {
            return flare::blas::detail::TeamSetInternal::invoke(member, A.extent(0), A.extent(1),
                                                 alpha, A.data(), A.stride_0(),
                                                 A.stride_1());
        }
    };

    ///
    /// TeamVector Set
    ///

    template <typename MemberType>
    struct TeamVectorSet {
        template <typename ScalarType, typename ATensorType>
        FLARE_INLINE_FUNCTION static int invoke(const MemberType &member,
                                                 const ScalarType alpha,
                                                 const ATensorType &A) {
            return flare::blas::detail::TeamVectorSetInternal::invoke(member, A.extent(0), A.extent(1),
                                                       alpha, A.data(), A.stride_0(),
                                                       A.stride_1());
        }
    };
}  // namespace flare::blas

#endif  // FLARE_KERNEL_BLAS_SET_H_
