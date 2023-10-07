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

#ifndef FLARE_KERNEL_BLAS_SET_IMPL_H_
#define FLARE_KERNEL_BLAS_SET_IMPL_H_

#include <flare/core.h>

namespace flare::blas::detail {

    ///
    /// Serial Internal Impl
    /// ====================
    struct SerialSetInternal {
        template <typename ScalarType, typename ValueType>
        FLARE_INLINE_FUNCTION static int invoke(const int m, const ScalarType alpha,
                /* */ ValueType *FLARE_RESTRICT A,
                                                 const int as0) {
#if defined(FLARE_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
            for (int i = 0; i < m; ++i) A[i * as0] = alpha;

            return 0;
        }

        template <typename ScalarType, typename ValueType>
        FLARE_INLINE_FUNCTION static int invoke(const int m, const int n,
                                                 const ScalarType alpha,
                /* */ ValueType *FLARE_RESTRICT A,
                                                 const int as0, const int as1) {
            if (as0 > as1)
                for (int i = 0; i < m; ++i) invoke(n, alpha, A + i * as0, as1);
            else
                for (int j = 0; j < n; ++j) invoke(m, alpha, A + j * as1, as0);

            return 0;
        }
    };

    ///
    /// Team Internal Impl
    /// ==================
    struct TeamSetInternal {
        template <typename MemberType, typename ScalarType, typename ValueType>
        FLARE_INLINE_FUNCTION static int invoke(const MemberType &member,
                                                 const int m, const ScalarType alpha,
                /* */ ValueType *FLARE_RESTRICT A,
                                                 const int as0) {
            flare::parallel_for(flare::TeamThreadRange(member, m),
                                 [&](const int &i) { A[i * as0] = alpha; });
            // member.team_barrier();
            return 0;
        }

        template <typename MemberType, typename ScalarType, typename ValueType>
        FLARE_INLINE_FUNCTION static int invoke(const MemberType &member,
                                                 const int m, const int n,
                                                 const ScalarType alpha,
                /* */ ValueType *FLARE_RESTRICT A,
                                                 const int as0, const int as1) {
            if (m > n) {
                flare::parallel_for(
                        flare::TeamThreadRange(member, m), [&](const int &i) {
                            SerialSetInternal::invoke(n, alpha, A + i * as0, as1);
                        });
            } else {
                flare::parallel_for(
                        flare::TeamThreadRange(member, n), [&](const int &j) {
                            SerialSetInternal::invoke(m, alpha, A + j * as1, as0);
                        });
            }
            // member.team_barrier();
            return 0;
        }
    };

    ///
    /// TeamVector Internal Impl
    /// ========================
    struct TeamVectorSetInternal {
        template <typename MemberType, typename ScalarType, typename ValueType>
        FLARE_INLINE_FUNCTION static int invoke(const MemberType &member,
                                                 const int m, const ScalarType alpha,
                /* */ ValueType *FLARE_RESTRICT A,
                                                 const int as0) {
            flare::parallel_for(flare::TeamVectorRange(member, m),
                                 [&](const int &i) { A[i * as0] = alpha; });
            // member.team_barrier();
            return 0;
        }

        template <typename MemberType, typename ScalarType, typename ValueType>
        FLARE_INLINE_FUNCTION static int invoke(const MemberType &member,
                                                 const int m, const int n,
                                                 const ScalarType alpha,
                /* */ ValueType *FLARE_RESTRICT A,
                                                 const int as0, const int as1) {
            if (m > n) {
                flare::parallel_for(
                        flare::TeamThreadRange(member, m), [&](const int &i) {
                            flare::parallel_for(
                                    flare::ThreadVectorRange(member, n),
                                    [&](const int &j) { A[i * as0 + j * as1] = alpha; });
                        });
            } else {
                flare::parallel_for(
                        flare::ThreadVectorRange(member, m), [&](const int &i) {
                            flare::parallel_for(
                                    flare::TeamThreadRange(member, n),
                                    [&](const int &j) { A[i * as0 + j * as1] = alpha; });
                        });
            }
            // member.team_barrier();
            return 0;
        }
    };

}  // flare::blas::detail

#endif  // FLARE_KERNEL_BLAS_SET_IMPL_H_
