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
// Created by jeff on 23-10-7.
//

#ifndef FLARE_KERNEL_BLAS_UTILITY_H_
#define FLARE_KERNEL_BLAS_UTILITY_H_

#include <flare/core/arith_traits.h>

namespace flare::blas {

    //////// Tags for BLAS ////////

    struct Mode {
        struct Serial {
            static const char *name() { return "Serial"; }
        };

        struct Team {
            static const char *name() { return "Team"; }
        };

        struct TeamVector {
            static const char *name() { return "TeamVector"; }
        };
    };

    struct Trans {
        struct Transpose {
        };
        struct NoTranspose {
        };
        struct ConjTranspose {
        };
    };

    struct Algo {
        struct Level3 {
            struct Unblocked {
                static const char *name() { return "Unblocked"; }
            };

            struct Blocked {
                static const char *name() { return "Blocked"; }

                // TODO:: for now harwire the blocksizes; this should reflect
                // register blocking (not about team parallelism).
                // this mb should vary according to
                // - team policy (smaller) or range policy (bigger)
                // - space (gpu vs host)
                // - blocksize input (blk <= 4 mb = 2, otherwise mb = 4), etc.
                static constexpr FLARE_FUNCTION int mb() {
                    FLARE_IF_ON_HOST((return 4;))
                    FLARE_IF_ON_DEVICE((return 2;))
                }
            };

            struct MKL {
                static const char *name() { return "MKL"; }
            };

            struct CompactMKL {
                static const char *name() { return "CompactMKL"; }
            };

            // When this is first developed, unblocked algorithm is a naive
            // implementation and blocked algorithm uses register blocking variant of
            // algorithm (manual unrolling). This distinction is almost meaningless and
            // it just adds more complications. Eventually, the blocked version will be
            // removed and we only use the default algorithm. For testing and
            // development purpose, we still leave algorithm tag in the template
            // arguments.
            using Default = Unblocked;
        };

        using Gemm = Level3;
        using Trsm = Level3;
        using Trmm = Level3;
        using Trtri = Level3;
        using LU = Level3;
        using InverseLU = Level3;
        using SolveLU = Level3;
        using QR = Level3;
        using UTV = Level3;

        struct Level2 {
            struct Unblocked {
            };

            struct Blocked {
                // TODO:: for now hardwire the blocksizes; this should reflect
                // register blocking (not about team parallelism).
                // this mb should vary according to
                // - team policy (smaller) or range policy (bigger)
                // - space (cuda vs host)
                // - blocksize input (blk <= 4 mb = 2, otherwise mb = 4), etc.
                static constexpr FLARE_FUNCTION int mb() {
                    FLARE_IF_ON_HOST((return 4;))
                    FLARE_IF_ON_DEVICE((return 1;))
                }
            };

            struct MKL {
            };
            struct CompactMKL {
            };

            // When this is first developed, unblocked algorithm is a naive
            // implementation and blocked algorithm uses register blocking variant of
            // algorithm (manual unrolling). This distinction is almost meaningless and
            // it just adds more complications. Eventually, the blocked version will be
            // removed and we only use the default algorithm. For testing and
            // development purpose, we still leave algorithm tag in the template
            // arguments.
            using Default = Unblocked;
        };

        using Gemv = Level2;
        using Trsv = Level2;
        using ApplyQ = Level2;
    };
}  // namespace flare::blas
namespace flare::blas::detail {

    // Helper to choose the work distribution for a TeamPolicy computing multiple
    // reductions. Each team computes a partial reduction and atomically contributes
    // to the final result.
    //
    // This was originally written for dot-based GEMM, but can also be applied to
    // multivector dots/norms.

    // Input params:
    //  * length: size of each vector to reduce
    //  * numReductions: number of reductions to compute
    // Output params:
    //  * teamsPerReduction: number of teams to use for each reduction
    template<typename ExecSpace, typename size_type>
    void multipleReductionWorkDistribution(size_type length,
                                           size_type numReductions,
                                           size_type &teamsPerDot) {
        constexpr size_type workPerTeam = 4096;  // Amount of work per team
        size_type appxNumTeams =
                (length * numReductions) / workPerTeam;  // Estimation for appxNumTeams

        // Adjust appxNumTeams in case it is too small or too large
        if (appxNumTeams < 1) appxNumTeams = 1;
        if (appxNumTeams > 1024) appxNumTeams = 1024;

        // If there are more reductions than the number of teams,
        // then set the number of teams to be number of reductions.
        // We don't want a team to contribute to more than one reduction.
        if (numReductions >= appxNumTeams) {
            teamsPerDot = 1;
        }
            // If there are more teams than reductions, each reduction can
            // potentially be performed by multiple teams. First, compute
            // teamsPerDot as an integer (take the floor, not ceiling), then,
            // compute actual number of teams by using this factor.
        else {
            teamsPerDot = appxNumTeams / numReductions;
        }
    }

    // Functor to apply sqrt() to each element of a 1D view.

    template<class RV>
    struct TakeSqrtFunctor {
        TakeSqrtFunctor(const RV &r_) : r(r_) {}

        FLARE_INLINE_FUNCTION void operator()(int i) const {
            r(i) = flare::ArithTraits<typename RV::non_const_value_type>::sqrt(r(i));
        }

        RV r;
    };

}  // namespace flare::blas::detail
#endif  // FLARE_KERNEL_BLAS_UTILITY_H_
