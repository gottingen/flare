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

#include "flare/core.h"

namespace Test {

    __global__ void offset_streams(int *p) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < 100) {
            p[idx] += idx;
        }
    }

    template<typename MemorySpace>
    struct FunctorRange {
        flare::Tensor<int *, MemorySpace, flare::MemoryTraits<flare::Unmanaged>> a;

        FunctorRange(
                flare::Tensor<int *, MemorySpace, flare::MemoryTraits<flare::Unmanaged>>
                a_)
                : a(a_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const int i) const { a(i) += 1; }
    };

    template<typename MemorySpace>
    struct FunctorRangeReduce {
        flare::Tensor<int *, MemorySpace, flare::MemoryTraits<flare::Unmanaged>> a;

        FunctorRangeReduce(
                flare::Tensor<int *, MemorySpace, flare::MemoryTraits<flare::Unmanaged>>
                a_)
                : a(a_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const int i, int &lsum) const { lsum += a(i); }
    };

    template<typename MemorySpace>
    struct FunctorMDRange {
        flare::Tensor<int *, MemorySpace, flare::MemoryTraits<flare::Unmanaged>> a;

        FunctorMDRange(
                flare::Tensor<int *, MemorySpace, flare::MemoryTraits<flare::Unmanaged>>
                a_)
                : a(a_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const int i, const int j) const { a(i * 10 + j) += 1; }
    };

    template<typename MemorySpace>
    struct FunctorMDRangeReduce {
        flare::Tensor<int *, MemorySpace, flare::MemoryTraits<flare::Unmanaged>> a;

        FunctorMDRangeReduce(
                flare::Tensor<int *, MemorySpace, flare::MemoryTraits<flare::Unmanaged>>
                a_)
                : a(a_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const int i, const int j, int &lsum) const {
            lsum += a(i * 10 + j);
        }
    };

    template<typename MemorySpace, typename ExecutionSpace>
    struct FunctorTeam {
        flare::Tensor<int *, MemorySpace, flare::MemoryTraits<flare::Unmanaged>> a;

        FunctorTeam(
                flare::Tensor<int *, MemorySpace, flare::MemoryTraits<flare::Unmanaged>>
                a_)
                : a(a_) {}

        FLARE_INLINE_FUNCTION
        void operator()(
                typename flare::TeamPolicy<ExecutionSpace>::member_type const &team)
        const {
            int i = team.league_rank();
            flare::parallel_for(flare::TeamThreadRange(team, 10),
                                [&](const int j) { a(i * 10 + j) += 1; });
        }
    };

    template<typename MemorySpace, typename ExecutionSpace>
    struct FunctorTeamReduce {
        flare::Tensor<int *, MemorySpace, flare::MemoryTraits<flare::Unmanaged>> a;

        FunctorTeamReduce(
                flare::Tensor<int *, MemorySpace, flare::MemoryTraits<flare::Unmanaged>>
                a_)
                : a(a_) {}

        FLARE_INLINE_FUNCTION
        void operator()(
                typename flare::TeamPolicy<ExecutionSpace>::member_type const &team,
                int &lsum) const {
            int i = team.league_rank();
            int team_sum;
            flare::parallel_reduce(
                    flare::TeamThreadRange(team, 10),
                    [&](const int j, int &tsum) { tsum += a(i * 10 + j); }, team_sum);
            flare::single(flare::PerTeam(team), [&]() { lsum += team_sum; });
        }
    };
}  // namespace Test
