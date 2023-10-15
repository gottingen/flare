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

#ifndef FLARE_ANN_DISTANCE_L1_IMPL_H_
#define FLARE_ANN_DISTANCE_L1_IMPL_H_

#include <flare/core.h>
#include <flare/core/arith_traits.h>
#include <flare/simd/simd.h>
#include <flare/core/layout_utility.h>
#include <flare/kernel/dense/distance_l1.h>
#include <flare/kernel/dense/batch_distance_l1.h>

namespace flare::ann::detail {

    template<typename execution_space, typename RV, typename XV>
    struct DistanceL1 {
        using size_type = typename XV::size_type;

        static void distance(const execution_space &space, const RV &R, const XV &X, const XV &Y) {
            static_assert(flare::is_tensor<RV>::value,
                          "flare::ann::detail::"
                          "DistanceL1<1-D>: RV is not a flare::Tensor.");
            static_assert(flare::is_tensor<XV>::value,
                          "flare::ann::detail::"
                          "DistanceL1<1-D>: XV is not a flare::Tensor.");
            static_assert(RV::rank == 0,
                          "flare::ann::detail::DistanceL1<1-D>: "
                          "RV is not rank 0.");
            static_assert(XV::rank == 1,
                          "flare::ann::detail::DistanceL1<1-D>: "
                          "XV is not rank 1.");
            flare::Profiling::pushRegion("flare::ann::DistanceL1");
            const size_type numRows = X.extent(0);

            if (numRows < static_cast<size_type>(INT_MAX)) {
                flare::kernel::dense::DistanceL1Invoke<execution_space, RV, XV, int>(space, R, X, Y);
            } else {
                using index_type = std::int64_t;
                flare::kernel::dense::DistanceL1Invoke<execution_space, RV, XV, index_type>(space, R, X, Y);
            }
            flare::Profiling::popRegion();
        }

        static void batch_distance(const execution_space &space, const RV &R, const XV &X, const XV &Y) {
            static_assert(flare::is_tensor<RV>::value,
                          "flare::ann::detail::"
                          "BatchDistanceL1<1-D>: RV is not a flare::Tensor.");
            static_assert(flare::is_tensor<XV>::value,
                          "flare::ann::detail::"
                          "DistanceL1<1-D>: XV is not a flare::Tensor.");
            static_assert(RV::rank == 0,
                          "flare::ann::detail::BatchDistanceL1<1-D>: "
                          "RV is not rank 0.");
            static_assert(XV::rank == 1,
                          "flare::ann::detail::BatchDistanceL1<1-D>: "
                          "XV is not rank 1.");
            flare::Profiling::pushRegion("flare::ann::BatchDistanceL1");
            const size_type numRows = X.extent(0);

            if (numRows < static_cast<size_type>(INT_MAX)) {
                FLARE_IF_ON_DEVICE((flare::kernel::dense::DistanceL1Invoke<execution_space, RV, XV, int>(space, R, X, Y);))
                FLARE_IF_ON_HOST((flare::kernel::dense::DistanceL1BatchInvoke<execution_space, RV, XV, int>(space, R, X, Y);))
            } else {
                using index_type = std::int64_t;
                FLARE_IF_ON_HOST((flare::kernel::dense::DistanceL1BatchInvoke<execution_space, RV, XV, index_type>(space, R, X, Y);))
                FLARE_IF_ON_DEVICE((flare::kernel::dense::DistanceL1Invoke<execution_space, RV, XV, index_type>(space, R, X, Y);))
            }
            flare::Profiling::popRegion();
        }
    };

}  // namespace flare::ann::detail

#endif  // FLARE_ANN_DISTANCE_L1_IMPL_H_
