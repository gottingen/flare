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

#ifndef FLARE_ANN_DISTANCE_TRAITS_H_
#define FLARE_ANN_DISTANCE_TRAITS_H_

#include <flare/core.h>
#if !defined(FLARE_ON_CUDA_DEVICE)
#include <flare/simd/simd.h>
#endif

namespace flare::ann {

    template<typename XVector, typename execution_space = typename XVector::execution_space>
    class DistanceTraits {
    public:
        using value_type = typename XVector::value_type;
        using mag_type =   typename XVector::non_const_value_type;
        using memory_space = typename XVector::memory_space;

        static_assert(flare::is_execution_space_v<execution_space>,
                      "flare::ann::DistanceTraits<1-D>: execution_space is not a Execute space.");

        static_assert(flare::is_view<XVector>::value,
                      "flare::ann::"
                      "DistanceTraits<1-D>: XVector is not a flare::View.");

        static_assert(XVector::rank == 1, "flare::ann::DistanceTraits<1-D>: XVector is not rank 1.");

        static constexpr bool is_batch_available = std::is_same_v<flare::HostSpace, memory_space>;
#if !defined(FLARE_ON_CUDA_DEVICE)
        using batch_type =  flare::simd::batch<mag_type, flare::simd::default_arch>;
        static constexpr size_t batch_size = batch_type::size;
#else
        using batch_type = mag_type;
        static constexpr size_t batch_size = 0ul;
#endif
    };

}  // namespace flare::ann
#endif  // FLARE_ANN_DISTANCE_TRAITS_H_
