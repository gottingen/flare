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

#ifndef FLARE_BACKEND_CUDA_CUDA_MDRANGE_POLICY_H_
#define FLARE_BACKEND_CUDA_CUDA_MDRANGE_POLICY_H_

#include <flare/core/policy/exp_mdrange_policy.h>

namespace flare {

template <>
struct default_outer_direction<flare::Cuda> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

template <>
struct default_inner_direction<flare::Cuda> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

namespace detail {

// Settings for MDRangePolicy
template <>
inline TileSizeProperties get_tile_size_properties<flare::Cuda>(
    const flare::Cuda& space) {
  TileSizeProperties properties;
  properties.max_threads =
      space.impl_internal_space_instance()->m_maxThreadsPerSM;
  properties.default_largest_tile_size = 16;
  properties.default_tile_size         = 2;
  properties.max_total_tile_size       = 512;
  return properties;
}

// Settings for TeamMDRangePolicy
template <typename Rank, TeamMDRangeThreadAndVector ThreadAndVector>
struct ThreadAndVectorNestLevel<Rank, Cuda, ThreadAndVector>
    : AcceleratorBasedNestLevel<Rank, ThreadAndVector> {};

}  // Namespace detail
}  // Namespace flare
#endif  // FLARE_BACKEND_CUDA_CUDA_MDRANGE_POLICY_H_
