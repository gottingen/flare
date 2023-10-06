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

#ifndef FLARE_CORE_POLICY_RANK_H_
#define FLARE_CORE_POLICY_RANK_H_

#include <flare/core/defines.h>
#include <flare/core/memory/layout.h>  // Iterate

namespace flare {

    // Iteration Pattern
    template<unsigned N, Iterate OuterDir = Iterate::Default,
            Iterate InnerDir = Iterate::Default>
    struct Rank {
        static_assert(N != 0u, "flare Error: rank 0 undefined");
        static_assert(N != 1u,
                      "flare Error: rank 1 is not a multi-dimensional range");
        static_assert(N < 9u, "flare Error: Unsupported rank...");

        using iteration_pattern = Rank<N, OuterDir, InnerDir>;

        static constexpr int rank = N;
        static constexpr Iterate outer_direction = OuterDir;
        static constexpr Iterate inner_direction = InnerDir;
    };

}  // end namespace flare

#endif  // FLARE_CORE_POLICY_RANK_H_
