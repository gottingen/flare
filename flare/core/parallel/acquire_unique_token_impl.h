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

#ifndef FLARE_CORE_PARALLEL_ACQUIRE_UNIQUE_TOKEN_IMPL_H_
#define FLARE_CORE_PARALLEL_ACQUIRE_UNIQUE_TOKEN_IMPL_H_

#include <flare/core.h>
#include <flare/core/parallel/unique_token.h>

namespace flare {

    template<typename TeamPolicy>
    FLARE_FUNCTION AcquireTeamUniqueToken<TeamPolicy>::AcquireTeamUniqueToken(
            AcquireTeamUniqueToken<TeamPolicy>::token_type t, team_member_type team)
            : my_token(t), my_team_acquired_val(team.team_scratch(0)), my_team(team) {
        flare::single(flare::PerTeam(my_team),
                      [&]() { my_team_acquired_val() = my_token.acquire(); });
        my_team.team_barrier();

        my_acquired_val = my_team_acquired_val();
    }

    template<typename TeamPolicy>
    FLARE_FUNCTION AcquireTeamUniqueToken<TeamPolicy>::~AcquireTeamUniqueToken() {
        my_team.team_barrier();
        flare::single(flare::PerTeam(my_team),
                      [&]() { my_token.release(my_acquired_val); });
        my_team.team_barrier();
    }

}  // namespace flare

#endif  // FLARE_CORE_PARALLEL_ACQUIRE_UNIQUE_TOKEN_IMPL_H_
