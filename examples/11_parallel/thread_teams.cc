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

#include <flare/core.h>
#include <cstdio>

// Using default execution space define a TeamPolicy and its member_type
// The member_type is what the operator of a functor or Lambda gets, for
// a simple RangePolicy the member_type is simply an integer
// For a TeamPolicy its a much richer object, since it provides all information
// to identify a thread uniquely and some team related function calls such as a
// barrier (which will be used in a subsequent example).
// A ThreadTeam consists of 1 to n threads where the maxmimum value of n is
// determined by the hardware. On a dual socket CPU machine with 8 cores per
// socket the maximum size of a team is 8. The number of teams (i.e. the
// league_size) is not limited by physical constraints. Its a pure logical
// number.

using team_policy = flare::TeamPolicy<>;
using team_member = team_policy::member_type;

// Define a functor which can be launched using the TeamPolicy
struct hello_world {
  using value_type = int;  // Specify value type for reduction target, sum

  // This is a reduction operator which now takes as first argument the
  // TeamPolicy member_type. Every member of the team contributes to the
  // total sum.
  // It is helpful to think of this operator as a parallel region for a team
  // (i.e. every team member is active and will execute the code).
  FLARE_INLINE_FUNCTION
  void operator()(const team_member& thread, int& sum) const {
    sum += 1;
    // The TeamPolicy<>::member_type provides functions to query the multi
    // dimensional index of a thread as well as the number of thread-teams and
    // the size of each team.
    printf("Hello World: %i %i // %i %i\n", thread.league_rank(),
           thread.team_rank(), thread.league_size(), thread.team_size());
  }
};

int main(int narg, char* args[]) {
  flare::initialize(narg, args);

  // Launch 12 teams of the maximum number of threads per team
  const int team_size_max = team_policy(1, 1).team_size_max(
      hello_world(), flare::ParallelReduceTag());
  const team_policy policy_a(12, team_size_max);

  int sum = 0;
  flare::parallel_reduce(policy_a, hello_world(), sum);

  // The result will be 12*team_size_max
  printf("Result A: %i == %i\n", sum, team_size_max * 12);

  // In practice it is often better to let flare decide on the team_size
  const team_policy policy_b(12, flare::AUTO);

  flare::parallel_reduce(policy_b, hello_world(), sum);
  // The result will be 12*policy_b.team_size_recommended( hello_world(),
  // flare::ParallelReduceTag())
  const int team_size_recommended = policy_b.team_size_recommended(
      hello_world(), flare::ParallelReduceTag());
  printf("Result B: %i %i\n", sum, team_size_recommended * 12);

  flare::finalize();
}
