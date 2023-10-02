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

// Demonstrate a parallel reduction using thread teams (TeamPolicy).
//
// A thread team consists of 1 to n threads.  The hardware determines
// the maxmimum value of n. On a dual-socket CPU machine with 8 cores
// per socket, the maximum size of a team is 8. The number of teams
// (the league_size) is not limited by physical constraints (up to
// some reasonable bound, which eventually depends upon the hardware
// and programming model implementation).

int main(int narg, char* args[]) {
  using flare::parallel_reduce;
  using team_policy = flare::TeamPolicy<>;
  using team_member = typename team_policy::member_type;

  flare::initialize(narg, args);

  // Set up a policy that launches 12 teams, with the maximum number
  // of threads per team.

  const team_policy policy(12, flare::AUTO);

  // This is a reduction with a team policy.  The team policy changes
  // the first argument of the lambda.  Rather than an integer index
  // (as with RangePolicy), it's now TeamPolicy::member_type.  This
  // object provides all information to identify a thread uniquely.
  // It also provides some team-related function calls such as a team
  // barrier (which a subsequent example will use).
  //
  // Every member of the team contributes to the total sum.  It is
  // helpful to think of the lambda's body as a "team parallel
  // region."  That is, every team member is active and will execute
  // the body of the lambda.
  int sum = 0;
// We also need to protect the usage of a lambda against compiling
// with a backend which doesn't support it (i.e. Cuda 6.5/7.0).
#if defined(FLARE_ENABLE_CXX11_DISPATCH_LAMBDA)
  parallel_reduce(
      policy,
      FLARE_LAMBDA(const team_member& thread, int& lsum) {
        lsum += 1;
    // TeamPolicy<>::member_type provides functions to query the
    // multidimensional index of a thread, as well as the number of
    // thread teams and the size of each team.
        printf("Hello World: %i %i // %i %i\n", thread.league_rank(),
               thread.team_rank(), thread.league_size(), thread.team_size());
      },
      sum);
#endif
  // The result will be 12*team_policy::team_size_max([=]{})
  printf("Result %i\n", sum);

  flare::finalize();
}
