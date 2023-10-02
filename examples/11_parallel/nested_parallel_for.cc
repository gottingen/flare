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

// See 01_thread_teams for an explanation of a basic TeamPolicy
using team_policy = flare::TeamPolicy<>;
using team_member = typename team_policy::member_type;

struct hello_world {
  using value_type = int;  // Specify value type for reduction target, sum
  FLARE_INLINE_FUNCTION
  void operator()(const team_member& thread, int& sum) const {
    sum += 1;
    // When using the TeamPolicy flare allows for nested parallel loops.
    // All three flare parallel patterns are allowed (for, reduce, scan) and
    // they largely follow the same syntax as on the global level. The execution
    // policy for the Thread level nesting (the Vector level is in the next
    // tutorial example) is flare::TeamThreadRange. This means the loop will be
    // executed by all members of the team and the loop count will be split
    // between threads of the team. Its arguments are the team_member, and a
    // loop count. Not every thread will do the same amount of iterations. On a
    // GPU for example with a team_size() larger than 31 only the first 31
    // threads would actually do anything. On a CPU with 8 threads 7 would
    // execute 4 loop iterations, and 1 thread would do
    // 3. Note also that the mode of splitting the count is architecture
    // dependent similar to what the RangePolicy on a global level does. The
    // call itself is not guaranteed to be synchronous. Also keep in mind that
    // the operator using a team_policy acts like a parallel region for the
    // team. That means that everything outside of the nested parallel_for is
    // also executed by all threads of the team.
    flare::parallel_for(flare::TeamThreadRange(thread, 31),
                         [&](const int& i) {
                           printf("Hello World: (%i , %i) executed loop %i \n",
                                  thread.league_rank(), thread.team_rank(), i);
                         });
  }
};

int main(int narg, char* args[]) {
  flare::initialize(narg, args);

  // Launch 3 teams of the maximum number of threads per team
  const int team_size_max = team_policy(3, 1).team_size_max(
      hello_world(), flare::ParallelReduceTag());
  const team_policy policy(3, team_size_max);

  int sum = 0;
  flare::parallel_reduce(policy, hello_world(), sum);
  printf("Result %i\n", sum);

  flare::finalize();
}
