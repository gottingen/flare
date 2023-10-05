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

#ifndef TEST_BLOCK_SIZE_DEDUCTION_H_
#define TEST_BLOCK_SIZE_DEDUCTION_H_

#include <flare/core.h>
#include <doctest.h>

struct PoorMansLambda {
  template <typename MemberType>
  FLARE_FUNCTION void operator()(MemberType const&) const {}
};

template <typename ExecutionSpace>
void test_bug_pr_3103() {
  using Policy =
      flare::TeamPolicy<ExecutionSpace, flare::LaunchBounds<32, 1>>;
  int const league_size   = 1;
  int const team_size     = std::min(32, ExecutionSpace().concurrency());
  int const vector_length = 1;

  flare::parallel_for(Policy(league_size, team_size, vector_length),
                       PoorMansLambda());
}

TEST_CASE("TEST_CATEGORY, test_block_deduction_bug_pr_3103") {
  test_bug_pr_3103<TEST_EXECSPACE>();
}

#endif  // TEST_BLOCK_SIZE_DEDUCTION_H_
