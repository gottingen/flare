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

#include <gtest/gtest.h>

#include <flare/core.h>

namespace {

template <typename Policy>
void test_run_time_parameters() {
  int league_size = 131;

  using ExecutionSpace = typename Policy::execution_space;
  int team_size =
      4 < ExecutionSpace().concurrency() ? 4 : ExecutionSpace().concurrency();

  int chunk_size         = 4;
  int per_team_scratch   = 1024;
  int per_thread_scratch = 16;
  int scratch_size       = per_team_scratch + per_thread_scratch * team_size;

  Policy p1(league_size, team_size);
  ASSERT_EQ(p1.league_size(), league_size);
  ASSERT_EQ(p1.team_size(), team_size);
  ASSERT_GT(p1.chunk_size(), 0);
  ASSERT_EQ(p1.scratch_size(0), 0u);

  Policy p2 = p1.set_chunk_size(chunk_size);
  ASSERT_EQ(p1.league_size(), league_size);
  ASSERT_EQ(p1.team_size(), team_size);
  ASSERT_EQ(p1.chunk_size(), chunk_size);
  ASSERT_EQ(p1.scratch_size(0), 0u);

  ASSERT_EQ(p2.league_size(), league_size);
  ASSERT_EQ(p2.team_size(), team_size);
  ASSERT_EQ(p2.chunk_size(), chunk_size);
  ASSERT_EQ(p2.scratch_size(0), 0u);

  Policy p3 = p2.set_scratch_size(0, flare::PerTeam(per_team_scratch));
  ASSERT_EQ(p2.league_size(), league_size);
  ASSERT_EQ(p2.team_size(), team_size);
  ASSERT_EQ(p2.chunk_size(), chunk_size);
  ASSERT_EQ(p2.scratch_size(0), size_t(per_team_scratch));
  ASSERT_EQ(p3.league_size(), league_size);
  ASSERT_EQ(p3.team_size(), team_size);
  ASSERT_EQ(p3.chunk_size(), chunk_size);
  ASSERT_EQ(p3.scratch_size(0), size_t(per_team_scratch));

  Policy p4 = p2.set_scratch_size(0, flare::PerThread(per_thread_scratch));
  ASSERT_EQ(p2.league_size(), league_size);
  ASSERT_EQ(p2.team_size(), team_size);
  ASSERT_EQ(p2.chunk_size(), chunk_size);
  ASSERT_EQ(p2.scratch_size(0), size_t(scratch_size));
  ASSERT_EQ(p4.league_size(), league_size);
  ASSERT_EQ(p4.team_size(), team_size);
  ASSERT_EQ(p4.chunk_size(), chunk_size);
  ASSERT_EQ(p4.scratch_size(0), size_t(scratch_size));

  Policy p5 = p2.set_scratch_size(0, flare::PerThread(per_thread_scratch),
                                  flare::PerTeam(per_team_scratch));
  ASSERT_EQ(p2.league_size(), league_size);
  ASSERT_EQ(p2.team_size(), team_size);
  ASSERT_EQ(p2.chunk_size(), chunk_size);
  ASSERT_EQ(p2.scratch_size(0), size_t(scratch_size));
  ASSERT_EQ(p5.league_size(), league_size);
  ASSERT_EQ(p5.team_size(), team_size);
  ASSERT_EQ(p5.chunk_size(), chunk_size);
  ASSERT_EQ(p5.scratch_size(0), size_t(scratch_size));

  Policy p6 = p2.set_scratch_size(0, flare::PerTeam(per_team_scratch),
                                  flare::PerThread(per_thread_scratch));
  ASSERT_EQ(p2.league_size(), league_size);
  ASSERT_EQ(p2.team_size(), team_size);
  ASSERT_EQ(p2.chunk_size(), chunk_size);
  ASSERT_EQ(p2.scratch_size(0), size_t(scratch_size));
  ASSERT_EQ(p6.league_size(), league_size);
  ASSERT_EQ(p6.team_size(), team_size);
  ASSERT_EQ(p6.chunk_size(), chunk_size);
  ASSERT_EQ(p6.scratch_size(0), size_t(scratch_size));

  Policy p7 = p3.set_scratch_size(0, flare::PerTeam(per_team_scratch),
                                  flare::PerThread(per_thread_scratch));
  ASSERT_EQ(p3.league_size(), league_size);
  ASSERT_EQ(p3.team_size(), team_size);
  ASSERT_EQ(p3.chunk_size(), chunk_size);
  ASSERT_EQ(p3.scratch_size(0), size_t(scratch_size));
  ASSERT_EQ(p7.league_size(), league_size);
  ASSERT_EQ(p7.team_size(), team_size);
  ASSERT_EQ(p7.chunk_size(), chunk_size);
  ASSERT_EQ(p7.scratch_size(0), size_t(scratch_size));

  Policy p8;  // default constructed
  ASSERT_EQ(p8.league_size(), 0);
  ASSERT_EQ(p8.scratch_size(0), 0u);
  p8 = p3;  // call assignment operator
  ASSERT_EQ(p3.league_size(), league_size);
  ASSERT_EQ(p3.team_size(), team_size);
  ASSERT_EQ(p3.chunk_size(), chunk_size);
  ASSERT_EQ(p3.scratch_size(0), size_t(scratch_size));
  ASSERT_EQ(p8.league_size(), league_size);
  ASSERT_EQ(p8.team_size(), team_size);
  ASSERT_EQ(p8.chunk_size(), chunk_size);
  ASSERT_EQ(p8.scratch_size(0), size_t(scratch_size));
}

TEST(TEST_CATEGORY, team_policy_runtime_parameters) {
  struct SomeTag {};

  using TestExecSpace   = TEST_EXECSPACE;
  using DynamicSchedule = flare::Schedule<flare::Dynamic>;
  using LongIndex       = flare::IndexType<long>;

  // clang-format off
  test_run_time_parameters<flare::TeamPolicy<TestExecSpace                                             >>();
  test_run_time_parameters<flare::TeamPolicy<TestExecSpace,   DynamicSchedule, LongIndex               >>();
  test_run_time_parameters<flare::TeamPolicy<LongIndex,       TestExecSpace,   DynamicSchedule         >>();
  test_run_time_parameters<flare::TeamPolicy<DynamicSchedule, LongIndex,       TestExecSpace,   SomeTag>>();
  // clang-format on
}

}  // namespace
