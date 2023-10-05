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

#ifndef FLARE_TEST_TEAM_REDUCTION_SCAN_HPP
#define FLARE_TEST_TEAM_REDUCTION_SCAN_HPP
#include <team_test.h>

namespace Test {

TEST(TEST_CATEGORY, team_reduction_scan) {
  TestScanTeam<TEST_EXECSPACE, flare::Schedule<flare::Static> >(0);
  TestScanTeam<TEST_EXECSPACE, flare::Schedule<flare::Dynamic> >(0);
  TestScanTeam<TEST_EXECSPACE, flare::Schedule<flare::Static> >(10);
  TestScanTeam<TEST_EXECSPACE, flare::Schedule<flare::Dynamic> >(10);
  TestScanTeam<TEST_EXECSPACE, flare::Schedule<flare::Static> >(10000);
  TestScanTeam<TEST_EXECSPACE, flare::Schedule<flare::Dynamic> >(10000);
}

TEST(TEST_CATEGORY, team_long_reduce) {
  {
    TestReduceTeam<long, TEST_EXECSPACE, flare::Schedule<flare::Static> >(0);
    TestReduceTeam<long, TEST_EXECSPACE, flare::Schedule<flare::Dynamic> >(0);
    TestReduceTeam<long, TEST_EXECSPACE, flare::Schedule<flare::Static> >(3);
    TestReduceTeam<long, TEST_EXECSPACE, flare::Schedule<flare::Dynamic> >(3);
    TestReduceTeam<long, TEST_EXECSPACE, flare::Schedule<flare::Static> >(
        100000);
    TestReduceTeam<long, TEST_EXECSPACE, flare::Schedule<flare::Dynamic> >(
        100000);
  }
}

TEST(TEST_CATEGORY, team_double_reduce) {
  {
    TestReduceTeam<double, TEST_EXECSPACE, flare::Schedule<flare::Static> >(
        0);
    TestReduceTeam<double, TEST_EXECSPACE, flare::Schedule<flare::Dynamic> >(
        0);
    TestReduceTeam<double, TEST_EXECSPACE, flare::Schedule<flare::Static> >(
        3);
    TestReduceTeam<double, TEST_EXECSPACE, flare::Schedule<flare::Dynamic> >(
        3);
    TestReduceTeam<double, TEST_EXECSPACE, flare::Schedule<flare::Static> >(
        100000);
    TestReduceTeam<double, TEST_EXECSPACE, flare::Schedule<flare::Dynamic> >(
        100000);
  }
}

template <typename ExecutionSpace>
struct DummyTeamReductionFunctor {
  using TeamPolicy     = flare::TeamPolicy<ExecutionSpace>;
  using TeamHandleType = typename TeamPolicy::member_type;

  FLARE_FUNCTION void operator()(const TeamHandleType&, double&) const {}
};

template <typename ExecutionSpace>
void test_team_parallel_reduce(const int num_loop_size) {
  using TeamPolicy = flare::TeamPolicy<ExecutionSpace>;

  using ReducerType = flare::Sum<double>;
  double result     = 10.;
  ReducerType reducer(result);

  const int bytes_per_team   = 0;
  const int bytes_per_thread = 117;

  TeamPolicy team_exec(num_loop_size, flare::AUTO);
  team_exec.set_scratch_size(1, flare::PerTeam(bytes_per_team),
                             flare::PerThread(bytes_per_thread));

  flare::parallel_reduce(team_exec,
                          DummyTeamReductionFunctor<ExecutionSpace>{}, reducer);
  ASSERT_EQ(result, 0.);
}


TEST(TEST_CATEGORY, repeated_team_reduce) {

#ifdef FLARE_IMPL_32BIT
  GTEST_SKIP() << "Failing FLARE_IMPL_32BIT";  // FIXME_32BIT
#endif

  TestRepeatedTeamReduce<TEST_EXECSPACE>();
}

}  // namespace Test
#endif
