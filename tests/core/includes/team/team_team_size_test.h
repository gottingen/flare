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

#include <cstdio>
#include <sstream>
#include <iostream>

#include <flare/core.h>

namespace Test {

namespace {
template <class T, int N>
class MyArray {
 public:
  T values[N];
  FLARE_INLINE_FUNCTION
  void operator+=(const MyArray& src) {
    for (int i = 0; i < N; i++) values[i] += src.values[i];
  }
};

template <class T, int N, class PolicyType, int S>
struct FunctorFor {
  double static_array[S];
  FLARE_INLINE_FUNCTION
  void operator()(const typename PolicyType::member_type& /*team*/) const {}
};
template <class T, int N, class PolicyType, int S>
struct FunctorReduce {
  double static_array[S];
  FLARE_INLINE_FUNCTION
  void operator()(const typename PolicyType::member_type& /*team*/,
                  MyArray<T, N>& lval) const {
    for (int j = 0; j < N; j++) lval.values[j] += 1 + lval.values[0];
  }
};
}  // namespace

using policy_type = flare::TeamPolicy<TEST_EXECSPACE>;
using policy_type_128_8 =
    flare::TeamPolicy<TEST_EXECSPACE, flare::LaunchBounds<128, 8> >;
using policy_type_1024_2 =
    flare::TeamPolicy<TEST_EXECSPACE, flare::LaunchBounds<1024, 2> >;

template <class T, int N, class PolicyType, int S>
void test_team_policy_max_recommended_static_size(int scratch_size) {
  PolicyType p = PolicyType(10000, flare::AUTO, 4)
                     .set_scratch_size(0, flare::PerTeam(scratch_size));
  int team_size_max_for = p.team_size_max(FunctorFor<T, N, PolicyType, S>(),
                                          flare::ParallelForTag());
  int team_size_rec_for = p.team_size_recommended(
      FunctorFor<T, N, PolicyType, S>(), flare::ParallelForTag());
  int team_size_max_reduce = p.team_size_max(
      FunctorReduce<T, N, PolicyType, S>(), flare::ParallelReduceTag());
  int team_size_rec_reduce = p.team_size_recommended(
      FunctorReduce<T, N, PolicyType, S>(), flare::ParallelReduceTag());

  ASSERT_GE(team_size_max_for, team_size_rec_for);
  ASSERT_GE(team_size_max_reduce, team_size_rec_reduce);
  ASSERT_GE(team_size_max_for, team_size_max_reduce);

  flare::parallel_for(PolicyType(10000, team_size_max_for, 4)
                           .set_scratch_size(0, flare::PerTeam(scratch_size)),
                       FunctorFor<T, N, PolicyType, S>());
  flare::parallel_for(PolicyType(10000, team_size_rec_for, 4)
                           .set_scratch_size(0, flare::PerTeam(scratch_size)),
                       FunctorFor<T, N, PolicyType, S>());
  MyArray<T, N> val;
  double n_leagues = 10000;

  flare::parallel_reduce(
      PolicyType(n_leagues, team_size_max_reduce, 4)
          .set_scratch_size(0, flare::PerTeam(scratch_size)),
      FunctorReduce<T, N, PolicyType, S>(), val);
  flare::parallel_reduce(
      PolicyType(n_leagues, team_size_rec_reduce, 4)
          .set_scratch_size(0, flare::PerTeam(scratch_size)),
      FunctorReduce<T, N, PolicyType, S>(), val);
  flare::fence();
}

template <class T, int N, class PolicyType>
void test_team_policy_max_recommended(int scratch_size) {
  test_team_policy_max_recommended_static_size<T, N, PolicyType, 1>(
      scratch_size);
  test_team_policy_max_recommended_static_size<T, N, PolicyType, 1000>(
      scratch_size);
}

TEST(TEST_CATEGORY, team_policy_max_recommended) {
  int max_scratch_size = policy_type::scratch_size_max(0);
  test_team_policy_max_recommended<double, 2, policy_type>(0);
  test_team_policy_max_recommended<double, 2, policy_type>(max_scratch_size /
                                                           3);
  test_team_policy_max_recommended<double, 2, policy_type>(max_scratch_size);
  test_team_policy_max_recommended<double, 2, policy_type_128_8>(0);
  test_team_policy_max_recommended<double, 2, policy_type_128_8>(
      max_scratch_size / 3 / 8);
  test_team_policy_max_recommended<double, 2, policy_type_128_8>(
      max_scratch_size / 8);
  test_team_policy_max_recommended<double, 2, policy_type_1024_2>(0);
  test_team_policy_max_recommended<double, 2, policy_type_1024_2>(
      max_scratch_size / 3 / 2);
  test_team_policy_max_recommended<double, 2, policy_type_1024_2>(
      max_scratch_size / 2);

  test_team_policy_max_recommended<double, 16, policy_type>(0);
  test_team_policy_max_recommended<double, 16, policy_type>(max_scratch_size /
                                                            3);
  test_team_policy_max_recommended<double, 16, policy_type>(max_scratch_size);
  test_team_policy_max_recommended<double, 16, policy_type_128_8>(0);
  test_team_policy_max_recommended<double, 16, policy_type_128_8>(
      max_scratch_size / 3 / 8);
  test_team_policy_max_recommended<double, 16, policy_type_128_8>(
      max_scratch_size / 8);
  test_team_policy_max_recommended<double, 16, policy_type_1024_2>(0);
  test_team_policy_max_recommended<double, 16, policy_type_1024_2>(
      max_scratch_size / 3 / 2);
  test_team_policy_max_recommended<double, 16, policy_type_1024_2>(
      max_scratch_size / 2);
}

template <typename TeamHandleType, typename ReducerValueType>
struct MinMaxTeamLeagueRank {
  FLARE_FUNCTION void operator()(const TeamHandleType& team,
                                  ReducerValueType& update) const {
    int const x = team.league_rank();
    if (x < update.min_val) {
      update.min_val = x;
    }
    if (x > update.max_val) {
      update.max_val = x;
    }
  }
};

TEST(TEST_CATEGORY, team_policy_minmax_scalar_without_plus_equal_k) {
  using ExecSpace           = TEST_EXECSPACE;
  using ReducerType         = flare::MinMax<int, flare::HostSpace>;
  using ReducerValueType    = typename ReducerType::value_type;
  using DynamicScheduleType = flare::Schedule<flare::Dynamic>;
  using TeamPolicyType = flare::TeamPolicy<ExecSpace, DynamicScheduleType>;
  using TeamHandleType = typename TeamPolicyType::member_type;

  static constexpr int num_teams = 17;
  ReducerValueType val;
  ReducerType reducer(val);

  TeamPolicyType p(num_teams, flare::AUTO);
  MinMaxTeamLeagueRank<TeamHandleType, ReducerValueType> f1;

  flare::parallel_reduce(p, f1, reducer);
  ASSERT_EQ(val.min_val, 0);
  ASSERT_EQ(val.max_val, num_teams - 1);
}

}  // namespace Test
