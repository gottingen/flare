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
#include <gtest/gtest.h>

namespace {


struct TeamTeamCombinedReducer {
 public:
  void test_team_thread_range_only_scalars(const int n) {
    auto policy = flare::TeamPolicy<TEST_EXECSPACE>(1, flare::AUTO);
    using team_member_type = decltype(policy)::member_type;

    auto teamView = flare::View<int[4], TEST_EXECSPACE::memory_space>("view");

    flare::parallel_for(
        policy, FLARE_LAMBDA(team_member_type const& team) {
          auto teamThreadRange = flare::TeamThreadRange(team, n);
          int teamResult0, teamResult1, teamResult2, teamResult3;

          flare::parallel_reduce(
              teamThreadRange,
              [=](int const& i, int& localVal0, int& localVal1, int& localVal2,
                  int& localVal3) {
                localVal0 += 1;
                localVal1 += i + 1;
                localVal2 += (i + 1) * n;
                localVal3 += n;
              },
              teamResult0, teamResult1, teamResult2, teamResult3);

          flare::single(flare::PerTeam(team), [=]() {
            teamView(0) = teamResult0;
            teamView(1) = teamResult1;
            teamView(2) = teamResult2;
            teamView(3) = teamResult3;
          });
        });

    auto hostView = flare::create_mirror_view_and_copy(
        flare::DefaultHostExecutionSpace(), teamView);

    if (n == 0) {
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(0));
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(1));
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(2));
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(3));
    } else {
      EXPECT_EQ(n, hostView(0));
      EXPECT_EQ((n * (n + 1) / 2), hostView(1));
      EXPECT_EQ(n * n * (n + 1) / 2, hostView(2));
      EXPECT_EQ(n * n, hostView(3));
    }
  }

  void test_team_thread_range_only_builtin(const int n) {
    auto policy = flare::TeamPolicy<TEST_EXECSPACE>(1, flare::AUTO);
    using team_member_type = decltype(policy)::member_type;

    auto teamView = flare::View<int[4], TEST_EXECSPACE::memory_space>("view");

    flare::parallel_for(
        policy, FLARE_LAMBDA(team_member_type const& team) {
          auto teamThreadRange = flare::TeamThreadRange(team, n);
          int teamResult0, teamResult1, teamResult2, teamResult3;

          flare::parallel_reduce(
              teamThreadRange,
              [=](int const& i, int& localVal0, int& localVal1, int& localVal2,
                  int& localVal3) {
                localVal0 += i + 1;
                localVal1 *= n;
                localVal2 = (localVal2 > (i + 1)) ? (i + 1) : localVal2;
                localVal3 = (localVal3 < (i + 1)) ? (i + 1) : localVal3;
              },
              flare::Sum<int>(teamResult0), flare::Prod<int>(teamResult1),
              flare::Min<int>(teamResult2), flare::Max<int>(teamResult3));

          flare::single(flare::PerTeam(team), [=]() {
            teamView(0) = teamResult0;
            teamView(1) = teamResult1;
            teamView(2) = teamResult2;
            teamView(3) = teamResult3;
          });
        });

    auto hostView = flare::create_mirror_view_and_copy(
        flare::DefaultHostExecutionSpace(), teamView);

    if (n == 0) {
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(0));
      EXPECT_EQ(flare::reduction_identity<int>::prod(), hostView(1));
      EXPECT_EQ(flare::reduction_identity<int>::min(), hostView(2));
      EXPECT_EQ(flare::reduction_identity<int>::max(), hostView(3));
    } else {
      EXPECT_EQ((n * (n + 1) / 2), hostView(0));
      EXPECT_EQ(std::pow(n, n), hostView(1));
      EXPECT_EQ(1, hostView(2));
      EXPECT_EQ(n, hostView(3));
    }
  }

  void test_team_thread_range_combined_reducers(const int n) {
    auto policy = flare::TeamPolicy<TEST_EXECSPACE>(1, flare::AUTO);
    using team_member_type = decltype(policy)::member_type;

    auto teamView = flare::View<int*, TEST_EXECSPACE::memory_space>("view", 4);

    flare::parallel_for(
        policy, FLARE_LAMBDA(team_member_type const& team) {
          auto teamThreadRange = flare::TeamThreadRange(team, n);
          int teamResult0, teamResult1, teamResult2, teamResult3;

          flare::parallel_reduce(
              teamThreadRange,
              [=](int const& i, int& localVal0, int& localVal1, int& localVal2,
                  int& localVal3) {
                localVal0 += i + 1;
                localVal1 += i + 1;
                localVal2 = (localVal2 < (i + 1)) ? (i + 1) : localVal2;
                localVal3 += n;
              },
              teamResult0, flare::Sum<int>(teamResult1),
              flare::Max<int>(teamResult2), teamResult3);

          flare::single(flare::PerTeam(team), [=]() {
            teamView(0) = teamResult0;
            teamView(1) = teamResult1;
            teamView(2) = teamResult2;
            teamView(3) = teamResult3;
          });
        });

    auto hostView = flare::create_mirror_view_and_copy(
        flare::DefaultHostExecutionSpace(), teamView);

    if (n == 0) {
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(0));
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(1));
      EXPECT_EQ(flare::reduction_identity<int>::max(), hostView(2));
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(3));
    } else {
      EXPECT_EQ((n * (n + 1) / 2), hostView(0));
      EXPECT_EQ((n * (n + 1) / 2), hostView(1));
      EXPECT_EQ(n, hostView(2));
      EXPECT_EQ(n * n, hostView(3));
    }
  }

  void test_thread_vector_range_only_scalars(const int n) {
    auto policy = flare::TeamPolicy<TEST_EXECSPACE>(1, flare::AUTO);
    using team_member_type = decltype(policy)::member_type;

    auto teamView = flare::View<int[4], TEST_EXECSPACE::memory_space>("view");

    flare::parallel_for(
        policy, FLARE_LAMBDA(team_member_type const& team) {
          auto teamThreadRange   = flare::TeamThreadRange(team, 1);
          auto threadVectorRange = flare::ThreadVectorRange(team, n);
          int teamResult0, teamResult1, teamResult2, teamResult3;

          flare::parallel_for(teamThreadRange, [&](int const&) {
            flare::parallel_reduce(
                threadVectorRange,
                [=](int const& i, int& localVal0, int& localVal1,
                    int& localVal2, int& localVal3) {
                  localVal0 += 1;
                  localVal1 += i + 1;
                  localVal2 += (i + 1) * n;
                  localVal3 += n;
                },
                teamResult0, teamResult1, teamResult2, teamResult3);

            teamView(0) = teamResult0;
            teamView(1) = teamResult1;
            teamView(2) = teamResult2;
            teamView(3) = teamResult3;
          });
        });

    auto hostView = flare::create_mirror_view_and_copy(
        flare::DefaultHostExecutionSpace(), teamView);

    if (n == 0) {
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(0));
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(1));
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(2));
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(3));
    } else {
      EXPECT_EQ(n, hostView(0));
      EXPECT_EQ((n * (n + 1) / 2), hostView(1));
      EXPECT_EQ(n * n * (n + 1) / 2, hostView(2));
      EXPECT_EQ(n * n, hostView(3));
    }
  }

  void test_thread_vector_range_only_builtin(const int n) {
    auto policy = flare::TeamPolicy<TEST_EXECSPACE>(1, flare::AUTO);
    using team_member_type = decltype(policy)::member_type;

    auto teamView = flare::View<int[4], TEST_EXECSPACE::memory_space>("view");

    flare::parallel_for(
        policy, FLARE_LAMBDA(team_member_type const& team) {
          auto teamThreadRange   = flare::TeamThreadRange(team, 1);
          auto threadVectorRange = flare::ThreadVectorRange(team, n);
          int teamResult0, teamResult1, teamResult2, teamResult3;

          flare::parallel_for(teamThreadRange, [&](int const&) {
            flare::parallel_reduce(
                threadVectorRange,
                [=](int const& i, int& localVal0, int& localVal1,
                    int& localVal2, int& localVal3) {
                  localVal0 += i + 1;
                  localVal1 *= n;
                  localVal2 = (localVal2 > (i + 1)) ? (i + 1) : localVal2;
                  localVal3 = (localVal3 < (i + 1)) ? (i + 1) : localVal3;
                },
                flare::Sum<int>(teamResult0), flare::Prod<int>(teamResult1),
                flare::Min<int>(teamResult2), flare::Max<int>(teamResult3));

            teamView(0) = teamResult0;
            teamView(1) = teamResult1;
            teamView(2) = teamResult2;
            teamView(3) = teamResult3;
          });
        });

    auto hostView = flare::create_mirror_view_and_copy(
        flare::DefaultHostExecutionSpace(), teamView);

    if (n == 0) {
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(0));
      EXPECT_EQ(flare::reduction_identity<int>::prod(), hostView(1));
      EXPECT_EQ(flare::reduction_identity<int>::min(), hostView(2));
      EXPECT_EQ(flare::reduction_identity<int>::max(), hostView(3));
    } else {
      EXPECT_EQ((n * (n + 1) / 2), hostView(0));
      EXPECT_EQ(std::pow(n, n), hostView(1));
      EXPECT_EQ(1, hostView(2));
      EXPECT_EQ(n, hostView(3));
    }
  }

  void test_thread_vector_range_combined_reducers(const int n) {
    auto policy = flare::TeamPolicy<TEST_EXECSPACE>(1, flare::AUTO);
    using team_member_type = decltype(policy)::member_type;

    auto teamView = flare::View<int[4], TEST_EXECSPACE::memory_space>("view");

    flare::parallel_for(
        policy, FLARE_LAMBDA(team_member_type const& team) {
          auto teamThreadRange   = flare::TeamThreadRange(team, 1);
          auto threadVectorRange = flare::ThreadVectorRange(team, n);
          int teamResult0, teamResult1, teamResult2, teamResult3;

          flare::parallel_for(teamThreadRange, [&](int const&) {
            flare::parallel_reduce(
                threadVectorRange,
                [=](int const& i, int& localVal0, int& localVal1,
                    int& localVal2, int& localVal3) {
                  localVal0 *= n;
                  localVal1 += i + 1;
                  localVal2 = (localVal2 > (i + 1)) ? (i + 1) : localVal2;
                  localVal3 += n;
                },
                flare::Prod<int>(teamResult0), teamResult1,
                flare::Min<int>(teamResult2), teamResult3);

            teamView(0) = teamResult0;
            teamView(1) = teamResult1;
            teamView(2) = teamResult2;
            teamView(3) = teamResult3;
          });
        });

    auto hostView = flare::create_mirror_view_and_copy(
        flare::DefaultHostExecutionSpace(), teamView);

    if (n == 0) {
      EXPECT_EQ(flare::reduction_identity<int>::prod(), hostView(0));
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(1));
      EXPECT_EQ(flare::reduction_identity<int>::min(), hostView(2));
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(3));
    } else {
      EXPECT_EQ(std::pow(n, n), hostView(0));
      EXPECT_EQ((n * (n + 1) / 2), hostView(1));
      EXPECT_EQ(1, hostView(2));
      EXPECT_EQ(n * n, hostView(3));
    }
  }

  void test_team_vector_range_only_scalars(const int n) {
    auto policy = flare::TeamPolicy<TEST_EXECSPACE>(1, flare::AUTO);
    using team_member_type = decltype(policy)::member_type;

    auto teamView = flare::View<int[4], TEST_EXECSPACE::memory_space>("view");

    flare::parallel_for(
        policy, FLARE_LAMBDA(team_member_type const& team) {
          auto teamVectorRange = flare::TeamVectorRange(team, n);
          int teamResult0, teamResult1, teamResult2, teamResult3;

          flare::parallel_reduce(
              teamVectorRange,
              [=](int const& i, int& localVal0, int& localVal1, int& localVal2,
                  int& localVal3) {
                localVal0 += 1;
                localVal1 += i + 1;
                localVal2 += (i + 1) * n;
                localVal3 += n;
              },
              teamResult0, teamResult1, teamResult2, teamResult3);

          flare::single(flare::PerTeam(team), [=]() {
            teamView(0) = teamResult0;
            teamView(1) = teamResult1;
            teamView(2) = teamResult2;
            teamView(3) = teamResult3;
          });
        });

    auto hostView = flare::create_mirror_view_and_copy(
        flare::DefaultHostExecutionSpace(), teamView);

    if (n == 0) {
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(0));
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(1));
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(2));
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(3));
    } else {
      EXPECT_EQ(n, hostView(0));
      EXPECT_EQ((n * (n + 1) / 2), hostView(1));
      EXPECT_EQ(n * n * (n + 1) / 2, hostView(2));
      EXPECT_EQ(n * n, hostView(3));
    }
  }

  void test_team_vector_range_only_builtin(const int n) {
    auto policy = flare::TeamPolicy<TEST_EXECSPACE>(1, flare::AUTO);
    using team_member_type = decltype(policy)::member_type;

    auto teamView = flare::View<int[4], TEST_EXECSPACE::memory_space>("view");

    flare::parallel_for(
        policy, FLARE_LAMBDA(team_member_type const& team) {
          auto teamVectorRange = flare::TeamVectorRange(team, n);
          int teamResult0, teamResult1, teamResult2, teamResult3;

          flare::parallel_reduce(
              teamVectorRange,
              [=](int const& i, int& localVal0, int& localVal1, int& localVal2,
                  int& localVal3) {
                localVal0 += i + 1;
                localVal1 *= n;
                localVal2 = (localVal2 > (i + 1)) ? (i + 1) : localVal2;
                localVal3 = (localVal3 < (i + 1)) ? (i + 1) : localVal3;
              },
              flare::Sum<int>(teamResult0), flare::Prod<int>(teamResult1),
              flare::Min<int>(teamResult2), flare::Max<int>(teamResult3));

          flare::single(flare::PerTeam(team), [=]() {
            teamView(0) = teamResult0;
            teamView(1) = teamResult1;
            teamView(2) = teamResult2;
            teamView(3) = teamResult3;
          });
        });

    auto hostView = flare::create_mirror_view_and_copy(
        flare::DefaultHostExecutionSpace(), teamView);

    if (n == 0) {
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(0));
      EXPECT_EQ(flare::reduction_identity<int>::prod(), hostView(1));
      EXPECT_EQ(flare::reduction_identity<int>::min(), hostView(2));
      EXPECT_EQ(flare::reduction_identity<int>::max(), hostView(3));
    } else {
      EXPECT_EQ((n * (n + 1) / 2), hostView(0));
      EXPECT_EQ(std::pow(n, n), hostView(1));
      EXPECT_EQ(1, hostView(2));
      EXPECT_EQ(n, hostView(3));
    }
  }

  void test_team_vector_range_combined_reducers(const int n) {
    auto policy = flare::TeamPolicy<TEST_EXECSPACE>(1, flare::AUTO);
    using team_member_type = decltype(policy)::member_type;

    auto teamView = flare::View<int[4], TEST_EXECSPACE::memory_space>("view");

    flare::parallel_for(
        policy, FLARE_LAMBDA(team_member_type const& team) {
          auto teamVectorRange = flare::TeamVectorRange(team, n);
          int teamResult0, teamResult1, teamResult2, teamResult3;

          flare::parallel_reduce(
              teamVectorRange,
              [=](int const& i, int& localVal0, int& localVal1, int& localVal2,
                  int& localVal3) {
                localVal0 += i + 1;
                localVal1 += i + 1;
                localVal2 = (localVal2 < (i + 1)) ? (i + 1) : localVal2;
                localVal3 += n;
              },
              teamResult0, flare::Sum<int>(teamResult1),
              flare::Max<int>(teamResult2), teamResult3);

          flare::single(flare::PerTeam(team), [=]() {
            teamView(0) = teamResult0;
            teamView(1) = teamResult1;
            teamView(2) = teamResult2;
            teamView(3) = teamResult3;
          });
        });

    auto hostView = flare::create_mirror_view_and_copy(
        flare::DefaultHostExecutionSpace(), teamView);

    if (n == 0) {
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(0));
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(1));
      EXPECT_EQ(flare::reduction_identity<int>::max(), hostView(2));
      EXPECT_EQ(flare::reduction_identity<int>::sum(), hostView(3));
    } else {
      EXPECT_EQ((n * (n + 1) / 2), hostView(0));
      EXPECT_EQ((n * (n + 1) / 2), hostView(1));
      EXPECT_EQ(n, hostView(2));
      EXPECT_EQ(n * n, hostView(3));
    }
  }
};

TEST(TEST_CATEGORY, team_thread_range_combined_reducers) {
  TeamTeamCombinedReducer tester;
  tester.test_team_thread_range_only_scalars(5);
  tester.test_team_thread_range_only_builtin(7);
  tester.test_team_thread_range_combined_reducers(0);
  tester.test_team_thread_range_combined_reducers(9);
}

TEST(TEST_CATEGORY, thread_vector_range_combined_reducers) {

  TeamTeamCombinedReducer tester;
  tester.test_thread_vector_range_only_scalars(5);
  tester.test_thread_vector_range_only_builtin(7);
  tester.test_thread_vector_range_combined_reducers(0);
  tester.test_thread_vector_range_combined_reducers(9);
}

TEST(TEST_CATEGORY, team_vector_range_combined_reducers) {


  TeamTeamCombinedReducer tester;
  tester.test_team_vector_range_only_scalars(5);
  tester.test_team_vector_range_only_builtin(7);
  tester.test_team_vector_range_combined_reducers(0);
  tester.test_team_vector_range_combined_reducers(9);
}


}  // namespace
