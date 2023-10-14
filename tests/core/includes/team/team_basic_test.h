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

#ifndef FLARE_TEST_TEAM_BASIC_HPP
#define FLARE_TEST_TEAM_BASIC_HPP
#include <team_test.h>

namespace Test {

TEST(TEST_CATEGORY, team_for) {
  TestTeamPolicy<TEST_EXECSPACE, flare::Schedule<flare::Static> >::test_for(
      0);
  TestTeamPolicy<TEST_EXECSPACE, flare::Schedule<flare::Dynamic> >::test_for(
      0);

  TestTeamPolicy<TEST_EXECSPACE, flare::Schedule<flare::Static> >::test_for(
      2);
  TestTeamPolicy<TEST_EXECSPACE, flare::Schedule<flare::Dynamic> >::test_for(
      2);

  TestTeamPolicy<TEST_EXECSPACE, flare::Schedule<flare::Static> >::test_for(
      1000);
  TestTeamPolicy<TEST_EXECSPACE, flare::Schedule<flare::Dynamic> >::test_for(
      1000);
}

TEST(TEST_CATEGORY, team_reduce) {
  TestTeamPolicy<TEST_EXECSPACE,
                 flare::Schedule<flare::Static> >::test_reduce(0);
  TestTeamPolicy<TEST_EXECSPACE,
                 flare::Schedule<flare::Dynamic> >::test_reduce(0);
  TestTeamPolicy<TEST_EXECSPACE,
                 flare::Schedule<flare::Static> >::test_reduce(2);
  TestTeamPolicy<TEST_EXECSPACE,
                 flare::Schedule<flare::Dynamic> >::test_reduce(2);
  TestTeamPolicy<TEST_EXECSPACE,
                 flare::Schedule<flare::Static> >::test_reduce(1000);
  TestTeamPolicy<TEST_EXECSPACE,
                 flare::Schedule<flare::Dynamic> >::test_reduce(1000);
}

template <typename ExecutionSpace>
struct TestTeamReduceLarge {
  using team_policy_t = flare::TeamPolicy<ExecutionSpace>;
  using member_t      = typename team_policy_t::member_type;

  int m_range;

  TestTeamReduceLarge(const int range) : m_range(range) {}

  FLARE_INLINE_FUNCTION
  void operator()(const member_t& t, int& update) const {
    flare::single(flare::PerTeam(t), [&]() { update++; });
  }

  void run() {
    int result = 0;
    flare::parallel_reduce(team_policy_t(m_range, flare::AUTO), *this,
                            result);
    EXPECT_EQ(m_range, result);
  }
};

TEST(TEST_CATEGORY, team_reduce_large) {
  std::vector<int> ranges{(2LU << 23) - 1, 2LU << 23, (2LU << 24),
                          (2LU << 24) + 1, 1LU << 29};
  for (const auto range : ranges) {
    TestTeamReduceLarge<TEST_EXECSPACE> test(range);
    test.run();
  }
}

/*! \brief Test passing an aggregate to flare::single in a parallel_for with
           team policy
*/
template <typename ExecutionSpace>
struct TestTeamForAggregate {
  using range_policy_t = flare::RangePolicy<ExecutionSpace>;
  using team_policy_t  = flare::TeamPolicy<ExecutionSpace>;
  using member_t       = typename team_policy_t::member_type;
  using memory_space   = typename ExecutionSpace::memory_space;
  using results_type   = flare::Tensor<double*, memory_space>;

  static constexpr double INIT_VALUE   = -1.0;
  static constexpr double EXPECT_VALUE = 1.0;

  struct Agg {
    double d;
  };
  results_type results_;

  TestTeamForAggregate(const size_t size) : results_("results", size) {}
  TestTeamForAggregate() : TestTeamForAggregate(0) {}

  FLARE_INLINE_FUNCTION
  void operator()(const member_t& t) const {
    Agg lagg;
    lagg.d = INIT_VALUE;
    flare::single(
        flare::PerTeam(t), [&](Agg& myAgg) { myAgg.d = EXPECT_VALUE; }, lagg);
    size_t i = t.league_rank() * t.team_size() + t.team_rank();
    if (i < results_.size()) {
      results_(i) = lagg.d;
    }
  }

  FLARE_INLINE_FUNCTION
  void operator()(const int i, int& lNumErrs) const {
    if (EXPECT_VALUE != results_(i)) {
      ++lNumErrs;
    }
  }

  static void run() {
    int minTeamSize = 1;

    int maxTeamSize;
    {
      TestTeamForAggregate test;
      maxTeamSize = team_policy_t(1, minTeamSize)
                        .team_size_max(test, flare::ParallelForTag());
    }

    for (int teamSize = minTeamSize; teamSize <= maxTeamSize; teamSize *= 2) {
      for (int problemSize : {1, 100, 10'000, 1'000'000}) {
        const int leagueSize = (problemSize + teamSize - 1) / teamSize;
        TestTeamForAggregate test(problemSize);
        flare::parallel_for(team_policy_t(leagueSize, teamSize), test);
        int numErrs = 0;
        flare::parallel_reduce(range_policy_t(0, problemSize), test, numErrs);
        EXPECT_EQ(numErrs, 0)
            << " teamSize=" << teamSize << " problemSize=" << problemSize;
      }
    }
  }
};

TEST(TEST_CATEGORY, team_parallel_single) {
  TestTeamForAggregate<TEST_EXECSPACE>::run();
}

template <typename ExecutionSpace>
struct LargeTeamScratchFunctor {
  using team_member = typename flare::TeamPolicy<ExecutionSpace>::member_type;
  const size_t m_per_team_bytes;

  FLARE_FUNCTION void operator()(const team_member& member) const {
    double* team_shared = static_cast<double*>(
        member.team_scratch(/*level*/ 1).get_shmem(m_per_team_bytes));
    if (team_shared == nullptr)
      flare::abort("Couldn't allocate required size!\n");
    double* team_shared_1 = static_cast<double*>(
        member.team_scratch(/*level*/ 1).get_shmem(sizeof(double)));
    if (team_shared_1 != nullptr)
      flare::abort("Allocated more memory than requested!\n");
  }
};

TEST(TEST_CATEGORY, large_team_scratch_size) {
#ifdef FLARE_IMPL_32BIT
  GTEST_SKIP() << "Fails on 32-bit";  // FIXME_32BIT
#endif
  const int level   = 1;
  const int n_teams = 1;

  // Value originally chosen in the reproducer.
  const size_t per_team_extent = 502795560;

  const size_t per_team_bytes = per_team_extent * sizeof(double);

  flare::TeamPolicy<TEST_EXECSPACE> policy(n_teams, 1);
  policy.set_scratch_size(level, flare::PerTeam(per_team_bytes));

  flare::parallel_for(policy,
                       LargeTeamScratchFunctor<TEST_EXECSPACE>{per_team_bytes});
  flare::fence();
}

TEST(TEST_CATEGORY, team_broadcast_long) {
  TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                    long>::test_teambroadcast(0, 1);
  TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                    long>::test_teambroadcast(0, 1);

  TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                    long>::test_teambroadcast(2, 1);
  TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                    long>::test_teambroadcast(2, 1);

  TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                    long>::test_teambroadcast(16, 1);
  TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                    long>::test_teambroadcast(16, 1);

  TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                    long>::test_teambroadcast(1000, 1);
  TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                    long>::test_teambroadcast(1000, 1);
}

struct long_wrapper {
  long value;

  FLARE_FUNCTION
  long_wrapper() : value(0) {}

  FLARE_FUNCTION
  long_wrapper(long val) : value(val) {}

  FLARE_FUNCTION
  long_wrapper(const long_wrapper& val) : value(val.value) {}

  FLARE_FUNCTION
  friend void operator+=(long_wrapper& lhs, const long_wrapper& rhs) {
    lhs.value += rhs.value;
  }

  FLARE_FUNCTION
  void operator=(const long_wrapper& other) { value = other.value; }

  FLARE_FUNCTION
  void operator=(const volatile long_wrapper& other) volatile {
    value = other.value;
  }
  FLARE_FUNCTION
  operator long() const { return value; }
};
}  // namespace Test

namespace flare {
template <>
struct reduction_identity<Test::long_wrapper>
    : public reduction_identity<long> {};
}  // namespace flare

namespace Test {

// Test for non-arithmetic type
TEST(TEST_CATEGORY, team_broadcast_long_wrapper) {
  static_assert(!std::is_arithmetic<long_wrapper>::value, "");

  TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                    long_wrapper>::test_teambroadcast(0, 1);
  TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                    long_wrapper>::test_teambroadcast(0, 1);

  TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                    long_wrapper>::test_teambroadcast(2, 1);
  TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                    long_wrapper>::test_teambroadcast(2, 1);
  TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                    long_wrapper>::test_teambroadcast(16, 1);
  TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                    long_wrapper>::test_teambroadcast(16, 1);

  TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                    long_wrapper>::test_teambroadcast(1000, 1);
  TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                    long_wrapper>::test_teambroadcast(1000, 1);
}

TEST(TEST_CATEGORY, team_broadcast_char) {
  {
    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                      unsigned char>::test_teambroadcast(0, 1);
    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                      unsigned char>::test_teambroadcast(0, 1);

    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                      unsigned char>::test_teambroadcast(2, 1);
    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                      unsigned char>::test_teambroadcast(2, 1);

    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                      unsigned char>::test_teambroadcast(16, 1);
    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                      unsigned char>::test_teambroadcast(16, 1);

    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                      long>::test_teambroadcast(1000, 1);
    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                      long>::test_teambroadcast(1000, 1);
  }
}

TEST(TEST_CATEGORY, team_broadcast_float) {
  {
    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                      float>::test_teambroadcast(0, 1.3);
    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                      float>::test_teambroadcast(0, 1.3);

    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                      float>::test_teambroadcast(2, 1.3);
    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                      float>::test_teambroadcast(2, 1.3);

    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                      float>::test_teambroadcast(16, 1.3);
    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                      float>::test_teambroadcast(16, 1.3);

    // FIXME_CUDA
#ifdef FLARE_ON_CUDA_DEVICE
    if (!std::is_same<TEST_EXECSPACE, flare::Cuda>::value)
#endif
      {
        TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                          float>::test_teambroadcast(1000, 1.3);
        TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                          float>::test_teambroadcast(1000, 1.3);
      }
  }
}

TEST(TEST_CATEGORY, team_broadcast_double) {
  {
    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                      double>::test_teambroadcast(0, 1.3);
    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                      double>::test_teambroadcast(0, 1.3);

    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                      double>::test_teambroadcast(2, 1.3);
    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                      double>::test_teambroadcast(2, 1.3);

    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                      double>::test_teambroadcast(16, 1.3);
    TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,
                      double>::test_teambroadcast(16, 1.3);

    // FIXME_CUDA
#ifdef FLARE_ON_CUDA_DEVICE
    if (!std::is_same<TEST_EXECSPACE, flare::Cuda>::value)
#endif

      {
        TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Static>,
                          double>::test_teambroadcast(1000, 1.3);
        TestTeamBroadcast<TEST_EXECSPACE, flare::Schedule<flare::Dynamic>,

                          double>::test_teambroadcast(1000, 1.3);
      }
  }
}

TEST(TEST_CATEGORY, team_handle_by_value) {
  { TestTeamPolicyHandleByValue<TEST_EXECSPACE>(); }
}

}  // namespace Test

#include <team_vector_test.h>
#endif
