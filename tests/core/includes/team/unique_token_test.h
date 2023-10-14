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

template <class Space, flare::experimental::UniqueTokenScope Scope>
class TestUniqueToken {
 public:
  using execution_space = typename Space::execution_space;
  using tensor_type       = flare::Tensor<int*, execution_space>;

  flare::experimental::UniqueToken<execution_space, Scope> tokens;

  tensor_type verify;
  tensor_type counts;
  tensor_type errors;

  struct count_test_start_tag {};
  struct count_test_check_tag {};

  FLARE_INLINE_FUNCTION
  void operator()(long) const {
    flare::experimental::AcquireUniqueToken<execution_space, Scope> token_val(
        tokens);
    const int32_t t = token_val.value();

    bool ok = true;

    ok = ok && 0 <= t;
    ok = ok && t < tokens.size();
    ok = ok && 0 == flare::atomic_fetch_add(&verify(t), 1);

    flare::atomic_fetch_add(&counts(t), 1);

    ok = ok && 1 == flare::atomic_fetch_add(&verify(t), -1);

    if (!ok) {
      flare::atomic_fetch_add(&errors(0), 1);
    }
  }

  FLARE_INLINE_FUNCTION
  void operator()(count_test_start_tag, long) const {
    constexpr int R = 10;
    int id          = tokens.acquire();
    for (int j = 0; j < R; j++) counts(id)++;
    tokens.release(id);
  }

  FLARE_INLINE_FUNCTION
  void operator()(count_test_check_tag, long i, int64_t& lsum) const {
    lsum += counts(i);
  }

  TestUniqueToken()
      : tokens(execution_space()),
        verify("TestUniqueTokenVerify", tokens.size()),
        counts("TestUniqueTokenCounts", tokens.size()),
        errors("TestUniqueTokenErrors", 1) {}

  static void run() {
    using policy = flare::RangePolicy<execution_space>;

    TestUniqueToken self;

    {
      const int duplicate = 100;
      const long n        = duplicate * self.tokens.size();

      flare::parallel_for(policy(0, n), self);
      flare::parallel_for(policy(0, n), self);
      flare::parallel_for(policy(0, n), self);
      flare::fence();
    }

    typename tensor_type::HostMirror host_counts =
        flare::create_mirror_tensor(self.counts);

    flare::deep_copy(host_counts, self.counts);

    int32_t max = 0;

    {
      const long n = host_counts.extent(0);
      for (long i = 0; i < n; ++i) {
        if (max < host_counts[i]) max = host_counts[i];
      }
    }

    // Count test for pull request #3260
    {
      constexpr int N = 1000000;
      constexpr int R = 10;
      int num         = self.tokens.size();
      flare::resize(self.counts, num);
      flare::deep_copy(self.counts, 0);
      flare::parallel_for(
          "Start", flare::RangePolicy<Space, count_test_start_tag>(0, N),
          self);
      int64_t sum = 0;
      flare::parallel_reduce(
          "Check", flare::RangePolicy<Space, count_test_check_tag>(0, num),
          self, sum);
      ASSERT_EQ(sum, int64_t(N) * R);
    }

    typename tensor_type::HostMirror host_errors =
        flare::create_mirror_tensor(self.errors);

    flare::deep_copy(host_errors, self.errors);

    ASSERT_EQ(host_errors(0), 0) << "max reuse was " << max;
  }
};

TEST(TEST_CATEGORY, unique_token_global) {
  TestUniqueToken<TEST_EXECSPACE,
                  flare::experimental::UniqueTokenScope::Global>::run();
}

TEST(TEST_CATEGORY, unique_token_instance) {
  TestUniqueToken<TEST_EXECSPACE,
                  flare::experimental::UniqueTokenScope::Instance>::run();
}

template <class Space>
class TestAcquireTeamUniqueToken {
 public:
  using execution_space = typename Space::execution_space;
  using tensor_type       = flare::Tensor<int*, execution_space>;
  using scratch_tensor =
      flare::Tensor<int, typename execution_space::scratch_memory_space,
                   flare::MemoryUnmanaged>;
  using team_policy_type = flare::TeamPolicy<execution_space>;
  using team_member_type = typename team_policy_type::member_type;
  using tokens_type      = flare::experimental::UniqueToken<execution_space>;

  tokens_type tokens;

  tensor_type verify;
  tensor_type counts;
  tensor_type errors;

  FLARE_INLINE_FUNCTION
  void operator()(team_member_type team) const {
    flare::experimental::AcquireTeamUniqueToken<team_policy_type> token_val(
        tokens, team);
    scratch_tensor team_rank_0_token_val(team.team_scratch(0));
    const int32_t t = token_val.value();

    bool ok = true;

    ok = ok && 0 <= t;
    ok = ok && t < tokens.size();

    flare::single(flare::PerTeam(team), [&]() {
      ok = ok && 0 == flare::atomic_fetch_add(&verify(t), 1);

      flare::atomic_fetch_add(&counts(t), 1);

      ok = ok && 1 == flare::atomic_fetch_add(&verify(t), -1);
    });

    if (team.team_rank() == 0) {
      team_rank_0_token_val() = t;
    }
    team.team_barrier();
    ok = ok && team_rank_0_token_val() == t;

    if (!ok) {
      flare::atomic_fetch_add(&errors(0), 1);
    }
  }

  TestAcquireTeamUniqueToken(int team_size)
      : tokens(execution_space().concurrency() / team_size, execution_space()),
        verify("TestAcquireTeamUniqueTokenVerify", tokens.size()),
        counts("TestAcquireTeamUniqueTokenCounts", tokens.size()),
        errors("TestAcquireTeamUniqueTokenErrors", 1) {}

  static void run() {
    const int max_team_size = team_policy_type(1, 1).team_size_max(
        TestAcquireTeamUniqueToken(1), flare::ParallelForTag());
    const int team_size = std::min(2, max_team_size);
    TestAcquireTeamUniqueToken self(team_size);

    {
      const int duplicate = 100;
      const long n = duplicate * self.tokens.size();

      team_policy_type team_policy(n, team_size);
      team_policy.set_scratch_size(
          0, flare::PerTeam(flare::experimental::AcquireTeamUniqueToken<
                                 team_policy_type>::shmem_size() +
                             scratch_tensor::shmem_size()));

      flare::parallel_for(team_policy, self);
      flare::fence();
    }

    typename tensor_type::HostMirror host_counts =
        flare::create_mirror_tensor(self.counts);

    flare::deep_copy(host_counts, self.counts);

    int32_t max = 0;

    {
      const long n = host_counts.extent(0);
      for (long i = 0; i < n; ++i) {
        if (max < host_counts[i]) max = host_counts[i];
      }
    }

    typename tensor_type::HostMirror host_errors =
        flare::create_mirror_tensor(self.errors);

    flare::deep_copy(host_errors, self.errors);

    ASSERT_EQ(host_errors(0), 0) << "max reuse was " << max;
  }
};

TEST(TEST_CATEGORY, unique_token_team_acquire) {
    TestAcquireTeamUniqueToken<TEST_EXECSPACE>::run();
}

}  // namespace
