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
#include <flare/core/common/stacktrace.h>
#include <cstdio>
#include <cstdint>
#include <sstream>
#include <type_traits>

namespace Test {

template <class ExecutionSpace, class DataType>
struct TestTeamScan {
  using execution_space = ExecutionSpace;
  using value_type      = DataType;
  using policy_type     = flare::TeamPolicy<execution_space>;
  using member_type     = typename policy_type::member_type;
  using tensor_type       = flare::Tensor<value_type**, execution_space>;

  tensor_type a_d;
  tensor_type a_r;
  int32_t M = 0;
  int32_t N = 0;

  FLARE_FUNCTION
  void operator()(const member_type& team) const {
    auto leagueRank = team.league_rank();

    auto beg = 0;
    auto end = N;

    flare::parallel_for(
        flare::TeamThreadRange(team, beg, end),
        [&](const int i) { a_d(leagueRank, i) = leagueRank * N + i; });

    flare::parallel_scan(flare::TeamThreadRange(team, beg, end),
                          [&](int i, DataType& val, const bool final) {
                            val += a_d(leagueRank, i);
                            if (final) a_r(leagueRank, i) = val;
                          });
  }

  auto operator()(int32_t _M, int32_t _N) {
    std::stringstream ss;
    ss << flare::detail::demangle(typeid(*this).name());
    ss << "(/*M=*/" << _M << ", /*N=*/" << _N << ")";
    std::string const test_id = ss.str();

    M   = _M;
    N   = _N;
    a_d = tensor_type("a_d", M, N);
    a_r = tensor_type("a_r", M, N);

    // Set team size explicitly to check whether non-power-of-two team sizes can
    // be used.
    if (ExecutionSpace().concurrency() > 10000)
      flare::parallel_for(policy_type(M, 127), *this);
    else if (ExecutionSpace().concurrency() > 2)
      flare::parallel_for(policy_type(M, 3), *this);
    else
      flare::parallel_for(policy_type(M, 1), *this);

    auto a_i = flare::create_mirror_tensor(a_d);
    auto a_o = flare::create_mirror_tensor(a_r);
    flare::deep_copy(a_i, a_d);
    flare::deep_copy(a_o, a_r);

    for (int32_t i = 0; i < M; ++i) {
      value_type scan_ref = 0;
      value_type scan_calc;
      value_type abs_err = 0;
      // each fp addition is subject to small loses in precision and these
      // compound as loop so we set the base error to be the machine epsilon and
      // then add in another epsilon each iteration. For example, with CUDA
      // backend + 32-bit float + large N values (e.g. 1,000) + high
      // thread-counts (e.g. 1024), this test will fail w/o epsilon
      // accommodation
      constexpr value_type epsilon = std::numeric_limits<value_type>::epsilon();
      for (int32_t j = 0; j < N; ++j) {
        scan_ref += a_i(i, j);
        scan_calc = a_o(i, j);
        if (std::is_integral<value_type>::value) {
          ASSERT_EQ(scan_ref, scan_calc)
              << test_id
              << " calculated scan output value differs from reference at "
                 "indices i="
              << i << " and j=" << j;
        } else {
          abs_err += epsilon;
          ASSERT_NEAR(scan_ref, scan_calc, abs_err)
              << test_id
              << " calculated scan output value differs from reference at "
                 "indices i="
              << i << " and j=" << j;
        }
      }
    }
  }
};

TEST(TEST_CATEGORY, team_scan) {
  TestTeamScan<TEST_EXECSPACE, int32_t>{}(0, 0);
  TestTeamScan<TEST_EXECSPACE, int32_t>{}(0, 1);
  TestTeamScan<TEST_EXECSPACE, int32_t>{}(1, 0);
  TestTeamScan<TEST_EXECSPACE, uint32_t>{}(99, 32);
  TestTeamScan<TEST_EXECSPACE, uint32_t>{}(139, 64);
  TestTeamScan<TEST_EXECSPACE, uint32_t>{}(163, 128);
  TestTeamScan<TEST_EXECSPACE, int64_t>{}(433, 256);
  TestTeamScan<TEST_EXECSPACE, uint64_t>{}(976, 512);
  TestTeamScan<TEST_EXECSPACE, uint64_t>{}(1234, 1024);
  TestTeamScan<TEST_EXECSPACE, float>{}(2596, 34);
  TestTeamScan<TEST_EXECSPACE, double>{}(2596, 59);
  TestTeamScan<TEST_EXECSPACE, float>{}(2596, 65);
  TestTeamScan<TEST_EXECSPACE, double>{}(2596, 371);
  TestTeamScan<TEST_EXECSPACE, int64_t>{}(2596, 987);
  TestTeamScan<TEST_EXECSPACE, double>{}(2596, 1311);
}

// Temporary: This condition will progressively be reduced when parallel_scan
// with return value will be implemented for more backends.
#if defined(FLARE_ENABLE_SERIAL) || defined(FLARE_ENABLE_OPENMP)
#if !defined(FLARE_ON_CUDA_DEVICE) &&            \
    !defined(FLARE_ENABLE_THREADS)
template <class ExecutionSpace, class DataType>
struct TestTeamScanRetVal {
  using execution_space = ExecutionSpace;
  using value_type      = DataType;
  using policy_type     = flare::TeamPolicy<execution_space>;
  using member_type     = typename policy_type::member_type;
  using tensor_1d_type    = flare::Tensor<value_type*, execution_space>;
  using tensor_2d_type    = flare::Tensor<value_type**, execution_space>;

  tensor_2d_type a_d;
  tensor_2d_type a_r;
  tensor_1d_type a_s;
  int32_t M = 0;
  int32_t N = 0;

  FLARE_FUNCTION
  void operator()(const member_type& team) const {
    auto leagueRank = team.league_rank();

    auto beg = 0;
    auto end = N;

    flare::parallel_for(
        flare::TeamThreadRange(team, beg, end),
        [&](const int i) { a_d(leagueRank, i) = leagueRank * N + i; });

    DataType accum;
    flare::parallel_scan(
        flare::TeamThreadRange(team, beg, end),
        [&](int i, DataType& val, const bool final) {
          val += a_d(leagueRank, i);
          if (final) a_r(leagueRank, i) = val;
        },
        accum);

    // Save return value from parallel_scan
    flare::single(flare::PerTeam(team), [&]() { a_s(leagueRank) = accum; });
  }

  auto operator()(int32_t _M, int32_t _N) {
    std::stringstream ss;
    ss << flare::detail::demangle(typeid(*this).name());
    ss << "(/*M=*/" << _M << ", /*N=*/" << _N << ")";
    std::string const test_id = ss.str();

    M   = _M;
    N   = _N;
    a_d = tensor_2d_type("a_d", M, N);
    a_r = tensor_2d_type("a_r", M, N);
    a_s = tensor_1d_type("a_s", M);

    // Execute calculations
    flare::parallel_for(policy_type(M, flare::AUTO), *this);

    flare::fence();
    auto a_i  = flare::create_mirror_tensor(a_d);
    auto a_o  = flare::create_mirror_tensor(a_r);
    auto a_os = flare::create_mirror_tensor(a_s);
    flare::deep_copy(a_i, a_d);
    flare::deep_copy(a_o, a_r);
    flare::deep_copy(a_os, a_s);

    for (int32_t i = 0; i < M; ++i) {
      value_type scan_ref = 0;
      value_type scan_calc;
      value_type abs_err = 0;
      // each fp addition is subject to small loses in precision and these
      // compound as loop so we set the base error to be the machine epsilon and
      // then add in another epsilon each iteration. For example, with CUDA
      // backend + 32-bit float + large N values (e.g. 1,000) + high
      // thread-counts (e.g. 1024), this test will fail w/o epsilon
      // accommodation
      constexpr value_type epsilon = std::numeric_limits<value_type>::epsilon();
      for (int32_t j = 0; j < N; ++j) {
        scan_ref += a_i(i, j);
        scan_calc = a_o(i, j);
        if (std::is_integral<value_type>::value) {
          ASSERT_EQ(scan_ref, scan_calc)
              << test_id
              << " calculated scan output value differs from reference at "
                 "indices i="
              << i << " and j=" << j;
        } else {
          abs_err += epsilon;
          ASSERT_NEAR(scan_ref, scan_calc, abs_err)
              << test_id
              << " calculated scan output value differs from reference at "
                 "indices i="
              << i << " and j=" << j;
        }
      }
      // Validate return value from parallel_scan
      if (std::is_integral<value_type>::value) {
        ASSERT_EQ(scan_ref, a_os(i));
      } else {
        ASSERT_NEAR(scan_ref, a_os(i), abs_err);
      }
    }
  }
};

TEST(TEST_CATEGORY, team_scan_ret_val) {
  TestTeamScanRetVal<TEST_EXECSPACE, int32_t>{}(0, 0);
  TestTeamScanRetVal<TEST_EXECSPACE, int32_t>{}(0, 1);
  TestTeamScanRetVal<TEST_EXECSPACE, int32_t>{}(1, 0);
  TestTeamScanRetVal<TEST_EXECSPACE, uint32_t>{}(99, 32);
  TestTeamScanRetVal<TEST_EXECSPACE, uint32_t>{}(139, 64);
  TestTeamScanRetVal<TEST_EXECSPACE, uint32_t>{}(163, 128);
  TestTeamScanRetVal<TEST_EXECSPACE, int64_t>{}(433, 256);
  TestTeamScanRetVal<TEST_EXECSPACE, uint64_t>{}(976, 512);
  TestTeamScanRetVal<TEST_EXECSPACE, uint64_t>{}(1234, 1024);
  TestTeamScanRetVal<TEST_EXECSPACE, float>{}(2596, 34);
  TestTeamScanRetVal<TEST_EXECSPACE, double>{}(2596, 59);
  TestTeamScanRetVal<TEST_EXECSPACE, float>{}(2596, 65);
  TestTeamScanRetVal<TEST_EXECSPACE, double>{}(2596, 371);
  TestTeamScanRetVal<TEST_EXECSPACE, int64_t>{}(2596, 987);
  TestTeamScanRetVal<TEST_EXECSPACE, double>{}(2596, 1311);
}
#endif
#endif

}  // namespace Test
