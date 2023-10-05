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

namespace TestConcept {

using ExecutionSpace = TEST_EXECSPACE;
using MemorySpace    = typename ExecutionSpace::memory_space;
using DeviceType     = typename ExecutionSpace::device_type;

static_assert(flare::is_execution_space<ExecutionSpace>{}, "");
static_assert(flare::is_execution_space<ExecutionSpace const>{}, "");
static_assert(!flare::is_execution_space<ExecutionSpace &>{}, "");
static_assert(!flare::is_execution_space<ExecutionSpace const &>{}, "");

static_assert(flare::is_memory_space<MemorySpace>{}, "");
static_assert(flare::is_memory_space<MemorySpace const>{}, "");
static_assert(!flare::is_memory_space<MemorySpace &>{}, "");
static_assert(!flare::is_memory_space<MemorySpace const &>{}, "");

static_assert(flare::is_device<DeviceType>{}, "");
static_assert(flare::is_device<DeviceType const>{}, "");
static_assert(!flare::is_device<DeviceType &>{}, "");
static_assert(!flare::is_device<DeviceType const &>{}, "");

static_assert(!flare::is_device<ExecutionSpace>{}, "");
static_assert(!flare::is_device<MemorySpace>{}, "");

static_assert(flare::is_space<ExecutionSpace>{}, "");
static_assert(flare::is_space<MemorySpace>{}, "");
static_assert(flare::is_space<DeviceType>{}, "");
static_assert(flare::is_space<ExecutionSpace const>{}, "");
static_assert(flare::is_space<MemorySpace const>{}, "");
static_assert(flare::is_space<DeviceType const>{}, "");
static_assert(!flare::is_space<ExecutionSpace &>{}, "");
static_assert(!flare::is_space<MemorySpace &>{}, "");
static_assert(!flare::is_space<DeviceType &>{}, "");

static_assert(flare::is_execution_space_v<ExecutionSpace>, "");
static_assert(!flare::is_execution_space_v<ExecutionSpace &>, "");

static_assert(
    std::is_same<float, flare::detail::remove_cvref_t<float const &>>{}, "");
static_assert(std::is_same<int, flare::detail::remove_cvref_t<int &>>{}, "");
static_assert(std::is_same<int, flare::detail::remove_cvref_t<int const>>{}, "");
static_assert(std::is_same<float, flare::detail::remove_cvref_t<float>>{}, "");


template <typename T>
struct is_team_handle_complete_trait_check {
 private:
  struct TrivialFunctor {
    void operator()(double &) {}
  };
  using test_value_type = double;
  test_value_type lvalueForMethodsNeedingIt_;
  test_value_type *ptrForMethodsNeedingIt_;
  // we use Sum here but any other reducer can be used
  // since we just want something that meets the ReducerConcept
  using reduction_to_test_t = ::flare::Sum<test_value_type>;

  // nested aliases
  template <class U>
  using ExecutionSpaceArchetypeAlias = typename U::execution_space;
  template <class U>
  using ScratchMemorySpaceArchetypeAlias = typename U::scratch_memory_space;

  // "indices" methods
  template <class U>
  using TeamRankArchetypeExpr = decltype(std::declval<U const &>().team_rank());

  template <class U>
  using TeamSizeArchetypeExpr = decltype(std::declval<U const &>().team_size());

  template <class U>
  using LeagueRankArchetypeExpr =
      decltype(std::declval<U const &>().league_rank());

  template <class U>
  using LeagueSizeArchetypeExpr =
      decltype(std::declval<U const &>().league_size());

  // scratch space
  template <class U>
  using TeamShmemArchetypeExpr =
      decltype(std::declval<U const &>().team_shmem());

  template <class U>
  using TeamScratchArchetypeExpr =
      decltype(std::declval<U const &>().team_scratch(0));

  template <class U>
  using ThreadScracthArchetypeExpr =
      decltype(std::declval<U const &>().thread_scratch(0));

  // collectives
  template <class U>
  using TeamBarrierArchetypeExpr =
      decltype(std::declval<U const &>().team_barrier());

  template <class U>
  using TeamBroadcastArchetypeExpr = decltype(
      std::declval<U const &>().team_broadcast(lvalueForMethodsNeedingIt_, 0));

  template <class U>
  using TeamBroadcastAcceptClosureArchetypeExpr =
      decltype(std::declval<U const &>().team_broadcast(
          TrivialFunctor{}, lvalueForMethodsNeedingIt_, 0));

  template <class U>
  using TeamReducedArchetypeExpr =
      decltype(std::declval<U const &>().team_reduce(
          std::declval<reduction_to_test_t>()));

  template <class U>
  using TeamScanArchetypeExpr = decltype(std::declval<U const &>().team_scan(
      lvalueForMethodsNeedingIt_, ptrForMethodsNeedingIt_));

 public:
  static constexpr bool value =
      flare::is_detected_v<ExecutionSpaceArchetypeAlias, T> &&
      flare::is_detected_v<ScratchMemorySpaceArchetypeAlias, T> &&
      //
      flare::is_detected_exact_v<int, TeamRankArchetypeExpr, T> &&
      flare::is_detected_exact_v<int, TeamSizeArchetypeExpr, T> &&
      flare::is_detected_exact_v<int, LeagueRankArchetypeExpr, T> &&
      flare::is_detected_exact_v<int, LeagueSizeArchetypeExpr, T> &&
      //
      flare::is_detected_exact_v<
          flare::detected_t<ScratchMemorySpaceArchetypeAlias, T> const &,
          TeamShmemArchetypeExpr, T> &&
      flare::is_detected_exact_v<
          flare::detected_t<ScratchMemorySpaceArchetypeAlias, T> const &,
          TeamScratchArchetypeExpr, T> &&
      flare::is_detected_exact_v<
          flare::detected_t<ScratchMemorySpaceArchetypeAlias, T> const &,
          ThreadScracthArchetypeExpr, T> &&
      //
      flare::is_detected_exact_v<void, TeamBarrierArchetypeExpr, T> &&
      flare::is_detected_exact_v<void, TeamBroadcastArchetypeExpr, T> &&
      flare::is_detected_exact_v<void, TeamBroadcastAcceptClosureArchetypeExpr,
                                  T> &&
      flare::is_detected_exact_v<void, TeamReducedArchetypeExpr, T> &&
      flare::is_detected_exact_v<test_value_type, TeamScanArchetypeExpr, T>;
  constexpr operator bool() const noexcept { return value; }
};

template <class T>
inline constexpr bool is_team_handle_complete_trait_check_v =
    is_team_handle_complete_trait_check<T>::value;

// actual test begins here


using space_t  = TEST_EXECSPACE;
using policy_t = flare::TeamPolicy<space_t>;
using member_t = typename policy_t::member_type;

// is_team_handle uses the actual core implementation
static_assert(flare::is_team_handle<member_t>::value);
static_assert(flare::is_team_handle_v<member_t>);
static_assert(flare::is_team_handle_v<member_t const>);
static_assert(!flare::is_team_handle_v<member_t &>);
static_assert(!flare::is_team_handle_v<member_t const &>);
static_assert(!flare::is_team_handle_v<member_t *>);
static_assert(!flare::is_team_handle_v<member_t const *>);
static_assert(!flare::is_team_handle_v<member_t *const>);

// is_team_handle_complete_trait_check uses the FULL trait class above
static_assert(is_team_handle_complete_trait_check<member_t>::value);
static_assert(is_team_handle_complete_trait_check_v<member_t>);
static_assert(is_team_handle_complete_trait_check_v<member_t const>);
static_assert(!is_team_handle_complete_trait_check_v<member_t &>);
static_assert(!is_team_handle_complete_trait_check_v<member_t const &>);
static_assert(!is_team_handle_complete_trait_check_v<member_t *>);
static_assert(!is_team_handle_complete_trait_check_v<member_t const *>);
static_assert(!is_team_handle_complete_trait_check_v<member_t *const>);

}  // namespace TestConcept
