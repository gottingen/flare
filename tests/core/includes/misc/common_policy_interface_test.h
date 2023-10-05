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

namespace {

template <typename Policy, typename ExpectedExecutionSpace,
          typename ExpectedIndexType, typename ExpectedScheduleType,
          typename ExpectedWorkTag>
constexpr bool compile_time_test() {
  using execution_space = typename Policy::execution_space;
  using index_type      = typename Policy::index_type;
  using schedule_type   = typename Policy::schedule_type;
  using work_tag        = typename Policy::work_tag;

  static_assert(std::is_same_v<execution_space, ExpectedExecutionSpace>);
  static_assert(std::is_same_v<index_type, ExpectedIndexType>);
  static_assert(std::is_same_v<schedule_type, ExpectedScheduleType>);
  static_assert(std::is_same_v<work_tag, ExpectedWorkTag>);

  return true;
}

// Separate class type from class template args so that different
// combinations of template args can be used, while still including
// any necessary templates args (stored in "Args...").
// Example: MDRangePolicy required an iteration pattern be included.
template <template <class...> class PolicyType, class... Args>
constexpr bool test_compile_time_parameters() {
  struct SomeTag {};

  using TestExecSpace    = TEST_EXECSPACE;
  using DefaultExecSpace = flare::DefaultExecutionSpace;
  using TestIndex        = TestExecSpace::size_type;
  using DefaultIndex     = DefaultExecSpace::size_type;
  using LongIndex        = flare::IndexType<long>;
  using StaticSchedule   = flare::Schedule<flare::Static>;
  using DynamicSchedule  = flare::Schedule<flare::Dynamic>;

  // clang-format off
  compile_time_test<PolicyType<                                                            Args...>, DefaultExecSpace, DefaultIndex, StaticSchedule,  void   >();
  compile_time_test<PolicyType<TestExecSpace,                                              Args...>, TestExecSpace,    TestIndex,    StaticSchedule,  void   >();
  compile_time_test<PolicyType<DynamicSchedule,                                            Args...>, DefaultExecSpace, DefaultIndex, DynamicSchedule, void   >();
  compile_time_test<PolicyType<TestExecSpace,   DynamicSchedule,                           Args...>, TestExecSpace,    TestIndex,    DynamicSchedule, void   >();
  compile_time_test<PolicyType<DynamicSchedule, LongIndex,                                 Args...>, DefaultExecSpace, long,         DynamicSchedule, void   >();
  compile_time_test<PolicyType<LongIndex,       DynamicSchedule,                           Args...>, DefaultExecSpace, long,         DynamicSchedule, void   >();
  compile_time_test<PolicyType<TestExecSpace,   DynamicSchedule, LongIndex,                Args...>, TestExecSpace,    long,         DynamicSchedule, void   >();
  compile_time_test<PolicyType<LongIndex,       TestExecSpace,   DynamicSchedule,          Args...>, TestExecSpace,    long,         DynamicSchedule, void   >();
  compile_time_test<PolicyType<DynamicSchedule, LongIndex,       SomeTag,                  Args...>, DefaultExecSpace, long,         DynamicSchedule, SomeTag>();
  compile_time_test<PolicyType<SomeTag,         DynamicSchedule, LongIndex,                Args...>, DefaultExecSpace, long,         DynamicSchedule, SomeTag>();
  compile_time_test<PolicyType<TestExecSpace,   DynamicSchedule, LongIndex, SomeTag,       Args...>, TestExecSpace,    long,         DynamicSchedule, SomeTag>();
  compile_time_test<PolicyType<DynamicSchedule, TestExecSpace,   LongIndex, SomeTag,       Args...>, TestExecSpace,    long,         DynamicSchedule, SomeTag>();
  compile_time_test<PolicyType<SomeTag,         DynamicSchedule, LongIndex, TestExecSpace, Args...>, TestExecSpace,    long,         DynamicSchedule, SomeTag>();
  // clang-format on

  return true;
}

static_assert(test_compile_time_parameters<flare::RangePolicy>());
static_assert(test_compile_time_parameters<flare::TeamPolicy>());
static_assert(
    test_compile_time_parameters<flare::MDRangePolicy, flare::Rank<2>>());

// Asserts that worktag conversion works properly.
template <class Policy>
constexpr bool test_worktag() {
  struct WorkTag1 {};
  struct WorkTag2 {};

  // Apply WorkTag1
  using PolicyWithWorkTag1 =
      flare::detail::WorkTagTrait::policy_with_trait<Policy, WorkTag1>;
  // Swap for WorkTag2
  using PolicyWithWorkTag2 =
      flare::detail::WorkTagTrait::policy_with_trait<PolicyWithWorkTag1,
                                                    WorkTag2>;

  static_assert(std::is_void_v<typename Policy::work_tag>);
  static_assert(
      std::is_same_v<typename PolicyWithWorkTag1::work_tag, WorkTag1>);
  static_assert(
      std::is_same_v<typename PolicyWithWorkTag2::work_tag, WorkTag2>);

  // Currently not possible to remove the work tag from a policy.
  // Uncomment the line below to see the compile error.
  // using PolicyRemoveWorkTag =
  // flare::detail::WorkTagTrait::policy_with_trait<PolicyWithWorkTag2, void>;
  // static_assert(std::is_void_v<PolicyRemoveWorkTag::work_tag>);

  return true;
}

static_assert(test_worktag<flare::RangePolicy<>>());
static_assert(test_worktag<flare::TeamPolicy<>>());
static_assert(test_worktag<flare::MDRangePolicy<flare::Rank<2>>>());

}  // namespace
