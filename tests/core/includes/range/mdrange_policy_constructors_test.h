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

#include <doctest.h>

#include <flare/core.h>

namespace {

template <class IndexType>
void construct_mdrange_policy_variable_type() {
  (void)flare::MDRangePolicy<TEST_EXECSPACE, flare::Rank<2>>{
      flare::Array<IndexType, 2>{}, flare::Array<IndexType, 2>{}};

  (void)flare::MDRangePolicy<TEST_EXECSPACE, flare::Rank<2>>{
      {{IndexType(0), IndexType(0)}}, {{IndexType(2), IndexType(2)}}};

  (void)flare::MDRangePolicy<TEST_EXECSPACE, flare::Rank<2>>{
      {IndexType(0), IndexType(0)}, {IndexType(2), IndexType(2)}};
}

TEST_CASE("TEST_CATEGORY, md_range_policy_construction_from_arrays") {
  {
    // Check that construction from flare::Array of the specified index type
    // works.
    using IndexType = unsigned long long;
    flare::MDRangePolicy<TEST_EXECSPACE, flare::Rank<2>,
                          flare::IndexType<IndexType>>
        p1(flare::Array<IndexType, 2>{{0, 1}},
           flare::Array<IndexType, 2>{{2, 3}});
    flare::MDRangePolicy<TEST_EXECSPACE, flare::Rank<2>,
                          flare::IndexType<IndexType>>
        p2(flare::Array<IndexType, 2>{{0, 1}},
           flare::Array<IndexType, 2>{{2, 3}});
    flare::MDRangePolicy<TEST_EXECSPACE, flare::Rank<2>,
                          flare::IndexType<IndexType>>
        p3(flare::Array<IndexType, 2>{{0, 1}},
           flare::Array<IndexType, 2>{{2, 3}},
           flare::Array<IndexType, 1>{{4}});
  }
  {
    // Check that construction from double-braced initializer list
    // works.
    using index_type = unsigned long long;
    flare::MDRangePolicy<TEST_EXECSPACE, flare::Rank<2>> p1({{0, 1}},
                                                              {{2, 3}});
    flare::MDRangePolicy<TEST_EXECSPACE, flare::Rank<2>,
                          flare::IndexType<index_type>>
        p2({{0, 1}}, {{2, 3}});
  }
  {

    flare::MDRangePolicy<TEST_EXECSPACE, flare::Rank<2>> p1(
        flare::Array<long, 2>{{0, 1}}, flare::Array<long, 2>{{2, 3}});
    flare::MDRangePolicy<TEST_EXECSPACE, flare::Rank<2>> p2(
        flare::Array<long, 2>{{0, 1}}, flare::Array<long, 2>{{2, 3}});
    flare::MDRangePolicy<TEST_EXECSPACE, flare::Rank<2>> p3(
        flare::Array<long, 2>{{0, 1}}, flare::Array<long, 2>{{2, 3}},
        flare::Array<long, 1>{{4}});
  }

  // Check that construction from various index types works.
  construct_mdrange_policy_variable_type<char>();
  construct_mdrange_policy_variable_type<int>();
  construct_mdrange_policy_variable_type<unsigned long>();
  construct_mdrange_policy_variable_type<std::int64_t>();
}
/*
TEST_CASE("TEST_CATEGORY_DEATH, policy_bounds_unsafe_narrowing_conversions") {
  using Policy = flare::MDRangePolicy<TEST_EXECSPACE, flare::Rank<2>,
                                       flare::IndexType<unsigned>>;

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_DEATH(
      {
        (void)Policy({-1, 0}, {2, 3});
      },
      "unsafe narrowing conversion");
}
*/
}  // namespace
