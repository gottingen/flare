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
#include <flare/core/common/checked_integer_ops.h>
#include <limits>

namespace {

TEST_CASE("TEST_CATEGORY, checked_integer_operations_multiply_overflow") {
  {
    auto result      = 1u;
    auto is_overflow = flare::detail::multiply_overflow(1u, 2u, result);
      REQUIRE_EQ(result, 2u);
      REQUIRE_FALSE(is_overflow);
  }
  {
    auto result      = 1u;
    auto is_overflow = flare::detail::multiply_overflow(
        std::numeric_limits<unsigned>::max(), 2u, result);
      REQUIRE(is_overflow);
  }
}
/*
TEST_CASE("TEST_CATEGORY_DEATH, checked_integer_operations_multiply_overflow_abort") {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  {
    auto result = flare::detail::multiply_overflow_abort(1u, 2u);
    EXPECT_EQ(result, 2u);
  }
  {
    ASSERT_DEATH(flare::detail::multiply_overflow_abort(
                     std::numeric_limits<unsigned>::max(), 2u),
                 "Arithmetic overflow detected.");
  }
}
*/
}  // namespace
