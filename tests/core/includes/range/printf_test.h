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

#include "flare/core.h"
/*
template <class ExecutionSpace>
void test_flare_printf() {
  std::string msg = "Print an integer: ";
  flare::parallel_for(
      flare::RangePolicy<ExecutionSpace>(0, 1),
      FLARE_LAMBDA(int) { flare::printf("Print an integer: %d", 2); });
  flare::fence();
  std::string expected_string("Print an integer: 2");
  REQUIRE_EQ(captured, expected_string);
}

TEST_CASE("TEST_CATEGORY, flare_printf") { test_flare_printf<TEST_EXECSPACE>(); }
*/