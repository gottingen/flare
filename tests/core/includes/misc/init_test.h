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
#include <doctest.h>
#include <flare/core.h>

namespace Test {
TEST_CASE("TEST_CATEGORY, init") { ; }

#ifdef FLARE_ENABLE_CXX11_DISPATCH_LAMBDA

template <class ExecSpace>
void test_dispatch() {
  const int repeat = 100;
  for (int i = 0; i < repeat; ++i) {
    for (int j = 0; j < repeat; ++j) {
      flare::parallel_for(flare::RangePolicy<TEST_EXECSPACE>(0, j),
                           FLARE_LAMBDA(int){});
    }
  }
}

TEST_CASE("TEST_CATEGORY, dispatch") { test_dispatch<TEST_EXECSPACE>(); }
#endif

}  // namespace Test
