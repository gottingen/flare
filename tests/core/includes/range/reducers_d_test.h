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
#include <range/reducers_test.h>
#include <range/non_trivial_scalar_types_test.h>

namespace Test {
TEST_CASE("TEST_CATEGORY, reducers_complex_double") {
  TestReducers<flare::complex<double>, TEST_EXECSPACE>::execute_basic();
}

TEST_CASE("TEST_CATEGORY, reducers_struct") {
  TestReducers<array_reduce<float, 1>, TEST_EXECSPACE>::test_sum(1031);
  TestReducers<array_reduce<float, 2>, TEST_EXECSPACE>::test_sum(1031);
  TestReducers<array_reduce<float, 4>, TEST_EXECSPACE>::test_sum(1031);
  TestReducers<array_reduce<float, 3>, TEST_EXECSPACE>::test_sum(1031);
  TestReducers<array_reduce<float, 7>, TEST_EXECSPACE>::test_sum(1031);
}

TEST_CASE("TEST_CATEGORY, reducers_half_t") {
  using ThisTestType = flare::experimental::half_t;
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_sum(2);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_sum(101);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_sum(202);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_sum(303);

  TestReducers<ThisTestType, TEST_EXECSPACE>::test_prod(5);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_prod(10);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_prod(15);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_prod(20);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_prod(25);
}

TEST_CASE("TEST_CATEGORY, reducers_bhalf_t") {
  using ThisTestType = flare::experimental::bhalf_t;

  TestReducers<ThisTestType, TEST_EXECSPACE>::test_sum(2);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_sum(25);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_sum(50);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_sum(51);

  TestReducers<ThisTestType, TEST_EXECSPACE>::test_prod(1);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_prod(2);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_prod(3);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_prod(4);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_prod(5);
}

TEST_CASE("TEST_CATEGORY, reducers_int8_t") {
  using ThisTestType = int8_t;

  TestReducers<ThisTestType, TEST_EXECSPACE>::test_sum(1);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_sum(2);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_sum(3);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_sum(4);

  TestReducers<ThisTestType, TEST_EXECSPACE>::test_prod(1);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_prod(2);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_prod(3);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_prod(4);
}

TEST_CASE("TEST_CATEGORY, reducers_point_t") {
  using ThisTestType = point_t;

  TestReducers<ThisTestType, TEST_EXECSPACE>::test_sum(1);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_sum(2);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_sum(3);
  TestReducers<ThisTestType, TEST_EXECSPACE>::test_sum(4);
}

}  // namespace Test
