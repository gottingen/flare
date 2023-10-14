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

#include <tensor/tensor_api_test.h>

namespace Test {

TEST_CASE("TEST_CATEGORY, tensor_api_d") {
  TestTensorAPI<double, TEST_EXECSPACE>::run_test_const();
  TestTensorAPI<double, TEST_EXECSPACE>::run_test_subtensor();
  TestTensorAPI<double, TEST_EXECSPACE>::run_test_subtensor_strided();
  TestTensorAPI<double, TEST_EXECSPACE>::run_test_vector();
  TestTensorAPI<double, TEST_EXECSPACE>::run_test_tensor_operator_c();
}

TEST_CASE("TEST_CATEGORY, view_allocation_error") {
  TestTensorAPI<double, TEST_EXECSPACE>::run_test_error();
}

}  // namespace Test
