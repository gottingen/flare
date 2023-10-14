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

#ifndef SUBTENSOR_A_TEST_H_
#define SUBTENSOR_A_TEST_H_
#include <tensor/tensor_subtensor_test.h>

namespace Test {

TEST_CASE("TEST_CATEGORY, tensor_subtensor_auto_1d_left") {
  TestTensorSubtensor::test_auto_1d<flare::LayoutLeft, TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, tensor_subtensor_auto_1d_right") {
  TestTensorSubtensor::test_auto_1d<flare::LayoutRight, TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, tensor_subtensor_auto_1d_stride") {
  TestTensorSubtensor::test_auto_1d<flare::LayoutStride, TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, tensor_subtensor_assign_strided") {
  TestTensorSubtensor::test_1d_strided_assignment<TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, tensor_subtensor_left_0") {
  TestTensorSubtensor::test_left_0<TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, tensor_subtensor_left_1") {
  TestTensorSubtensor::test_left_1<TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, tensor_subtensor_left_2") {
  TestTensorSubtensor::test_left_2<TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, tensor_subtensor_left_3") {
  TestTensorSubtensor::test_left_3<TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, tensor_subtensor_right_0") {
  TestTensorSubtensor::test_right_0<TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, tensor_subtensor_right_1") {
  TestTensorSubtensor::test_right_1<TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, tensor_subtensor_right_3") {
  TestTensorSubtensor::test_right_3<TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, view_static_tests") {
  TestTensorSubtensor::TestSubtensorStaticSizes<TEST_EXECSPACE,
                                          flare::LayoutLeft>()();
  TestTensorSubtensor::TestSubtensorStaticSizes<TEST_EXECSPACE,
                                          flare::LayoutRight>()();
  TestTensorSubtensor::TestExtentsStaticTests<TEST_EXECSPACE>();
}

}  // namespace Test
#endif  // SUBTENSOR_A_TEST_H_
