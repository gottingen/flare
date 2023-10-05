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

#include <range/mdrange_test.h>
#include <doctest.h>

namespace Test {

TEST_CASE("TEST_CATEGORY, mdrange_3d") {
  TestMDRange_3D<TEST_EXECSPACE>::test_for3(1, 10, 100);
  TestMDRange_3D<TEST_EXECSPACE>::test_for3(100, 10, 100);
  TestMDRange_3D<TEST_EXECSPACE>::test_reduce3(1, 10, 100);
  TestMDRange_3D<TEST_EXECSPACE>::test_reduce3(100, 10, 100);
}

TEST_CASE("TEST_CATEGORY, mdrange_neg_idx") {
  TestMDRange_2D_NegIdx<TEST_EXECSPACE>::test_2D_negidx(128, 32);
  TestMDRange_3D_NegIdx<TEST_EXECSPACE>::test_3D_negidx(128, 32, 8);
  TestMDRange_4D_NegIdx<TEST_EXECSPACE>::test_4D_negidx(128, 32, 8, 8);
  TestMDRange_5D_NegIdx<TEST_EXECSPACE>::test_5D_negidx(128, 32, 8, 8, 4);
  TestMDRange_6D_NegIdx<TEST_EXECSPACE>::test_6D_negidx(128, 32, 8, 8, 4, 2);
}

}  // namespace Test
