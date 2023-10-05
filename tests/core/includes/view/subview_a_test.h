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

#ifndef FLARE_TEST_SUBVIEW_A_HPP
#define FLARE_TEST_SUBVIEW_A_HPP
#include <view_subview_test.h>

namespace Test {

TEST(TEST_CATEGORY, view_subview_auto_1d_left) {
  TestViewSubview::test_auto_1d<flare::LayoutLeft, TEST_EXECSPACE>();
}

TEST(TEST_CATEGORY, view_subview_auto_1d_right) {
  TestViewSubview::test_auto_1d<flare::LayoutRight, TEST_EXECSPACE>();
}

TEST(TEST_CATEGORY, view_subview_auto_1d_stride) {
  TestViewSubview::test_auto_1d<flare::LayoutStride, TEST_EXECSPACE>();
}

TEST(TEST_CATEGORY, view_subview_assign_strided) {
  TestViewSubview::test_1d_strided_assignment<TEST_EXECSPACE>();
}

TEST(TEST_CATEGORY, view_subview_left_0) {
  TestViewSubview::test_left_0<TEST_EXECSPACE>();
}

TEST(TEST_CATEGORY, view_subview_left_1) {
  TestViewSubview::test_left_1<TEST_EXECSPACE>();
}

TEST(TEST_CATEGORY, view_subview_left_2) {
  TestViewSubview::test_left_2<TEST_EXECSPACE>();
}

TEST(TEST_CATEGORY, view_subview_left_3) {
  TestViewSubview::test_left_3<TEST_EXECSPACE>();
}

TEST(TEST_CATEGORY, view_subview_right_0) {
  TestViewSubview::test_right_0<TEST_EXECSPACE>();
}

TEST(TEST_CATEGORY, view_subview_right_1) {
  TestViewSubview::test_right_1<TEST_EXECSPACE>();
}

TEST(TEST_CATEGORY, view_subview_right_3) {
  TestViewSubview::test_right_3<TEST_EXECSPACE>();
}

TEST(TEST_CATEGORY, view_static_tests) {
  TestViewSubview::TestSubviewStaticSizes<TEST_EXECSPACE,
                                          flare::LayoutLeft>()();
  TestViewSubview::TestSubviewStaticSizes<TEST_EXECSPACE,
                                          flare::LayoutRight>()();
  TestViewSubview::TestExtentsStaticTests<TEST_EXECSPACE>();
}

}  // namespace Test
#endif
