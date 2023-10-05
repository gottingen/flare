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
#ifndef TESTVIEWHOOKS_HPP_
#define TESTVIEWHOOKS_HPP_

#include <doctest.h>
#include <flare/core.h>

namespace Test {
template <class DeviceType>
struct TestViewHooks {
  struct TestSubscriber;

  static_assert(
      flare::experimental::is_hooks_policy<
          flare::experimental::SubscribableViewHooks<TestSubscriber> >::value,
      "Must be a hooks policy");

  using test_view_type =
      flare::View<double **,
                   flare::experimental::SubscribableViewHooks<TestSubscriber>,
                   DeviceType>;

  struct TestSubscriber {
    static test_view_type *self_ptr;
    static const test_view_type *other_ptr;

    template <typename View>
    static void copy_constructed(View &self, const View &other) {
      self_ptr  = &self;
      other_ptr = &other;
    }

    template <typename View>
    static void move_constructed(View &self, const View &other) {
      self_ptr  = &self;
      other_ptr = &other;
    }

    template <typename View>
    static void copy_assigned(View &self, const View &other) {
      self_ptr  = &self;
      other_ptr = &other;
    }

    template <typename View>
    static void move_assigned(View &self, const View &other) {
      self_ptr  = &self;
      other_ptr = &other;
    }

    static void reset() {
      self_ptr  = nullptr;
      other_ptr = nullptr;
    }
  };

  static void testViewHooksCopyConstruct() {
    TestSubscriber::reset();
    test_view_type testa;

    test_view_type testb(testa);
    REQUIRE_EQ(TestSubscriber::self_ptr, &testb);
    REQUIRE_EQ(TestSubscriber::other_ptr, &testa);
  }

  static void testViewHooksMoveConstruct() {
    TestSubscriber::reset();
    test_view_type testa;

    test_view_type testb(std::move(testa));
    REQUIRE_EQ(TestSubscriber::self_ptr, &testb);

    // This is valid, even if the view is moved-from
    REQUIRE_EQ(TestSubscriber::other_ptr, &testa);
  }

  static void testViewHooksCopyAssign() {
    TestSubscriber::reset();
    test_view_type testa;

    test_view_type testb;
    testb = testa;
    REQUIRE_EQ(TestSubscriber::self_ptr, &testb);
    REQUIRE_EQ(TestSubscriber::other_ptr, &testa);
  }

  static void testViewHooksMoveAssign() {
    TestSubscriber::reset();
    test_view_type testa;

    test_view_type testb;
    testb = std::move(testa);
    REQUIRE_EQ(TestSubscriber::self_ptr, &testb);

    // This is valid, even if the view is moved-from
    REQUIRE_EQ(TestSubscriber::other_ptr, &testa);
  }
};

template <class DeviceType>
typename TestViewHooks<DeviceType>::test_view_type
    *TestViewHooks<DeviceType>::TestSubscriber::self_ptr = nullptr;

template <class DeviceType>
const typename TestViewHooks<DeviceType>::test_view_type
    *TestViewHooks<DeviceType>::TestSubscriber::other_ptr = nullptr;

TEST_CASE("TEST_CATEGORY, view_hooks") {
  using ExecSpace = TEST_EXECSPACE;
  TestViewHooks<ExecSpace>::testViewHooksCopyConstruct();
  TestViewHooks<ExecSpace>::testViewHooksMoveConstruct();
  TestViewHooks<ExecSpace>::testViewHooksCopyAssign();
  TestViewHooks<ExecSpace>::testViewHooksMoveAssign();
}

}  // namespace Test
#endif  // TESTVIEWHOOKS_HPP_
