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
#ifndef TENSOR_HOOKS_TEST_H_
#define TENSOR_HOOKS_TEST_H_

#include <doctest.h>
#include <flare/core.h>

namespace Test {
template <class DeviceType>
struct TestTensorHooks {
  struct TestSubscriber;

  static_assert(
      flare::experimental::is_hooks_policy<
          flare::experimental::SubscribableTensorHooks<TestSubscriber> >::value,
      "Must be a hooks policy");

  using test_tensor_type =
      flare::Tensor<double **,
                   flare::experimental::SubscribableTensorHooks<TestSubscriber>,
                   DeviceType>;

  struct TestSubscriber {
    static test_tensor_type *self_ptr;
    static const test_tensor_type *other_ptr;

    template <typename Tensor>
    static void copy_constructed(Tensor &self, const Tensor &other) {
      self_ptr  = &self;
      other_ptr = &other;
    }

    template <typename Tensor>
    static void move_constructed(Tensor &self, const Tensor &other) {
      self_ptr  = &self;
      other_ptr = &other;
    }

    template <typename Tensor>
    static void copy_assigned(Tensor &self, const Tensor &other) {
      self_ptr  = &self;
      other_ptr = &other;
    }

    template <typename Tensor>
    static void move_assigned(Tensor &self, const Tensor &other) {
      self_ptr  = &self;
      other_ptr = &other;
    }

    static void reset() {
      self_ptr  = nullptr;
      other_ptr = nullptr;
    }
  };

  static void testTensorHooksCopyConstruct() {
    TestSubscriber::reset();
    test_tensor_type testa;

    test_tensor_type testb(testa);
    REQUIRE_EQ(TestSubscriber::self_ptr, &testb);
    REQUIRE_EQ(TestSubscriber::other_ptr, &testa);
  }

  static void testTensorHooksMoveConstruct() {
    TestSubscriber::reset();
    test_tensor_type testa;

    test_tensor_type testb(std::move(testa));
    REQUIRE_EQ(TestSubscriber::self_ptr, &testb);

    // This is valid, even if the tensor is moved-from
    REQUIRE_EQ(TestSubscriber::other_ptr, &testa);
  }

  static void testTensorHooksCopyAssign() {
    TestSubscriber::reset();
    test_tensor_type testa;

    test_tensor_type testb;
    testb = testa;
    REQUIRE_EQ(TestSubscriber::self_ptr, &testb);
    REQUIRE_EQ(TestSubscriber::other_ptr, &testa);
  }

  static void testTensorHooksMoveAssign() {
    TestSubscriber::reset();
    test_tensor_type testa;

    test_tensor_type testb;
    testb = std::move(testa);
    REQUIRE_EQ(TestSubscriber::self_ptr, &testb);

    // This is valid, even if the tensor is moved-from
    REQUIRE_EQ(TestSubscriber::other_ptr, &testa);
  }
};

template <class DeviceType>
typename TestTensorHooks<DeviceType>::test_tensor_type
    *TestTensorHooks<DeviceType>::TestSubscriber::self_ptr = nullptr;

template <class DeviceType>
const typename TestTensorHooks<DeviceType>::test_tensor_type
    *TestTensorHooks<DeviceType>::TestSubscriber::other_ptr = nullptr;

TEST_CASE("TEST_CATEGORY, tensor_hooks") {
  using ExecSpace = TEST_EXECSPACE;
  TestTensorHooks<ExecSpace>::testTensorHooksCopyConstruct();
  TestTensorHooks<ExecSpace>::testTensorHooksMoveConstruct();
  TestTensorHooks<ExecSpace>::testTensorHooksCopyAssign();
  TestTensorHooks<ExecSpace>::testTensorHooksMoveAssign();
}

}  // namespace Test
#endif  // TENSOR_HOOKS_TEST_H_
