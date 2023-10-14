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

namespace Test {
namespace detail {
template <class TensorTypeDst, class TensorTypeSrc>
struct TestAssignability {
  using mapping_type =
      flare::detail::TensorMapping<typename TensorTypeDst::traits,
                                typename TensorTypeSrc::traits,
                                typename TensorTypeDst::specialize>;

  template <class MappingType>
  static void try_assign(
      TensorTypeDst& dst, TensorTypeSrc& src,
      std::enable_if_t<MappingType::is_assignable>* = nullptr) {
    dst = src;
  }

  template <class MappingType>
  static void try_assign(
      TensorTypeDst&, TensorTypeSrc&,
      std::enable_if_t<!MappingType::is_assignable>* = nullptr) {
    flare::detail::throw_runtime_exception(
        "TestAssignability::try_assign: Unexpected call path");
  }

  template <class... Dimensions>
  static void test(bool always, bool sometimes, Dimensions... dims) {
    TensorTypeDst dst;
    TensorTypeSrc src("SRC", dims...);

    bool is_always_assignable =
        flare::is_always_assignable<TensorTypeDst, TensorTypeSrc>::value;
    bool is_assignable = flare::is_assignable(dst, src);

    // Print out if there is an error with typeid so you can just filter the
    // output with c++filt -t to see which assignment causes the error.
    if (is_always_assignable != always || is_assignable != sometimes)
      printf(
          "is_always_assignable: %i (%i), is_assignable: %i (%i) [ %s ] to [ "
          "%s ]\n",
          is_always_assignable ? 1 : 0, always ? 1 : 0, is_assignable ? 1 : 0,
          sometimes ? 1 : 0, typeid(TensorTypeSrc).name(),
          typeid(TensorTypeDst).name());
    if (sometimes) {
      REQUIRE_NOTHROW(try_assign<mapping_type>(dst, src));
    }
    REQUIRE_EQ(always, is_always_assignable);
    REQUIRE_EQ(sometimes, is_assignable);
  }
};

}  // namespace detail

TEST_CASE("TEST_CATEGORY, tensor_is_assignable") {
  using namespace flare;
  using h_exec = typename DefaultHostExecutionSpace::memory_space;
  using d_exec = typename TEST_EXECSPACE::memory_space;
  using left   = LayoutLeft;
  using right  = LayoutRight;
  using stride = LayoutStride;
  // Static/Dynamic Extents
  detail::TestAssignability<Tensor<int*, left, d_exec>,
                          Tensor<int*, left, d_exec>>::test(true, true, 10);
  detail::TestAssignability<Tensor<int[10], left, d_exec>,
                          Tensor<int*, left, d_exec>>::test(false, true, 10);
  detail::TestAssignability<Tensor<int[5], left, d_exec>,
                          Tensor<int*, left, d_exec>>::test(false, false, 10);
  detail::TestAssignability<Tensor<int*, left, d_exec>,
                          Tensor<int[10], left, d_exec>>::test(true, true);
  detail::TestAssignability<Tensor<int[10], left, d_exec>,
                          Tensor<int[10], left, d_exec>>::test(true, true);
  detail::TestAssignability<Tensor<int[5], left, d_exec>,
                          Tensor<int[10], left, d_exec>>::test(false, false);
  detail::TestAssignability<Tensor<int**, left, d_exec>,
                          Tensor<int**, left, d_exec>>::test(true, true, 10, 10);
  detail::TestAssignability<Tensor<int * [10], left, d_exec>,
                          Tensor<int**, left, d_exec>>::test(false, true, 10, 10);
  detail::TestAssignability<Tensor<int * [5], left, d_exec>,
                          Tensor<int**, left, d_exec>>::test(false, false, 10,
                                                           10);
  detail::TestAssignability<Tensor<int**, left, d_exec>,
                          Tensor<int * [10], left, d_exec>>::test(true, true, 10);
  detail::TestAssignability<Tensor<int * [10], left, d_exec>,
                          Tensor<int * [10], left, d_exec>>::test(true, true, 10);
  detail::TestAssignability<Tensor<int * [5], left, d_exec>,
                          Tensor<int * [10], left, d_exec>>::test(false, false,
                                                                10);

  // Mismatch value_type
  detail::TestAssignability<Tensor<int*, left, d_exec>,
                          Tensor<double*, left, d_exec>>::test(false, false, 10);

  // Layout assignment
  detail::TestAssignability<Tensor<int, left, d_exec>,
                          Tensor<int, right, d_exec>>::test(true, true);
  detail::TestAssignability<Tensor<int*, left, d_exec>,
                          Tensor<int*, right, d_exec>>::test(true, true, 10);
  detail::TestAssignability<Tensor<int[5], left, d_exec>,
                          Tensor<int*, right, d_exec>>::test(false, false, 10);
  detail::TestAssignability<Tensor<int[10], left, d_exec>,
                          Tensor<int*, right, d_exec>>::test(false, true, 10);
  detail::TestAssignability<Tensor<int*, left, d_exec>,
                          Tensor<int[5], right, d_exec>>::test(true, true);
  detail::TestAssignability<Tensor<int[5], left, d_exec>,
                          Tensor<int[10], right, d_exec>>::test(false, false);

  // This could be made possible (due to the degenerate nature of the tensors) but
  // we do not allow this yet
  // TestAssignability<Tensor<int**,left,d_exec>,Tensor<int**,right,d_exec>>::test(false,true,10,1);
  detail::TestAssignability<Tensor<int**, left, d_exec>,
                          Tensor<int**, right, d_exec>>::test(false, false, 10,
                                                            2);
  detail::TestAssignability<Tensor<int**, stride, d_exec>,
                          Tensor<int**, right, d_exec>>::test(true, true, 10, 2);
  detail::TestAssignability<Tensor<int**, stride, d_exec>,
                          Tensor<int**, left, d_exec>>::test(true, true, 10, 2);

  // Space Assignment
  bool expected = flare::detail::MemorySpaceAccess<d_exec, h_exec>::assignable;
  detail::TestAssignability<Tensor<int*, left, d_exec>,
                          Tensor<int*, left, h_exec>>::test(expected, expected,
                                                          10);
  expected = flare::detail::MemorySpaceAccess<h_exec, d_exec>::assignable;
  detail::TestAssignability<Tensor<int*, left, h_exec>,
                          Tensor<int*, left, d_exec>>::test(expected, expected,
                                                          10);

  // reference type and const-qualified types
  using SomeTensorType = Tensor<int*, left, d_exec>;
  static_assert(is_always_assignable_v<SomeTensorType, SomeTensorType>);
  static_assert(is_always_assignable_v<SomeTensorType, SomeTensorType&>);
  static_assert(is_always_assignable_v<SomeTensorType, SomeTensorType const>);
  static_assert(is_always_assignable_v<SomeTensorType, SomeTensorType const&>);
  static_assert(is_always_assignable_v<SomeTensorType&, SomeTensorType>);
  static_assert(is_always_assignable_v<SomeTensorType&, SomeTensorType&>);
  static_assert(is_always_assignable_v<SomeTensorType&, SomeTensorType const>);
  static_assert(is_always_assignable_v<SomeTensorType&, SomeTensorType const&>);
}
}  // namespace Test
