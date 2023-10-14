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
#include <sstream>
#include <iostream>

#include <flare/core.h>

namespace Test {

/*--------------------------------------------------------------------------*/

template <class Space>
struct TestTensorMappingAtomic {
  using ExecSpace = typename Space::execution_space;
  using MemSpace  = typename Space::memory_space;

  using mem_trait = flare::MemoryTraits<flare::Atomic>;

  using T      = flare::Tensor<int *, ExecSpace>;
  using T_atom = flare::Tensor<int *, ExecSpace, mem_trait>;

  T x;
  T_atom x_atom;

  enum { N = 100000 };

  struct TagInit {};
  struct TagUpdate {};
  struct TagVerify {};

  FLARE_INLINE_FUNCTION
  void operator()(const TagInit &, const int i) const { x(i) = i; }

  FLARE_INLINE_FUNCTION
  void operator()(const TagUpdate &, const int i) const { x_atom(i % 2) += 1; }

  FLARE_INLINE_FUNCTION
  void operator()(const TagVerify &, const int i, long &error_count) const {
    if (i < 2) {
      if (x(i) != int(i + N / 2)) ++error_count;
    } else {
      if (x(i) != int(i)) ++error_count;
    }
  }

  TestTensorMappingAtomic() : x("x", N), x_atom(x) {}

  void run() {
    REQUIRE(T::reference_type_is_lvalue_reference);
      REQUIRE_FALSE(T_atom::reference_type_is_lvalue_reference);

    flare::parallel_for(flare::RangePolicy<ExecSpace, TagInit>(0, N), *this);
    flare::parallel_for(flare::RangePolicy<ExecSpace, TagUpdate>(0, N),
                         *this);

    long error_count = -1;

    flare::parallel_reduce(flare::RangePolicy<ExecSpace, TagVerify>(0, N),
                            *this, error_count);

    REQUIRE_EQ(0, error_count);

    typename T_atom::HostMirror x_host = flare::create_mirror_tensor(x);
    flare::deep_copy(x_host, x);

    error_count = -1;

    flare::parallel_reduce(
        flare::RangePolicy<flare::DefaultHostExecutionSpace, TagVerify>(0, N),
        [=](const TagVerify &, const int i, long &tmp_error_count) {
          if (i < 2) {
            if (x_host(i) != int(i + N / 2)) ++tmp_error_count;
          } else {
            if (x_host(i) != int(i)) ++tmp_error_count;
          }
        },
        error_count);

    REQUIRE_EQ(0, error_count);
    flare::deep_copy(x, x_host);
  }
};

TEST_CASE("TEST_CATEGORY, tensor_mapping_atomic") {
  TestTensorMappingAtomic<TEST_EXECSPACE> f;
  f.run();
}

}  // namespace Test

/*--------------------------------------------------------------------------*/

namespace Test {

struct MappingClassValueType {
  FLARE_INLINE_FUNCTION
  MappingClassValueType() {
#if 0
    FLARE_IF_ON_DEVICE(
        (printf("TestTensorMappingClassValue construct on Device\n");))
    FLARE_IF_ON_HOST((printf("TestTensorMappingClassValue construct on Host\n");))
#endif
  }
  FLARE_INLINE_FUNCTION
  ~MappingClassValueType() {
#if 0
    FLARE_IF_ON_DEVICE(
        (printf("TestTensorMappingClassValue destruct on Device\n");))
    FLARE_IF_ON_HOST((printf("TestTensorMappingClassValue destruct on Host\n");))
#endif
  }
};

template <class Space>
void test_tensor_mapping_class_value() {
  using ExecSpace = typename Space::execution_space;

  ExecSpace().fence();
  {
    flare::Tensor<MappingClassValueType, ExecSpace> a("a");
    ExecSpace().fence();
  }
  ExecSpace().fence();
}

TEST_CASE("TEST_CATEGORY, tensor_mapping_class_value") {
  test_tensor_mapping_class_value<TEST_EXECSPACE>();
}

}  // namespace Test

/*--------------------------------------------------------------------------*/

namespace Test {

TEST_CASE("TEST_CATEGORY, tensor_mapping_assignable") {
  using exec_space = TEST_EXECSPACE;

  {  // Assignment of rank-0 Left = Right
    using dst_traits = flare::TensorTraits<int, flare::LayoutLeft, exec_space>;
    using src_traits = flare::TensorTraits<int, flare::LayoutRight, exec_space>;
    using mapping    = flare::detail::TensorMapping<dst_traits, src_traits, void>;
    static_assert(mapping::is_assignable, "");

    flare::Tensor<int, flare::LayoutRight, exec_space> src;
    flare::Tensor<int, flare::LayoutLeft, exec_space> dst(src);
    dst = src;
  }

  {  // Assignment of rank-0 Right = Left
    using dst_traits = flare::TensorTraits<int, flare::LayoutRight, exec_space>;
    using src_traits = flare::TensorTraits<int, flare::LayoutLeft, exec_space>;
    using mapping    = flare::detail::TensorMapping<dst_traits, src_traits, void>;
    static_assert(mapping::is_assignable, "");

    flare::Tensor<int, flare::LayoutLeft, exec_space> src;
    flare::Tensor<int, flare::LayoutRight, exec_space> dst(src);
    dst = src;
  }

  {  // Assignment of rank-1 Left = Right
    using dst_traits =
        flare::TensorTraits<int *, flare::LayoutLeft, exec_space>;
    using src_traits =
        flare::TensorTraits<int *, flare::LayoutRight, exec_space>;
    using mapping = flare::detail::TensorMapping<dst_traits, src_traits, void>;
    static_assert(mapping::is_assignable, "");

    flare::Tensor<int *, flare::LayoutRight, exec_space> src;
    flare::Tensor<int *, flare::LayoutLeft, exec_space> dst(src);
    dst = src;
  }

  {  // Assignment of rank-1 Right = Left
    using dst_traits =
        flare::TensorTraits<int *, flare::LayoutRight, exec_space>;
    using src_traits =
        flare::TensorTraits<int *, flare::LayoutLeft, exec_space>;
    using mapping = flare::detail::TensorMapping<dst_traits, src_traits, void>;
    static_assert(mapping::is_assignable, "");

    flare::Tensor<int *, flare::LayoutLeft, exec_space> src;
    flare::Tensor<int *, flare::LayoutRight, exec_space> dst(src);
    dst = src;
  }

  {  // Assignment of rank-2 Left = Right
    using dst_traits =
        flare::TensorTraits<int **, flare::LayoutLeft, exec_space>;
    using src_traits =
        flare::TensorTraits<int **, flare::LayoutRight, exec_space>;
    using mapping = flare::detail::TensorMapping<dst_traits, src_traits, void>;
    static_assert(!mapping::is_assignable, "");
  }

  {  // Assignment of rank-2 Right = Left
    using dst_traits =
        flare::TensorTraits<int **, flare::LayoutRight, exec_space>;
    using src_traits =
        flare::TensorTraits<int **, flare::LayoutLeft, exec_space>;
    using mapping = flare::detail::TensorMapping<dst_traits, src_traits, void>;
    static_assert(!mapping::is_assignable, "");
  }
}

TEST_CASE("TEST_CATEGORY, tensor_mapping_trivially_copyable") {
  using exec_space = TEST_EXECSPACE;

  using dst_traits = flare::TensorTraits<int *, exec_space>;
  using src_traits = dst_traits;
  using mapping    = flare::detail::TensorMapping<dst_traits, src_traits, void>;

  static_assert(std::is_trivially_copyable<mapping>{}, "");
}

}  // namespace Test
