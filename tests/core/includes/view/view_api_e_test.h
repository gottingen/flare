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

#include <flare/core.h>
#include <sstream>
#include <iostream>

namespace Test {

TEST_CASE("TEST_CATEGORY, view_remap") {
  enum { N0 = 3, N1 = 2, N2 = 8, N3 = 9 };

#if defined(FLARE_ON_CUDA_DEVICE)
#define EXECSPACE                                                     \
  std::conditional<std::is_same<TEST_EXECSPACE, flare::Cuda>::value, \
                   flare::CudaHostPinnedSpace, TEST_EXECSPACE>::type
#else
#define EXECSPACE TEST_EXECSPACE
#endif

  using output_type =
      flare::View<double * [N1][N2][N3], flare::LayoutRight, EXECSPACE>;

  using input_type =
      flare::View<int* * [N2][N3], flare::LayoutLeft, EXECSPACE>;

  using diff_type =
      flare::View<int * [N0][N2][N3], flare::LayoutLeft, EXECSPACE>;

  output_type output("output", N0);
  input_type input("input", N0, N1);
  diff_type diff("diff", N0);

  flare::fence();
  int value = 0;

  for (size_t i3 = 0; i3 < N3; ++i3)
    for (size_t i2 = 0; i2 < N2; ++i2)
      for (size_t i1 = 0; i1 < N1; ++i1)
        for (size_t i0 = 0; i0 < N0; ++i0) {
          input(i0, i1, i2, i3) = ++value;
        }

  flare::fence();
  // flare::deep_copy( diff, input ); // Throw with incompatible shape.
  flare::deep_copy(output, input);
  flare::fence();

  value = 0;

  for (size_t i3 = 0; i3 < N3; ++i3)
    for (size_t i2 = 0; i2 < N2; ++i2)
      for (size_t i1 = 0; i1 < N1; ++i1)
        for (size_t i0 = 0; i0 < N0; ++i0) {
          ++value;
          REQUIRE_EQ(value, ((int)output(i0, i1, i2, i3)));
        }
}

TEST_CASE("TEST_CATEGORY, view_mirror_nonconst") {
  flare::View<int*, TEST_EXECSPACE> d_view("d_view", 10);
  flare::View<const int*, TEST_EXECSPACE> d_view_const = d_view;
  auto h_view = flare::create_mirror(d_view_const);
  flare::deep_copy(h_view, d_view_const);
  auto h_view2 = flare::create_mirror(flare::HostSpace(), d_view_const);
  flare::deep_copy(h_view2, d_view_const);
}

template <typename DataType, typename... Extents>
void test_left_stride(Extents... extents) {
  using view_type =
      flare::View<DataType, flare::LayoutLeft, flare::HostSpace>;
  view_type view("view", extents...);
  size_t expected_stride = 1;
  size_t all_strides[view_type::rank + 1];
  view.stride(all_strides);
  for (size_t i = 0; i < view_type::rank; ++i) {
    REQUIRE_EQ(view.stride(i), expected_stride);
    REQUIRE_EQ(all_strides[i], expected_stride);
    expected_stride *= view.extent(i);
  }
}

template <typename DataType, typename... Extents>
void test_right_stride(Extents... extents) {
  using view_type =
      flare::View<DataType, flare::LayoutRight, flare::HostSpace>;
  view_type view("view", extents...);
  size_t expected_stride = 1;
  size_t all_strides[view_type::rank + 1];
  view.stride(all_strides);
  for (size_t ri = 0; ri < view_type::rank; ++ri) {
    auto i = view_type::rank - 1 - ri;
    REQUIRE_EQ(view.stride(i), expected_stride);
    REQUIRE_EQ(all_strides[i], expected_stride);
    expected_stride *= view.extent(i);
  }
}

template <typename DataType, typename... Extents>
void test_stride(Extents... extents) {
  test_right_stride<DataType>(extents...);
  test_left_stride<DataType>(extents...);
}

TEST_CASE("TEST_CATEGORY, view_stride_method") {
  test_stride<double[3]>();
  test_stride<double*>(3);
  test_stride<double[3][7][13]>();
  test_stride<double***>(3, 7, 13);
  // factorial(8) = 40320
  test_stride<double[1][2][3][4][5][6][7][8]>();
  test_stride<double********>(1, 2, 3, 4, 5, 6, 7, 8);
}

inline void test_anonymous_space() {
  /* apparently TEST_EXECSPACE is sometimes a memory space. */
  using ExecSpace = TEST_EXECSPACE::execution_space;
  int host_array[10];
  flare::View<int[10], flare::AnonymousSpace> host_anon_stat_view(host_array);
  flare::View<int*, flare::AnonymousSpace> host_anon_dyn_view(host_array, 10);
  flare::View<int*, flare::HostSpace> host_view("host_view", 10);
  flare::View<int*, flare::AnonymousSpace> host_anon_assign_view = host_view;
  for (int i = 0; i < 10; ++i) {
    host_anon_stat_view(i) = host_anon_dyn_view(i) = 142;
    host_anon_assign_view(i)                       = 142;
  }
  flare::View<int**, flare::LayoutRight, ExecSpace> d_view("d_view", 100, 10);
#ifdef FLARE_ENABLE_CXX11_DISPATCH_LAMBDA
  flare::parallel_for(
      flare::RangePolicy<ExecSpace, int>(0, 100), FLARE_LAMBDA(int i) {
        int* ptr = &(d_view(i, 0));
        flare::View<int[10], flare::AnonymousSpace> d_anon_stat_view(ptr);
        flare::View<int*, flare::AnonymousSpace> d_anon_dyn_view(ptr, 10);
        auto sub = flare::subview(d_view, i, flare::ALL());
        flare::View<int*, flare::AnonymousSpace> d_anon_assign_view = sub;
        for (int j = 0; j < 10; ++j) {
          d_anon_stat_view(j) = 50;
          d_anon_assign_view(j) += 50;
          d_anon_dyn_view(j) += 42;
        }
      });
  flare::fence();
#endif
}

TEST_CASE("TEST_CATEGORY, anonymous_space") { test_anonymous_space(); }

template <class ExecSpace>
struct TestViewOverloadResolution {
  // Overload based on value_type and rank
  static int foo(flare::View<const double**, ExecSpace> /*a*/) { return 1; }
  static int foo(flare::View<const int**, ExecSpace> /*a*/) { return 2; }
  static int foo(flare::View<const double***, ExecSpace> /*a*/) { return 3; }

  // Overload based on compile time dimensions
  static int bar(flare::View<double * [3], ExecSpace> /*a*/) { return 4; }
  static int bar(flare::View<double * [4], ExecSpace> /*a*/) { return 5; }

  static void test_function_overload() {
    flare::View<double**, typename ExecSpace::execution_space::array_layout,
                 ExecSpace>
        a("A", 10, 3);
    int data_type_1 = foo(a);
    int data_type_3 =
        foo(flare::View<const double**,
                         typename ExecSpace::execution_space::array_layout,
                         ExecSpace>(a));
    flare::View<double***, typename ExecSpace::execution_space::array_layout,
                 ExecSpace>
        b("B", 10, 3, 4);
    int data_type_2 = foo(b);
    flare::View<double * [3],
                 typename ExecSpace::execution_space::array_layout, ExecSpace>
        c(a);
    int static_extent = bar(c);
    REQUIRE_EQ(1, data_type_1);
    REQUIRE_EQ(3, data_type_2);
    REQUIRE_EQ(1, data_type_3);
    REQUIRE_EQ(4, static_extent);
  }
};

TEST_CASE("TEST_CATEGORY, view_overload_resolution") {
  TestViewOverloadResolution<TEST_EXECSPACE>::test_function_overload();
}

template <typename MemorySpace>
struct TestViewAllocationLargeRank {
  using ViewType = flare::View<char********, MemorySpace>;

  FLARE_FUNCTION void operator()(int) const {
    size_t idx = v.extent(0) - 1;
    auto& lhs  = v(idx, idx, idx, idx, idx, idx, idx, idx);
    lhs        = 42;  // This is where it segfaulted
  }

  ViewType v;
};

TEST_CASE("TEST_CATEGORY, view_allocation_large_rank") {
  using ExecutionSpace = typename TEST_EXECSPACE::execution_space;
  using MemorySpace    = typename TEST_EXECSPACE::memory_space;
  constexpr int dim    = 16;
  using FunctorType    = TestViewAllocationLargeRank<MemorySpace>;
  typename FunctorType::ViewType v("v", dim, dim, dim, dim, dim, dim, dim, dim);

  flare::parallel_for(flare::RangePolicy<ExecutionSpace>(0, 1),
                       FunctorType{v});
  typename FunctorType::ViewType v_single(v.data() + v.size() - 1, 1, 1, 1, 1,
                                          1, 1, 1, 1);
  auto result =
      flare::create_mirror_view_and_copy(flare::HostSpace{}, v_single);
  REQUIRE_EQ(result(0, 0, 0, 0, 0, 0, 0, 0), 42);
}

template <typename ExecSpace, typename ViewType>
struct TestViewShmemSizeOnDevice {
  using ViewTestType = flare::View<size_t, ExecSpace>;

  TestViewShmemSizeOnDevice(size_t d1_, size_t d2_, size_t d3_)
      : d1(d1_), d2(d2_), d3(d3_), shmemSize("shmemSize") {}

  FLARE_FUNCTION void operator()(const int&) const {
    auto shmem  = ViewType::shmem_size(d1, d2, d3);
    shmemSize() = shmem;
  }

  size_t d1, d2, d3;
  ViewTestType shmemSize;
};

TEST_CASE("TEST_CATEGORY, view_shmem_size_on_device") {
  using ExecSpace = typename TEST_EXECSPACE::execution_space;
  using ViewType  = flare::View<int64_t***, ExecSpace>;

  constexpr size_t d1 = 5;
  constexpr size_t d2 = 7;
  constexpr size_t d3 = 11;

  TestViewShmemSizeOnDevice<ExecSpace, ViewType> testShmemSize(d1, d2, d3);

  flare::parallel_for(flare::RangePolicy<ExecSpace>(0, 1), testShmemSize);

  auto size = ViewType::shmem_size(d1, d2, d3);

  auto shmemSizeHost = flare::create_mirror_view_and_copy(
      flare::HostSpace(), testShmemSize.shmemSize);

  REQUIRE_EQ(size, shmemSizeHost());
}

}  // namespace Test

#include <view/view_is_assignable_test.h>
