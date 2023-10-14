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

TEST_CASE("TEST_CATEGORY, tensor_remap") {
  enum { N0 = 3, N1 = 2, N2 = 8, N3 = 9 };

#if defined(FLARE_ON_CUDA_DEVICE)
#define EXECSPACE                                                     \
  std::conditional<std::is_same<TEST_EXECSPACE, flare::Cuda>::value, \
                   flare::CudaHostPinnedSpace, TEST_EXECSPACE>::type
#else
#define EXECSPACE TEST_EXECSPACE
#endif

  using output_type =
      flare::Tensor<double * [N1][N2][N3], flare::LayoutRight, EXECSPACE>;

  using input_type =
      flare::Tensor<int* * [N2][N3], flare::LayoutLeft, EXECSPACE>;

  using diff_type =
      flare::Tensor<int * [N0][N2][N3], flare::LayoutLeft, EXECSPACE>;

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

TEST_CASE("TEST_CATEGORY, tensor_mirror_nonconst") {
  flare::Tensor<int*, TEST_EXECSPACE> d_tensor("d_tensor", 10);
  flare::Tensor<const int*, TEST_EXECSPACE> d_tensor_const = d_tensor;
  auto h_tensor = flare::create_mirror(d_tensor_const);
  flare::deep_copy(h_tensor, d_tensor_const);
  auto h_tensor2 = flare::create_mirror(flare::HostSpace(), d_tensor_const);
  flare::deep_copy(h_tensor2, d_tensor_const);
}

template <typename DataType, typename... Extents>
void test_left_stride(Extents... extents) {
  using tensor_type =
      flare::Tensor<DataType, flare::LayoutLeft, flare::HostSpace>;
  tensor_type tensor("tensor", extents...);
  size_t expected_stride = 1;
  size_t all_strides[tensor_type::rank + 1];
  tensor.stride(all_strides);
  for (size_t i = 0; i < tensor_type::rank; ++i) {
    REQUIRE_EQ(tensor.stride(i), expected_stride);
    REQUIRE_EQ(all_strides[i], expected_stride);
    expected_stride *= tensor.extent(i);
  }
}

template <typename DataType, typename... Extents>
void test_right_stride(Extents... extents) {
  using tensor_type =
      flare::Tensor<DataType, flare::LayoutRight, flare::HostSpace>;
  tensor_type tensor("tensor", extents...);
  size_t expected_stride = 1;
  size_t all_strides[tensor_type::rank + 1];
  tensor.stride(all_strides);
  for (size_t ri = 0; ri < tensor_type::rank; ++ri) {
    auto i = tensor_type::rank - 1 - ri;
    REQUIRE_EQ(tensor.stride(i), expected_stride);
    REQUIRE_EQ(all_strides[i], expected_stride);
    expected_stride *= tensor.extent(i);
  }
}

template <typename DataType, typename... Extents>
void test_stride(Extents... extents) {
  test_right_stride<DataType>(extents...);
  test_left_stride<DataType>(extents...);
}

TEST_CASE("TEST_CATEGORY, tensor_stride_method") {
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
  flare::Tensor<int[10], flare::AnonymousSpace> host_anon_stat_tensor(host_array);
  flare::Tensor<int*, flare::AnonymousSpace> host_anon_dyn_tensor(host_array, 10);
  flare::Tensor<int*, flare::HostSpace> host_tensor("host_tensor", 10);
  flare::Tensor<int*, flare::AnonymousSpace> host_anon_assign_tensor = host_tensor;
  for (int i = 0; i < 10; ++i) {
    host_anon_stat_tensor(i) = host_anon_dyn_tensor(i) = 142;
    host_anon_assign_tensor(i)                       = 142;
  }
  flare::Tensor<int**, flare::LayoutRight, ExecSpace> d_tensor("d_tensor", 100, 10);
#ifdef FLARE_ENABLE_CXX11_DISPATCH_LAMBDA
  flare::parallel_for(
      flare::RangePolicy<ExecSpace, int>(0, 100), FLARE_LAMBDA(int i) {
        int* ptr = &(d_tensor(i, 0));
        flare::Tensor<int[10], flare::AnonymousSpace> d_anon_stat_tensor(ptr);
        flare::Tensor<int*, flare::AnonymousSpace> d_anon_dyn_tensor(ptr, 10);
        auto sub = flare::subtensor(d_tensor, i, flare::ALL());
        flare::Tensor<int*, flare::AnonymousSpace> d_anon_assign_tensor = sub;
        for (int j = 0; j < 10; ++j) {
          d_anon_stat_tensor(j) = 50;
          d_anon_assign_tensor(j) += 50;
          d_anon_dyn_tensor(j) += 42;
        }
      });
  flare::fence();
#endif
}

TEST_CASE("TEST_CATEGORY, anonymous_space") { test_anonymous_space(); }

template <class ExecSpace>
struct TestTensorOverloadResolution {
  // Overload based on value_type and rank
  static int foo(flare::Tensor<const double**, ExecSpace> /*a*/) { return 1; }
  static int foo(flare::Tensor<const int**, ExecSpace> /*a*/) { return 2; }
  static int foo(flare::Tensor<const double***, ExecSpace> /*a*/) { return 3; }

  // Overload based on compile time dimensions
  static int bar(flare::Tensor<double * [3], ExecSpace> /*a*/) { return 4; }
  static int bar(flare::Tensor<double * [4], ExecSpace> /*a*/) { return 5; }

  static void test_function_overload() {
    flare::Tensor<double**, typename ExecSpace::execution_space::array_layout,
                 ExecSpace>
        a("A", 10, 3);
    int data_type_1 = foo(a);
    int data_type_3 =
        foo(flare::Tensor<const double**,
                         typename ExecSpace::execution_space::array_layout,
                         ExecSpace>(a));
    flare::Tensor<double***, typename ExecSpace::execution_space::array_layout,
                 ExecSpace>
        b("B", 10, 3, 4);
    int data_type_2 = foo(b);
    flare::Tensor<double * [3],
                 typename ExecSpace::execution_space::array_layout, ExecSpace>
        c(a);
    int static_extent = bar(c);
    REQUIRE_EQ(1, data_type_1);
    REQUIRE_EQ(3, data_type_2);
    REQUIRE_EQ(1, data_type_3);
    REQUIRE_EQ(4, static_extent);
  }
};

TEST_CASE("TEST_CATEGORY, tensor_overload_resolution") {
  TestTensorOverloadResolution<TEST_EXECSPACE>::test_function_overload();
}

template <typename MemorySpace>
struct TestTensorAllocationLargeRank {
  using TensorType = flare::Tensor<char********, MemorySpace>;

  FLARE_FUNCTION void operator()(int) const {
    size_t idx = v.extent(0) - 1;
    auto& lhs  = v(idx, idx, idx, idx, idx, idx, idx, idx);
    lhs        = 42;  // This is where it segfaulted
  }

  TensorType v;
};

TEST_CASE("TEST_CATEGORY, tensor_allocation_large_rank") {
  using ExecutionSpace = typename TEST_EXECSPACE::execution_space;
  using MemorySpace    = typename TEST_EXECSPACE::memory_space;
  constexpr int dim    = 16;
  using FunctorType    = TestTensorAllocationLargeRank<MemorySpace>;
  typename FunctorType::TensorType v("v", dim, dim, dim, dim, dim, dim, dim, dim);

  flare::parallel_for(flare::RangePolicy<ExecutionSpace>(0, 1),
                       FunctorType{v});
  typename FunctorType::TensorType v_single(v.data() + v.size() - 1, 1, 1, 1, 1,
                                          1, 1, 1, 1);
  auto result =
      flare::create_mirror_tensor_and_copy(flare::HostSpace{}, v_single);
  REQUIRE_EQ(result(0, 0, 0, 0, 0, 0, 0, 0), 42);
}

template <typename ExecSpace, typename TensorType>
struct TestTensorShmemSizeOnDevice {
  using TensorTestType = flare::Tensor<size_t, ExecSpace>;

  TestTensorShmemSizeOnDevice(size_t d1_, size_t d2_, size_t d3_)
      : d1(d1_), d2(d2_), d3(d3_), shmemSize("shmemSize") {}

  FLARE_FUNCTION void operator()(const int&) const {
    auto shmem  = TensorType::shmem_size(d1, d2, d3);
    shmemSize() = shmem;
  }

  size_t d1, d2, d3;
  TensorTestType shmemSize;
};

TEST_CASE("TEST_CATEGORY, tensor_shmem_size_on_device") {
  using ExecSpace = typename TEST_EXECSPACE::execution_space;
  using TensorType  = flare::Tensor<int64_t***, ExecSpace>;

  constexpr size_t d1 = 5;
  constexpr size_t d2 = 7;
  constexpr size_t d3 = 11;

  TestTensorShmemSizeOnDevice<ExecSpace, TensorType> testShmemSize(d1, d2, d3);

  flare::parallel_for(flare::RangePolicy<ExecSpace>(0, 1), testShmemSize);

  auto size = TensorType::shmem_size(d1, d2, d3);

  auto shmemSizeHost = flare::create_mirror_tensor_and_copy(
      flare::HostSpace(), testShmemSize.shmemSize);

  REQUIRE_EQ(size, shmemSizeHost());
}

}  // namespace Test

#include <tensor/tensor_is_assignable_test.h>
