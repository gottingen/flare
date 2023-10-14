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

#ifndef FLARE_DUAL_TENSOR_TEST_H_
#define FLARE_DUAL_TENSOR_TEST_H_

#include <doctest.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <flare/timer.h>
#include <flare/dual_tensor.h>
#include <flare/core.h>
namespace Test {

namespace detail {
template <typename Scalar, class Device>
struct test_dualtensor_alloc {
  using scalar_type     = Scalar;
  using execution_space = Device;

  template <typename TensorType>
  bool run_me(unsigned int n, unsigned int m) {
    if (n < 10) n = 10;
    if (m < 3) m = 3;

    {
      TensorType b1;
      if (b1.is_allocated() == true) return false;

      b1 = TensorType("B1", n, m);
      TensorType b2(b1);
      TensorType b3("B3", n, m);

      if (b1.is_allocated() == false) return false;
      if (b2.is_allocated() == false) return false;
      if (b3.is_allocated() == false) return false;
    }
    return true;
  }

  bool result = false;

  test_dualtensor_alloc(unsigned int size) {
    result = run_me<flare::DualTensor<Scalar**, flare::LayoutLeft, Device> >(
        size, 3);
  }
};

template <typename Scalar, class Device>
struct test_dualtensor_combinations {
  using self_type = test_dualtensor_combinations<Scalar, Device>;

  using scalar_type     = Scalar;
  using execution_space = Device;

  Scalar reference;
  Scalar result;

  template <typename TensorType>
  Scalar run_me(unsigned int n, unsigned int m, bool with_init) {
    if (n < 10) n = 10;
    if (m < 3) m = 3;

    TensorType a;

    if (with_init) {
      a = TensorType("A", n, m);
    } else {
      a = TensorType(flare::tensor_alloc(flare::WithoutInitializing, "A"), n, m);
    }
    flare::deep_copy(a.d_tensor, 1);

    a.template modify<typename TensorType::execution_space>();
    a.template sync<typename TensorType::host_mirror_space>();
    a.template sync<typename TensorType::host_mirror_space>(
        flare::DefaultExecutionSpace{});

    a.h_tensor(5, 1) = 3;
    a.h_tensor(6, 1) = 4;
    a.h_tensor(7, 2) = 5;
    a.template modify<typename TensorType::host_mirror_space>();
    TensorType b = flare::subtensor(a, std::pair<unsigned int, unsigned int>(6, 9),
                                 std::pair<unsigned int, unsigned int>(0, 1));
    a.template sync<typename TensorType::execution_space>();
    a.template sync<typename TensorType::execution_space>(
        flare::DefaultExecutionSpace{});
    b.template modify<typename TensorType::execution_space>();

    flare::deep_copy(b.d_tensor, 2);

    a.template sync<typename TensorType::host_mirror_space>();
    a.template sync<typename TensorType::host_mirror_space>(
        flare::DefaultExecutionSpace{});
    Scalar count = 0;
    for (unsigned int i = 0; i < a.d_tensor.extent(0); i++)
      for (unsigned int j = 0; j < a.d_tensor.extent(1); j++)
        count += a.h_tensor(i, j);
    return count - a.d_tensor.extent(0) * a.d_tensor.extent(1) - 2 - 4 - 3 * 2;
  }

  test_dualtensor_combinations(unsigned int size, bool with_init) {
    result = run_me<flare::DualTensor<Scalar**, flare::LayoutLeft, Device> >(
        size, 3, with_init);
  }
};

template <typename Scalar, class TensorType>
struct SumTensorEntriesFunctor {
  using value_type = Scalar;

  TensorType fv;

  SumTensorEntriesFunctor(const TensorType& fv_) : fv(fv_) {}

  FLARE_INLINE_FUNCTION
  void operator()(const int i, value_type& total) const {
    for (size_t j = 0; j < fv.extent(1); ++j) {
      total += fv(i, j);
    }
  }
};

template <typename Scalar, class Device>
struct test_dual_tensor_deep_copy {
  using scalar_type     = Scalar;
  using execution_space = Device;

  template <typename TensorType>
  void run_me(int n, const int m, const bool use_templ_sync) {
    TensorType a, b;
    if (n >= 0) {
      a = TensorType("A", n, m);
      b = TensorType("B", n, m);
    } else {
      n = 0;
    }
    const scalar_type sum_total = scalar_type(n * m);

    flare::deep_copy(a.d_tensor, 1);

    if (use_templ_sync) {
      a.template modify<typename TensorType::execution_space>();
      a.template sync<typename TensorType::host_mirror_space>();
    } else {
      a.modify_device();
      a.sync_host();
      a.sync_host(flare::DefaultExecutionSpace{});
    }

    // Check device tensor is initialized as expected
    scalar_type a_d_sum = 0;
    // Execute on the execution_space associated with t_dev's memory space
    using t_dev_exec_space =
        typename TensorType::t_dev::memory_space::execution_space;
    flare::parallel_reduce(
        flare::RangePolicy<t_dev_exec_space>(0, n),
        SumTensorEntriesFunctor<scalar_type, typename TensorType::t_dev>(a.d_tensor),
        a_d_sum);
    REQUIRE_EQ(a_d_sum, sum_total);

    // Check host tensor is synced as expected
    scalar_type a_h_sum = 0;
    for (size_t i = 0; i < a.h_tensor.extent(0); ++i)
      for (size_t j = 0; j < a.h_tensor.extent(1); ++j) {
        a_h_sum += a.h_tensor(i, j);
      }

    REQUIRE_EQ(a_h_sum, sum_total);

    // Test deep_copy
    flare::deep_copy(b, a);
    if (use_templ_sync) {
      b.template sync<typename TensorType::host_mirror_space>();
    } else {
      b.sync_host();
      b.sync_host(flare::DefaultExecutionSpace{});
    }

    // Perform same checks on b as done on a
    // Check device tensor is initialized as expected
    scalar_type b_d_sum = 0;
    // Execute on the execution_space associated with t_dev's memory space
    flare::parallel_reduce(
        flare::RangePolicy<t_dev_exec_space>(0, n),
        SumTensorEntriesFunctor<scalar_type, typename TensorType::t_dev>(b.d_tensor),
        b_d_sum);
    REQUIRE_EQ(b_d_sum, sum_total);

    // Check host tensor is synced as expected
    scalar_type b_h_sum = 0;
    for (size_t i = 0; i < b.h_tensor.extent(0); ++i)
      for (size_t j = 0; j < b.h_tensor.extent(1); ++j) {
        b_h_sum += b.h_tensor(i, j);
      }

    REQUIRE_EQ(b_h_sum, sum_total);

  }  // end run_me

  test_dual_tensor_deep_copy() {
    run_me<flare::DualTensor<Scalar**, flare::LayoutLeft, Device> >(10, 5,
                                                                    true);
    run_me<flare::DualTensor<Scalar**, flare::LayoutLeft, Device> >(10, 5,
                                                                    false);
    // Test zero length but allocated (a.d_tensor.data!=nullptr but
    // a.d_tensor.span()==0)
    run_me<flare::DualTensor<Scalar**, flare::LayoutLeft, Device> >(0, 5, true);
    run_me<flare::DualTensor<Scalar**, flare::LayoutLeft, Device> >(0, 5,
                                                                    false);

    // Test default constructed tensor
    run_me<flare::DualTensor<Scalar**, flare::LayoutLeft, Device> >(-1, 5,
                                                                    true);
    run_me<flare::DualTensor<Scalar**, flare::LayoutLeft, Device> >(-1, 5,
                                                                    false);
  }
};

template <typename Scalar, class Device, bool Initialize>
struct test_dualtensor_resize {
  using scalar_type     = Scalar;
  using execution_space = Device;

  template <typename TensorType>
  void run_me() {
    const unsigned int n      = 10;
    const unsigned int m      = 5;
    const unsigned int factor = 2;

    TensorType a("A", n, m);
    flare::deep_copy(a.d_tensor, 1);

    /* Covers case "Resize on Device" */
    a.modify_device();
    if (Initialize)
      flare::resize(flare::WithoutInitializing, a, factor * n, factor * m);
    else
      flare::resize(a, factor * n, factor * m);
    REQUIRE_EQ(a.extent(0), n * factor);
    REQUIRE_EQ(a.extent(1), m * factor);

    flare::deep_copy(a.d_tensor, 1);
    a.sync_host();

    // Check device tensor is initialized as expected
    scalar_type a_d_sum = 0;
    // Execute on the execution_space associated with t_dev's memory space
    using t_dev_exec_space =
        typename TensorType::t_dev::memory_space::execution_space;
    flare::parallel_reduce(
        flare::RangePolicy<t_dev_exec_space>(0, a.d_tensor.extent(0)),
        SumTensorEntriesFunctor<scalar_type, typename TensorType::t_dev>(a.d_tensor),
        a_d_sum);

    // Check host tensor is synced as expected
    scalar_type a_h_sum = 0;
    for (size_t i = 0; i < a.h_tensor.extent(0); ++i)
      for (size_t j = 0; j < a.h_tensor.extent(1); ++j) {
        a_h_sum += a.h_tensor(i, j);
      }

    // Check
    REQUIRE_EQ(a_h_sum, a_d_sum);
    REQUIRE_EQ(a_h_sum, scalar_type(a.extent(0) * a.extent(1)));

    /* Covers case "Resize on Host" */
    a.modify_host();

    if (Initialize)
      flare::resize(flare::WithoutInitializing, a, n / factor, m / factor);
    else
      flare::resize(a, n / factor, m / factor);
    REQUIRE_EQ(a.extent(0), n / factor);
    REQUIRE_EQ(a.extent(1), m / factor);

    a.sync_device();
    a.sync_device(flare::DefaultExecutionSpace{});

    // Check device tensor is initialized as expected
    a_d_sum = 0;
    // Execute on the execution_space associated with t_dev's memory space
    using t_dev_exec_space =
        typename TensorType::t_dev::memory_space::execution_space;
    flare::parallel_reduce(
        flare::RangePolicy<t_dev_exec_space>(0, a.d_tensor.extent(0)),
        SumTensorEntriesFunctor<scalar_type, typename TensorType::t_dev>(a.d_tensor),
        a_d_sum);

    // Check host tensor is synced as expected
    a_h_sum = 0;
    for (size_t i = 0; i < a.h_tensor.extent(0); ++i)
      for (size_t j = 0; j < a.h_tensor.extent(1); ++j) {
        a_h_sum += a.h_tensor(i, j);
      }

    // Check
    REQUIRE_EQ(a_h_sum, scalar_type(a.extent(0) * a.extent(1)));
    REQUIRE_EQ(a_h_sum, a_d_sum);

  }  // end run_me

  test_dualtensor_resize() {
    run_me<flare::DualTensor<Scalar**, flare::LayoutLeft, Device> >();
  }
};

template <typename Scalar, class Device, bool Initialize>
struct test_dualtensor_realloc {
  using scalar_type     = Scalar;
  using execution_space = Device;

  template <typename TensorType>
  void run_me() {
    const unsigned int n = 10;
    const unsigned int m = 5;

    TensorType a("A", n, m);
    if (Initialize)
      flare::realloc(flare::WithoutInitializing, a, n, m);
    else
      flare::realloc(a, n, m);

    flare::deep_copy(a.d_tensor, 1);
    a.modify_device();
    a.sync_host();

    // Check device tensor is initialized as expected
    scalar_type a_d_sum = 0;
    // Execute on the execution_space associated with t_dev's memory space
    using t_dev_exec_space =
        typename TensorType::t_dev::memory_space::execution_space;
    flare::parallel_reduce(
        flare::RangePolicy<t_dev_exec_space>(0, a.d_tensor.extent(0)),
        SumTensorEntriesFunctor<scalar_type, typename TensorType::t_dev>(a.d_tensor),
        a_d_sum);

    // Check host tensor is synced as expected
    scalar_type a_h_sum = 0;
    for (size_t i = 0; i < a.h_tensor.extent(0); ++i)
      for (size_t j = 0; j < a.h_tensor.extent(1); ++j) {
        a_h_sum += a.h_tensor(i, j);
      }

    // Check
    REQUIRE_EQ(a_h_sum, scalar_type(a.extent(0) * a.extent(1)));
    REQUIRE_EQ(a_h_sum, a_d_sum);
  }  // end run_me

  test_dualtensor_realloc() {
    run_me<flare::DualTensor<Scalar**, flare::LayoutLeft, Device> >();
  }
};

}  // namespace detail

template <typename Scalar, typename Device>
void test_dualtensor_combinations(unsigned int size, bool with_init) {
  detail::test_dualtensor_combinations<Scalar, Device> test(size, with_init);
  REQUIRE_EQ(test.result, 0);
}

template <typename Scalar, typename Device>
void test_dualtensor_alloc(unsigned int size) {
  detail::test_dualtensor_alloc<Scalar, Device> test(size);
  REQUIRE(test.result);
}

template <typename Scalar, typename Device>
void test_dualtensor_deep_copy() {
  detail::test_dual_tensor_deep_copy<Scalar, Device>();
}

template <typename Scalar, typename Device>
void test_dualtensor_realloc() {
  detail::test_dualtensor_realloc<Scalar, Device, false>();
  detail::test_dualtensor_realloc<Scalar, Device, true>();
}

template <typename Scalar, typename Device>
void test_dualtensor_resize() {
  detail::test_dualtensor_resize<Scalar, Device, false>();
  detail::test_dualtensor_resize<Scalar, Device, true>();
}

TEST_CASE("TEST_CATEGORY, dualtensor_combination") {
  test_dualtensor_combinations<int, TEST_EXECSPACE>(10, true);
}

TEST_CASE("TEST_CATEGORY, dualtensor_alloc") {
  test_dualtensor_alloc<int, TEST_EXECSPACE>(10);
}

TEST_CASE("TEST_CATEGORY, dualtensor_combinations_without_init") {
  test_dualtensor_combinations<int, TEST_EXECSPACE>(10, false);
}

TEST_CASE("TEST_CATEGORY, dualtensor_deep_copy") {
  test_dualtensor_deep_copy<int, TEST_EXECSPACE>();
  test_dualtensor_deep_copy<double, TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, dualtensor_realloc") {
  test_dualtensor_realloc<int, TEST_EXECSPACE>();
}

TEST_CASE("TEST_CATEGORY, dualtensor_resize") {
  test_dualtensor_resize<int, TEST_EXECSPACE>();
}

namespace {

// modify if we have other UVM enabled backends
#if defined(FLARE_ON_CUDA_DEVICE)
#define UVM_ENABLED_BUILD
#endif

#ifdef UVM_ENABLED_BUILD
template <typename ExecSpace>
struct UVMSpaceFor;
#endif

#ifdef FLARE_ON_CUDA_DEVICE  // specific to CUDA
template <>
struct UVMSpaceFor<flare::Cuda> {
  using type = flare::CudaUVMSpace;
};
#endif

#ifdef UVM_ENABLED_BUILD
template <>
struct UVMSpaceFor<flare::DefaultHostExecutionSpace> {
  using type = typename UVMSpaceFor<flare::DefaultExecutionSpace>::type;
};
#else
template <typename ExecSpace>
struct UVMSpaceFor {
  using type = typename ExecSpace::memory_space;
};
#endif

using ExecSpace  = flare::DefaultExecutionSpace;
using MemSpace   = typename UVMSpaceFor<flare::DefaultExecutionSpace>::type;
using DeviceType = flare::Device<ExecSpace, MemSpace>;

using DualTensorType = flare::DualTensor<double*, flare::LayoutLeft, DeviceType>;
using d_device     = DeviceType;
using h_device     = flare::Device<
    flare::DefaultHostExecutionSpace,
    typename UVMSpaceFor<flare::DefaultHostExecutionSpace>::type>;

TEST_CASE("TEST_CATEGORY, dualtensor_device_correct_flare_device") {
  DualTensorType dv("myTensor", 100);
  dv.clear_sync_state();
  auto v_d      = dv.template tensor<d_device>();
  using vdt     = decltype(v_d);
  using vdt_d   = vdt::device_type;
  using vdt_d_e = vdt_d::execution_space;
  REQUIRE_EQ(vdt_d_e::name(), flare::DefaultExecutionSpace::name());
}
TEST_CASE("TEST_CATEGORY, dualtensor_host_correct_flare_device") {
  DualTensorType dv("myTensor", 100);
  dv.clear_sync_state();
  auto v_h      = dv.template tensor<h_device>();
  using vht     = decltype(v_h);
  using vht_d   = vht::device_type;
  using vht_d_e = vht_d::execution_space;
  REQUIRE_EQ(vht_d_e::name(), flare::DefaultHostExecutionSpace::name());
}

TEST_CASE("TEST_CATEGORY, dualtensor_host_modify_template_device_sync") {
  DualTensorType dv("myTensor", 100);
  dv.clear_sync_state();
  dv.modify_host();
  dv.template sync<d_device>();
  REQUIRE(!dv.need_sync_device());
  REQUIRE(!dv.need_sync_host());
  dv.clear_sync_state();
}

TEST_CASE("TEST_CATEGORY, dualtensor_host_modify_template_device_execspace_sync") {
  DualTensorType dv("myTensor", 100);
  dv.clear_sync_state();
  dv.modify_host();
  dv.template sync<d_device::execution_space>();
  REQUIRE(!dv.need_sync_device());
  REQUIRE(!dv.need_sync_host());
  dv.clear_sync_state();
}

TEST_CASE("TEST_CATEGORY, dualtensor_device_modify_template_host_sync") {
  DualTensorType dv("myTensor", 100);
  dv.clear_sync_state();
  dv.modify_device();
  dv.template sync<h_device>();
  REQUIRE(!dv.need_sync_device());
  REQUIRE(!dv.need_sync_host());
  dv.clear_sync_state();
}
TEST_CASE("TEST_CATEGORY, dualtensor_device_modify_template_host_execspace_sync") {
  DualTensorType dv("myTensor", 100);
  dv.clear_sync_state();
  dv.modify_device();
  dv.template sync<h_device::execution_space>();
  REQUIRE(!dv.need_sync_device());
  REQUIRE(!dv.need_sync_host());
  dv.clear_sync_state();
}

TEST_CASE("TEST_CATEGORY, dualtensor_template_tensors_return_correct_executionspace_tensors") {
  DualTensorType dv("myTensor", 100);
  dv.clear_sync_state();
  using hvt = decltype(dv.tensor<typename flare::DefaultHostExecutionSpace>());
  using dvt = decltype(dv.tensor<typename flare::DefaultExecutionSpace>());
  REQUIRE_EQ(flare::DefaultExecutionSpace::name(),
               dvt::device_type::execution_space::name());
  REQUIRE_EQ(flare::DefaultHostExecutionSpace::name(),
               hvt::device_type::execution_space::name());
}

}  // anonymous namespace
}  // namespace Test

#endif  // FLARE_DUAL_TENSOR_TEST_H_
