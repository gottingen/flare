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

#ifndef FLARE_test_scatter_tensor_HPP
#define FLARE_test_scatter_tensor_HPP

#include <flare/scatter_tensor.h>
#include <doctest.h>

namespace Test {

template <typename DeviceType, typename Layout, typename Duplication,
          typename Contribution, typename Op, typename NumberType>
struct test_scatter_tensor_impl_cls;

template <typename DeviceType, typename Layout, typename Duplication,
          typename Contribution, typename NumberType>
struct test_scatter_tensor_impl_cls<DeviceType, Layout, Duplication, Contribution,
                                  flare::experimental::ScatterSum,
                                  NumberType> {
 public:
  using scatter_tensor_type =
      flare::experimental::ScatterTensor<NumberType * [12], Layout, DeviceType,
                                        flare::experimental::ScatterSum,
                                        Duplication, Contribution>;

  using orig_tensor_type = flare::Tensor<NumberType * [12], Layout, DeviceType>;

  using size_type = typename flare::HostSpace::size_type;

  scatter_tensor_type scatter_tensor;
  int scatterSize;

  test_scatter_tensor_impl_cls(const scatter_tensor_type& tensor) {
    scatter_tensor = tensor;
    scatterSize  = 0;
  }

  void initialize(orig_tensor_type orig) {
    auto host_tensor =
        flare::create_mirror_tensor_and_copy(flare::HostSpace(), orig);
    flare::fence();
    flare::deep_copy(host_tensor, 0);
    flare::fence();
    flare::deep_copy(orig, host_tensor);
  }

  void run_parallel(int n) {
    scatterSize = n;
    auto policy =
        flare::RangePolicy<typename DeviceType::execution_space, int>(0, n);
    flare::parallel_for("scatter_tensor_test: Sum", policy, *this);
  }

  FLARE_INLINE_FUNCTION
  void operator()(int i) const {
    auto scatter_access = scatter_tensor.access();
    auto scatter_access_atomic =
        scatter_tensor.template access<flare::experimental::ScatterAtomic>();
    for (int j = 0; j < 10; ++j) {
      auto k = (i + j) % scatterSize;
      scatter_access(k, 0) += 4;
      ++scatter_access(k, 1);
      --scatter_access(k, 2);
      scatter_access(k, 3)++;
      scatter_access(k, 4)--;
      scatter_access(k, 5) -= 5;
      scatter_access_atomic(k, 6) += 2;
      scatter_access_atomic(k, 7)++;
      scatter_access_atomic(k, 8)--;
      --scatter_access_atomic(k, 9);
      ++scatter_access_atomic(k, 10);
      scatter_access(k, 11) -= 3;
    }
  }

  void validateResults(orig_tensor_type orig) {
    auto host_tensor =
        flare::create_mirror_tensor_and_copy(flare::HostSpace(), orig);
    flare::fence();
    for (size_type i = 0; i < host_tensor.extent(0); ++i) {
      for (size_type j = 0; j < host_tensor.extent(1); ++j) {
        REQUIRE_LE(std::abs(host_tensor(i, j) - NumberType(ref[j])), 1e-14);
      }
    }
  }

  // check for correct padding
  void validateResultsForSubtensor(
      orig_tensor_type orig, std::pair<size_type, size_type>& subRangeDim0,
      std::pair<size_type, size_type>& subRangeDim1) {
    auto host_tensor =
        flare::create_mirror_tensor_and_copy(flare::HostSpace(), orig);
    flare::fence();
    for (size_type i = 0; i < host_tensor.extent(0); ++i) {
      for (size_type j = 0; j < host_tensor.extent(1); ++j) {
        auto val = host_tensor(i, j);
        if ((i >= std::get<0>(subRangeDim0) && i < std::get<1>(subRangeDim0)) &&
            (j >= std::get<0>(subRangeDim1) && j < std::get<1>(subRangeDim1))) {
          // is in subtensor
            REQUIRE_LE(std::abs(val- NumberType(ref[j])), 1e-14);
        } else {
          // is outside of subtensor
            REQUIRE_LE(std::abs(val - NumberType(0)), 1e-14);
        }
      }
    }
  }

 private:
  NumberType ref[12] = {80, 20, -20, 20, -20, -100, 40, 20, -20, -20, 20, -60};
};

template <typename DeviceType, typename Layout, typename Duplication,
          typename Contribution, typename NumberType>
struct test_scatter_tensor_impl_cls<DeviceType, Layout, Duplication, Contribution,
                                  flare::experimental::ScatterProd,
                                  NumberType> {
 public:
  using scatter_tensor_type =
      flare::experimental::ScatterTensor<NumberType * [3], Layout, DeviceType,
                                        flare::experimental::ScatterProd,
                                        Duplication, Contribution>;

  using orig_tensor_type = flare::Tensor<NumberType * [3], Layout, DeviceType>;

  using size_type = typename flare::HostSpace::size_type;

  scatter_tensor_type scatter_tensor;
  int scatterSize;

  test_scatter_tensor_impl_cls(const scatter_tensor_type& tensor) {
    scatter_tensor = tensor;
    scatterSize  = 0;
  }

  void initialize(orig_tensor_type orig) {
    auto host_tensor =
        flare::create_mirror_tensor_and_copy(flare::HostSpace(), orig);
    flare::fence();
    for (size_type i = 0; i < host_tensor.extent(0); ++i) {
      host_tensor(i, 0) = 1.0;
      host_tensor(i, 1) = 1.0;
      host_tensor(i, 2) = 1.0;
    }
    flare::fence();
    flare::deep_copy(orig, host_tensor);
  }

  void run_parallel(int n) {
    scatterSize = n;
    auto policy =
        flare::RangePolicy<typename DeviceType::execution_space, int>(0, n);
    flare::parallel_for("scatter_tensor_test: Prod", policy, *this);
  }

  FLARE_INLINE_FUNCTION
  void operator()(int i) const {
    auto scatter_access = scatter_tensor.access();
    auto scatter_access_atomic =
        scatter_tensor.template access<flare::experimental::ScatterAtomic>();
    for (int j = 0; j < 4; ++j) {
      auto k = (i + j) % scatterSize;
      scatter_access(k, 0) *= 4.0;
      scatter_access_atomic(k, 1) *= 2.0;
      scatter_access(k, 2) *= 1.0;
    }
  }

  void validateResults(orig_tensor_type orig) {
    auto host_tensor =
        flare::create_mirror_tensor_and_copy(flare::HostSpace(), orig);
    flare::fence();
    for (size_type i = 0; i < host_tensor.extent(0); ++i) {
      auto val0 = host_tensor(i, 0);
      auto val1 = host_tensor(i, 1);
      auto val2 = host_tensor(i, 2);
        REQUIRE_LE(std::abs(val0 - 65536.0), 1e-14 * 65536.0);
        REQUIRE_LE(std::abs(val1 - 256.0), 1e-14 * 256.0);
        REQUIRE_LE(std::abs(val2 - 1.0), 1e-14 * 1.0);
    }
  }

  // check for correct padding
  void validateResultsForSubtensor(
      orig_tensor_type orig, std::pair<size_type, size_type>& subRangeDim0,
      std::pair<size_type, size_type>& subRangeDim1) {
    (void)subRangeDim1;
    auto host_tensor =
        flare::create_mirror_tensor_and_copy(flare::HostSpace(), orig);
    flare::fence();
    for (size_type i = 0; i < host_tensor.extent(0); ++i) {
      auto val0 = host_tensor(i, 0);
      auto val1 = host_tensor(i, 1);
      auto val2 = host_tensor(i, 2);
      if (i >= std::get<0>(subRangeDim0) && i < std::get<1>(subRangeDim0)) {
        // is in subtensor
          REQUIRE_LE(std::abs(val0 - 65536.0), 1e-14 * 65536.0);
          REQUIRE_LE(std::abs(val1 - 256.0), 1e-14 * 256.0);
          REQUIRE_LE(std::abs(val2 - 1.0), 1e-14 * 1.0);
      } else {
        // is outside of subtensor
          REQUIRE_LE(std::abs(val0 - NumberType(1)), 1e-14);
          REQUIRE_LE(std::abs(val1 - NumberType(1)), 1e-14);
          REQUIRE_LE(std::abs(val2 - NumberType(1)), 1e-14);
      }
    }
  }
};

template <typename DeviceType, typename Layout, typename Duplication,
          typename Contribution, typename NumberType>
struct test_scatter_tensor_impl_cls<DeviceType, Layout, Duplication, Contribution,
                                  flare::experimental::ScatterMin,
                                  NumberType> {
 public:
  using scatter_tensor_type =
      flare::experimental::ScatterTensor<NumberType * [3], Layout, DeviceType,
                                        flare::experimental::ScatterMin,
                                        Duplication, Contribution>;

  using orig_tensor_type = flare::Tensor<NumberType * [3], Layout, DeviceType>;

  using size_type = typename flare::HostSpace::size_type;

  scatter_tensor_type scatter_tensor;
  int scatterSize;

  test_scatter_tensor_impl_cls(const scatter_tensor_type& tensor) {
    scatter_tensor = tensor;
    scatterSize  = 0;
  }

  void initialize(orig_tensor_type orig) {
    auto host_tensor =
        flare::create_mirror_tensor_and_copy(flare::HostSpace(), orig);
    flare::fence();
    for (size_type i = 0; i < host_tensor.extent(0); ++i) {
      host_tensor(i, 0) = 999999.0;
      host_tensor(i, 1) = 999999.0;
      host_tensor(i, 2) = 999999.0;
    }
    flare::fence();
    flare::deep_copy(orig, host_tensor);
  }

  void run_parallel(int n) {
    scatterSize = n;
    auto policy =
        flare::RangePolicy<typename DeviceType::execution_space, int>(0, n);
    flare::parallel_for("scatter_tensor_test: Prod", policy, *this);
  }

  FLARE_INLINE_FUNCTION
  void operator()(int i) const {
    auto scatter_access = scatter_tensor.access();
    auto scatter_access_atomic =
        scatter_tensor.template access<flare::experimental::ScatterAtomic>();
    for (int j = 0; j < 4; ++j) {
      auto k = (i + j) % scatterSize;
      scatter_access(k, 0).update((NumberType)(j + 1) * 4);
      scatter_access_atomic(k, 1).update((NumberType)(j + 1) * 2.0);
      scatter_access(k, 2).update((NumberType)(j + 1) * 1.0);
    }
  }

  void validateResults(orig_tensor_type orig) {
    auto host_tensor =
        flare::create_mirror_tensor_and_copy(flare::HostSpace(), orig);
    flare::fence();
    for (size_type i = 0; i < host_tensor.extent(0); ++i) {
      auto val0 = host_tensor(i, 0);
      auto val1 = host_tensor(i, 1);
      auto val2 = host_tensor(i, 2);
        REQUIRE_LE(std::abs(val0 - 4.0), 1e-14 * 4.0);
        REQUIRE_LE(std::abs(val1 - 2.0), 1e-14 * 2.0);
        REQUIRE_LE(std::abs(val2 - 1.0), 1e-14 * 1.0);
    }
  }

  // check for correct padding
  void validateResultsForSubtensor(
      orig_tensor_type orig, std::pair<size_type, size_type>& subRangeDim0,
      std::pair<size_type, size_type>& subRangeDim1) {
    (void)subRangeDim1;
    auto host_tensor =
        flare::create_mirror_tensor_and_copy(flare::HostSpace(), orig);
    flare::fence();
    for (size_type i = 0; i < host_tensor.extent(0); ++i) {
      auto val0 = host_tensor(i, 0);
      auto val1 = host_tensor(i, 1);
      auto val2 = host_tensor(i, 2);
      if (i >= std::get<0>(subRangeDim0) && i < std::get<1>(subRangeDim0)) {
        // is in subtensor
          REQUIRE_LE(std::abs(val0 - 4.0), 1e-14 * 4.0);
          REQUIRE_LE(std::abs(val1 - 2.0), 1e-14 * 2.0);
          REQUIRE_LE(std::abs(val2 - 1.0), 1e-14 * 1.0);
      } else {
        // is outside of subtensor
          REQUIRE_LE(std::abs(val0 - NumberType(999999)), 1e-14);
          REQUIRE_LE(std::abs(val1 - NumberType(999999)), 1e-14);
          REQUIRE_LE(std::abs(val2 - NumberType(999999)), 1e-14);
      }
    }
  }
};

template <typename DeviceType, typename Layout, typename Duplication,
          typename Contribution, typename NumberType>
struct test_scatter_tensor_impl_cls<DeviceType, Layout, Duplication, Contribution,
                                  flare::experimental::ScatterMax,
                                  NumberType> {
 public:
  using scatter_tensor_type =
      flare::experimental::ScatterTensor<NumberType * [3], Layout, DeviceType,
                                        flare::experimental::ScatterMax,
                                        Duplication, Contribution>;

  using orig_tensor_type = flare::Tensor<NumberType * [3], Layout, DeviceType>;

  using size_type = typename flare::HostSpace::size_type;

  scatter_tensor_type scatter_tensor;
  int scatterSize;

  test_scatter_tensor_impl_cls(const scatter_tensor_type& tensor) {
    scatter_tensor = tensor;
    scatterSize  = 0;
  }

  void initialize(orig_tensor_type orig) {
    auto host_tensor =
        flare::create_mirror_tensor_and_copy(flare::HostSpace(), orig);
    flare::fence();
    for (size_type i = 0; i < host_tensor.extent(0); ++i) {
      host_tensor(i, 0) = 0.0;
      host_tensor(i, 1) = 0.0;
      host_tensor(i, 2) = 0.0;
    }
    flare::fence();
    flare::deep_copy(orig, host_tensor);
  }

  void run_parallel(int n) {
    scatterSize = n;
    flare::RangePolicy<typename DeviceType::execution_space, int> policy(0, n);
    flare::parallel_for("scatter_tensor_test: Prod", policy, *this);
  }

  FLARE_INLINE_FUNCTION
  void operator()(int i) const {
    auto scatter_access = scatter_tensor.access();
    auto scatter_access_atomic =
        scatter_tensor.template access<flare::experimental::ScatterAtomic>();
    for (int j = 0; j < 4; ++j) {
      auto k = (i + j) % scatterSize;
      scatter_access(k, 0).update((NumberType)(j + 1) * 4);
      scatter_access_atomic(k, 1).update((NumberType)(j + 1) * 2.0);
      scatter_access(k, 2).update((NumberType)(j + 1) * 1.0);
    }
  }

  void validateResults(orig_tensor_type orig) {
    auto host_tensor =
        flare::create_mirror_tensor_and_copy(flare::HostSpace(), orig);
    flare::fence();
    for (size_type i = 0; i < host_tensor.extent(0); ++i) {
      auto val0 = host_tensor(i, 0);
      auto val1 = host_tensor(i, 1);
      auto val2 = host_tensor(i, 2);
        REQUIRE_LE(std::abs(val0 - 16.0), 1e-14 * 16.0);
        REQUIRE_LE(std::abs(val1 - 8.0), 1e-14 * 8.0);
        REQUIRE_LE(std::abs(val2 - 4.0), 1e-14 * 4.0);
    }
  }

  // check for correct padding
  void validateResultsForSubtensor(
      orig_tensor_type orig, std::pair<size_type, size_type>& subRangeDim0,
      std::pair<size_type, size_type>& subRangeDim1) {
    (void)subRangeDim1;
    auto host_tensor =
        flare::create_mirror_tensor_and_copy(flare::HostSpace(), orig);
    flare::fence();
    for (size_type i = 0; i < host_tensor.extent(0); ++i) {
      auto val0 = host_tensor(i, 0);
      auto val1 = host_tensor(i, 1);
      auto val2 = host_tensor(i, 2);
      if (i >= std::get<0>(subRangeDim0) && i < std::get<1>(subRangeDim0)) {
        // is in subtensor
        REQUIRE_LE(std::abs(val0 - 16.0), 1e-14 * 16.0);
          REQUIRE_LE(std::abs(val1 - 8.0), 1e-14 * 8.0);
          REQUIRE_LE(std::abs(val2 - 4.0), 1e-14 * 4.0);
      } else {
        // is outside of subtensor
          REQUIRE_LE(std::abs(val0 - NumberType(0)), 1e-14);
          REQUIRE_LE(std::abs(val1 - NumberType(0)), 1e-14);
          REQUIRE_LE(std::abs(val2 - NumberType(0)), 1e-14);
      }
    }
  }
};

// Test ScatterTensor on subtensor
template <typename DeviceType, typename Layout, typename Op,
          typename NumberType>
struct test_default_scatter_sub_tensor {
 public:
  using default_duplication = flare::detail::experimental::DefaultDuplication<
      typename DeviceType::execution_space>;
  using Duplication  = typename default_duplication::type;
  using Contribution = typename flare::detail::experimental::DefaultContribution<
      typename DeviceType::execution_space, Duplication>::type;
  using scatter_tensor_def =
      typename test_scatter_tensor_impl_cls<DeviceType, Layout, Duplication,
                                          Contribution, Op,
                                          NumberType>::scatter_tensor_type;
  using orig_tensor_def =
      typename test_scatter_tensor_impl_cls<DeviceType, Layout, Duplication,
                                          Contribution, Op,
                                          NumberType>::orig_tensor_type;

  using size_type = typename flare::HostSpace::size_type;

  void run_test(int n) {
    // Test creation via create_scatter_tensor overload 1
    {
      orig_tensor_def original_tensor("original_tensor", n);

      auto rangeDim0 = std::pair<size_type, size_type>(0 + 1, n - 1);
      auto rangeDim1 =
          std::pair<size_type, size_type>(0, original_tensor.extent(1));

      auto original_sub_tensor =
          flare::subtensor(original_tensor, rangeDim0, rangeDim1);

      scatter_tensor_def scatter_tensor =
          flare::experimental::create_scatter_tensor(Op{}, original_sub_tensor);

      test_scatter_tensor_impl_cls<DeviceType, Layout, Duplication, Contribution,
                                 Op, NumberType>
          scatter_tensor_test_impl(scatter_tensor);
      scatter_tensor_test_impl.initialize(original_tensor);
      scatter_tensor_test_impl.run_parallel(original_sub_tensor.extent(0));

      flare::experimental::contribute(original_sub_tensor, scatter_tensor);
      scatter_tensor.reset_except(original_sub_tensor);

      scatter_tensor_test_impl.run_parallel(original_sub_tensor.extent(0));

      flare::experimental::contribute(original_sub_tensor, scatter_tensor);
      flare::fence();

      scatter_tensor_test_impl.validateResultsForSubtensor(original_tensor, rangeDim0,
                                                       rangeDim1);
    }
  }
};

template <typename DeviceType, typename Layout, typename Op,
          typename NumberType>
struct test_default_scatter_tensor {
 public:
  using default_duplication = flare::detail::experimental::DefaultDuplication<
      typename DeviceType::execution_space>;
  using Duplication  = typename default_duplication::type;
  using Contribution = typename flare::detail::experimental::DefaultContribution<
      typename DeviceType::execution_space, Duplication>::type;
  using scatter_tensor_def =
      typename test_scatter_tensor_impl_cls<DeviceType, Layout, Duplication,
                                          Contribution, Op,
                                          NumberType>::scatter_tensor_type;
  using orig_tensor_def =
      typename test_scatter_tensor_impl_cls<DeviceType, Layout, Duplication,
                                          Contribution, Op,
                                          NumberType>::orig_tensor_type;

  void run_test(int n) {
    // Test creation via create_scatter_tensor overload 1
    {
      orig_tensor_def original_tensor("original_tensor", n);
      scatter_tensor_def scatter_tensor =
          flare::experimental::create_scatter_tensor(Op{}, original_tensor);

      test_scatter_tensor_impl_cls<DeviceType, Layout, Duplication, Contribution,
                                 Op, NumberType>
          scatter_tensor_test_impl(scatter_tensor);
      scatter_tensor_test_impl.initialize(original_tensor);
      scatter_tensor_test_impl.run_parallel(n);

      flare::experimental::contribute(original_tensor, scatter_tensor);
      scatter_tensor.reset_except(original_tensor);

      scatter_tensor_test_impl.run_parallel(n);

      flare::experimental::contribute(original_tensor, scatter_tensor);
      flare::fence();

      scatter_tensor_test_impl.validateResults(original_tensor);

      {
        scatter_tensor_def persistent_tensor("persistent", n);
        auto result_tensor = persistent_tensor.subtensor();
        contribute(result_tensor, persistent_tensor);
        flare::fence();
      }
    }
  }
};

template <typename DeviceType, typename Layout, typename Duplication,
          typename Contribution, typename Op, typename NumberType>
struct test_scatter_tensor_config {
 public:
  using scatter_tensor_def =
      typename test_scatter_tensor_impl_cls<DeviceType, Layout, Duplication,
                                          Contribution, Op,
                                          NumberType>::scatter_tensor_type;
  using orig_tensor_def =
      typename test_scatter_tensor_impl_cls<DeviceType, Layout, Duplication,
                                          Contribution, Op,
                                          NumberType>::orig_tensor_type;

  void compile_constructor() {
    auto sv = scatter_tensor_def(flare::tensor_alloc(DeviceType{}, "label"), 10);
  }

  void run_test(int n) {
    // test allocation
    {
      orig_tensor_def ov1("ov1", n);
      scatter_tensor_def sv1;

      REQUIRE_FALSE(sv1.is_allocated());

      sv1 = flare::experimental::create_scatter_tensor<Op, Duplication,
                                                      Contribution>(ov1);

      scatter_tensor_def sv2(sv1);
      scatter_tensor_def sv3("sv3", n);

      REQUIRE(sv1.is_allocated());
      REQUIRE(sv2.is_allocated());
      REQUIRE(sv3.is_allocated());
    }

    // Test creation via create_scatter_tensor
    {
      orig_tensor_def original_tensor("original_tensor", n);
      scatter_tensor_def scatter_tensor = flare::experimental::create_scatter_tensor<
          Op, Duplication, Contribution>(original_tensor);

      test_scatter_tensor_impl_cls<DeviceType, Layout, Duplication, Contribution,
                                 Op, NumberType>
          scatter_tensor_test_impl(scatter_tensor);
      scatter_tensor_test_impl.initialize(original_tensor);
      scatter_tensor_test_impl.run_parallel(n);

      flare::experimental::contribute(original_tensor, scatter_tensor);
      scatter_tensor.reset_except(original_tensor);

      scatter_tensor_test_impl.run_parallel(n);

      flare::experimental::contribute(original_tensor, scatter_tensor);
      flare::fence();

      scatter_tensor_test_impl.validateResults(original_tensor);

      {
        scatter_tensor_def persistent_tensor("persistent", n);
        auto result_tensor = persistent_tensor.subtensor();
        contribute(result_tensor, persistent_tensor);
        flare::fence();
      }
    }
    // Test creation via create_scatter_tensor overload 2
    {
      orig_tensor_def original_tensor("original_tensor", n);
      scatter_tensor_def scatter_tensor = flare::experimental::create_scatter_tensor(
          Op{}, Duplication{}, Contribution{}, original_tensor);

      test_scatter_tensor_impl_cls<DeviceType, Layout, Duplication, Contribution,
                                 Op, NumberType>
          scatter_tensor_test_impl(scatter_tensor);
      scatter_tensor_test_impl.initialize(original_tensor);
      scatter_tensor_test_impl.run_parallel(n);

      flare::experimental::contribute(original_tensor, scatter_tensor);
      scatter_tensor.reset_except(original_tensor);

      scatter_tensor_test_impl.run_parallel(n);

      flare::experimental::contribute(original_tensor, scatter_tensor);
      flare::fence();

      scatter_tensor_test_impl.validateResults(original_tensor);

      {
        scatter_tensor_def persistent_tensor("persistent", n);
        auto result_tensor = persistent_tensor.subtensor();
        contribute(result_tensor, persistent_tensor);
        flare::fence();
      }
    }
    // Test creation via constructor
    {
      orig_tensor_def original_tensor("original_tensor", n);
      scatter_tensor_def scatter_tensor(original_tensor);

      test_scatter_tensor_impl_cls<DeviceType, Layout, Duplication, Contribution,
                                 Op, NumberType>
          scatter_tensor_test_impl(scatter_tensor);
      scatter_tensor_test_impl.initialize(original_tensor);
      scatter_tensor_test_impl.run_parallel(n);

      flare::experimental::contribute(original_tensor, scatter_tensor);
      scatter_tensor.reset_except(original_tensor);

      scatter_tensor_test_impl.run_parallel(n);

      flare::experimental::contribute(original_tensor, scatter_tensor);
      flare::fence();

      scatter_tensor_test_impl.validateResults(original_tensor);

      {
        scatter_tensor_def persistent_tensor("persistent", n);
        auto result_tensor = persistent_tensor.subtensor();
        contribute(result_tensor, persistent_tensor);
        flare::fence();
      }
    }
  }
};

template <typename DeviceType, typename ScatterType, typename NumberType>
struct TestDuplicatedScatterTensor {
  TestDuplicatedScatterTensor(int n) {
    // ScatterSum test
    test_scatter_tensor_config<DeviceType, flare::LayoutRight,
                             flare::experimental::ScatterDuplicated,
                             flare::experimental::ScatterNonAtomic,
                             ScatterType, NumberType>
        test_sv_right_config;
    test_sv_right_config.run_test(n);
    test_scatter_tensor_config<
        DeviceType, flare::LayoutLeft, flare::experimental::ScatterDuplicated,
        flare::experimental::ScatterNonAtomic, ScatterType, NumberType>
        test_sv_left_config;
    test_sv_left_config.run_test(n);
  }
};

#ifdef FLARE_ON_CUDA_DEVICE
// disable duplicated instantiation with CUDA until
// UniqueToken can support it
template <typename ScatterType, typename NumberType>
struct TestDuplicatedScatterTensor<flare::Cuda, ScatterType, NumberType> {
  TestDuplicatedScatterTensor(int) {}
};
template <typename ScatterType, typename NumberType>
struct TestDuplicatedScatterTensor<
    flare::Device<flare::Cuda, flare::CudaSpace>, ScatterType, NumberType> {
  TestDuplicatedScatterTensor(int) {}
};
template <typename ScatterType, typename NumberType>
struct TestDuplicatedScatterTensor<
    flare::Device<flare::Cuda, flare::CudaUVMSpace>, ScatterType,
    NumberType> {
  TestDuplicatedScatterTensor(int) {}
};
#endif

template <typename DeviceType, typename ScatterType,
          typename NumberType = double>
void test_scatter_tensor(int64_t n) {
  using execution_space = typename DeviceType::execution_space;

  // no atomics or duplication is only sensible if the execution space
  // is running essentially in serial (doesn't have to be Serial though,
  // we also test OpenMP with one thread: LAMMPS cares about that)
  if (execution_space().concurrency() == 1) {
    test_scatter_tensor_config<DeviceType, flare::LayoutRight,
                             flare::experimental::ScatterNonDuplicated,
                             flare::experimental::ScatterNonAtomic,
                             ScatterType, NumberType>
        test_sv_config;
    test_sv_config.run_test(n);
  }
#ifdef FLARE_ENABLE_SERIAL
  if (!std::is_same<DeviceType, flare::Serial>::value) {
#endif
    test_scatter_tensor_config<DeviceType, flare::LayoutRight,
                             flare::experimental::ScatterNonDuplicated,
                             flare::experimental::ScatterAtomic, ScatterType,
                             NumberType>
        test_sv_config;
    test_sv_config.run_test(n);
#ifdef FLARE_ENABLE_SERIAL
  }
#endif
  // with hundreds of threads we were running out of memory.
  // limit (n) so that duplication doesn't exceed 1GB
  constexpr std::size_t maximum_allowed_total_bytes =
      1ull * 1024ull * 1024ull * 1024ull;
  std::size_t const maximum_allowed_copy_bytes =
      maximum_allowed_total_bytes /
      std::size_t(execution_space().concurrency());
  constexpr std::size_t bytes_per_value = sizeof(NumberType) * 12;
  std::size_t const maximum_allowed_copy_values =
      maximum_allowed_copy_bytes / bytes_per_value;
  n = std::min(n, int64_t(maximum_allowed_copy_values));

  // if the default is duplicated, this needs to follow the limit
  {
    test_default_scatter_tensor<DeviceType, flare::LayoutRight, ScatterType,
                              NumberType>
        test_default_sv;
    test_default_sv.run_test(n);
  }

  // run same test but on a subtensor (this covers support for padded
  // ScatterTensors)
  {
    test_default_scatter_sub_tensor<DeviceType, flare::LayoutRight, ScatterType,
                                  NumberType>
        test_default_scatter_tensor_subtensor;
    test_default_scatter_tensor_subtensor.run_test(n);
  }

  TestDuplicatedScatterTensor<DeviceType, ScatterType, NumberType> duptest(n);
}

TEST_CASE("TEST_CATEGORY, scattertensor") {
  test_scatter_tensor<TEST_EXECSPACE, flare::experimental::ScatterSum, double>(
      10);

  test_scatter_tensor<TEST_EXECSPACE, flare::experimental::ScatterSum, int>(10);
  test_scatter_tensor<TEST_EXECSPACE, flare::experimental::ScatterProd>(10);
  test_scatter_tensor<TEST_EXECSPACE, flare::experimental::ScatterMin>(10);
  test_scatter_tensor<TEST_EXECSPACE, flare::experimental::ScatterMax>(10);
  // tests were timing out in DEBUG mode, reduce the amount of work
#ifdef FLARE_ENABLE_DEBUG
  int big_n = 100 * 1000;
#else

#if defined(FLARE_ENABLE_SERIAL) || defined(FLARE_ENABLE_OPENMP)
#if defined(FLARE_ENABLE_SERIAL)
  bool is_serial = std::is_same<TEST_EXECSPACE, flare::Serial>::value;
#else
  bool is_serial = false;
#endif
#if defined(FLARE_ENABLE_OPENMP)
  bool is_openmp = std::is_same<TEST_EXECSPACE, flare::OpenMP>::value;
#else
  bool is_openmp = false;
#endif
  int big_n      = is_serial || is_openmp ? 100 * 1000 : 10000 * 1000;
#else
  int big_n = 10000 * 1000;
#endif

#endif

  test_scatter_tensor<TEST_EXECSPACE, flare::experimental::ScatterSum, double>(
      big_n);
  test_scatter_tensor<TEST_EXECSPACE, flare::experimental::ScatterSum, int>(
      big_n);
  test_scatter_tensor<TEST_EXECSPACE, flare::experimental::ScatterProd>(big_n);
  test_scatter_tensor<TEST_EXECSPACE, flare::experimental::ScatterMin>(big_n);
  test_scatter_tensor<TEST_EXECSPACE, flare::experimental::ScatterMax>(big_n);
}

TEST_CASE("TEST_CATEGORY, scattertensor_devicetype") {
  using device_type =
      flare::Device<TEST_EXECSPACE, typename TEST_EXECSPACE::memory_space>;

  test_scatter_tensor<device_type, flare::experimental::ScatterSum, double>(10);
  test_scatter_tensor<device_type, flare::experimental::ScatterSum, int>(10);
  test_scatter_tensor<device_type, flare::experimental::ScatterProd>(10);
  test_scatter_tensor<device_type, flare::experimental::ScatterMin>(10);
  test_scatter_tensor<device_type, flare::experimental::ScatterMax>(10);

#if defined(FLARE_ON_CUDA_DEVICE)
  using device_execution_space = flare::Cuda;
  using device_memory_space    = flare::CudaSpace;
  using host_accessible_space  = flare::CudaUVMSpace;
  if (std::is_same<TEST_EXECSPACE, device_execution_space>::value) {
    using device_device_type =
        flare::Device<device_execution_space, device_memory_space>;
    test_scatter_tensor<device_device_type, flare::experimental::ScatterSum,
                      double>(10);
    test_scatter_tensor<device_device_type, flare::experimental::ScatterSum,
                      int>(10);
    test_scatter_tensor<device_device_type, flare::experimental::ScatterProd>(
        10);
    test_scatter_tensor<device_device_type, flare::experimental::ScatterMin>(10);
    test_scatter_tensor<device_device_type, flare::experimental::ScatterMax>(10);
    using host_device_type =
        flare::Device<device_execution_space, host_accessible_space>;
    test_scatter_tensor<host_device_type, flare::experimental::ScatterSum,
                      double>(10);
    test_scatter_tensor<host_device_type, flare::experimental::ScatterSum, int>(
        10);
    test_scatter_tensor<host_device_type, flare::experimental::ScatterProd>(10);
    test_scatter_tensor<host_device_type, flare::experimental::ScatterMin>(10);
    test_scatter_tensor<host_device_type, flare::experimental::ScatterMax>(10);
  }
#endif
}

}  // namespace Test

#endif  // FLARE_test_scatter_tensor_HPP
