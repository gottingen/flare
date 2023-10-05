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

#ifndef SIMD_SHIFT_OPS_H_
#define SIMD_SHIFT_OPS_H_

#include <flare/simd/simd.h>
#include <simd_utilities.h>

template <typename Abi, typename Loader, typename ShiftOp, typename DataType>
inline void host_check_shift_on_one_loader(ShiftOp shift_op,
                                           DataType test_vals[],
                                           DataType shift_by[], std::size_t n) {
  using simd_type             = flare::experimental::simd<DataType, Abi>;
  constexpr std::size_t width = simd_type::size();
  Loader loader;

  for (std::size_t i = 0; i < n; ++i) {
    simd_type simd_vals;
    bool const loaded_arg = loader.host_load(test_vals, width, simd_vals);
    if (!loaded_arg) {
      continue;
    }

    simd_type expected_result;

    for (std::size_t lane = 0; lane < width; ++lane) {
      DataType value = simd_vals[lane];
      expected_result[lane] =
          shift_op.on_host(value, static_cast<int>(shift_by[i]));
        REQUIRE_EQ(value, value);
    }

    simd_type const computed_result =
        shift_op.on_host(simd_vals, static_cast<int>(shift_by[i]));
    host_check_equality(expected_result, computed_result, width);
  }
}

template <typename Abi, typename Loader, typename ShiftOp, typename DataType>
inline void host_check_shift_by_lanes_on_one_loader(
    ShiftOp shift_op, DataType test_vals[],
    flare::experimental::simd<DataType, Abi>& shift_by) {
  using simd_type             = flare::experimental::simd<DataType, Abi>;
  constexpr std::size_t width = simd_type::size();
  Loader loader;

  simd_type simd_vals;
  bool const loaded_arg = loader.host_load(test_vals, width, simd_vals);
  REQUIRE(loaded_arg);

  simd_type expected_result;

  for (std::size_t lane = 0; lane < width; ++lane) {
    DataType value = simd_vals[lane];
    expected_result[lane] =
        shift_op.on_host(value, static_cast<int>(shift_by[lane]));
    REQUIRE_EQ(value, value);
  }
  simd_type const computed_result = shift_op.on_host(simd_vals, shift_by);
  host_check_equality(expected_result, computed_result, width);
}

template <typename Abi, typename ShiftOp, typename DataType>
inline void host_check_shift_op_all_loaders(ShiftOp shift_op,
                                            DataType test_vals[],
                                            DataType shift_by[],
                                            std::size_t n) {
  host_check_shift_on_one_loader<Abi, load_element_aligned>(shift_op, test_vals,
                                                            shift_by, n);
  host_check_shift_on_one_loader<Abi, load_masked>(shift_op, test_vals,
                                                   shift_by, n);
  host_check_shift_on_one_loader<Abi, load_as_scalars>(shift_op, test_vals,
                                                       shift_by, n);

  flare::experimental::simd<DataType, Abi> shift_by_lanes;
  shift_by_lanes.copy_from(shift_by,
                           flare::experimental::element_aligned_tag());

  host_check_shift_by_lanes_on_one_loader<Abi, load_element_aligned>(
      shift_op, test_vals, shift_by_lanes);
  host_check_shift_by_lanes_on_one_loader<Abi, load_masked>(shift_op, test_vals,
                                                            shift_by_lanes);
  host_check_shift_by_lanes_on_one_loader<Abi, load_as_scalars>(
      shift_op, test_vals, shift_by_lanes);
}

template <typename Abi, typename DataType>
inline void host_check_shift_ops() {
  if constexpr (std::is_integral_v<DataType>) {
    using simd_type                 = flare::experimental::simd<DataType, Abi>;
    constexpr std::size_t width     = simd_type::size();
    constexpr std::size_t num_cases = 8;

    DataType max = std::numeric_limits<DataType>::max();

    DataType shift_by[num_cases] = {
        0, 1, 3, width / 2, width / 2 + 1, width - 1, width, width + 1};
    DataType test_vals[width];
    for (std::size_t i = 0; i < width; ++i) {
      DataType inc = max / width;
      test_vals[i] = i * inc + 1;
    }

    host_check_shift_op_all_loaders<Abi>(shift_right(), test_vals, shift_by,
                                         num_cases);
    host_check_shift_op_all_loaders<Abi>(shift_left(), test_vals, shift_by,
                                         num_cases);

    if constexpr (std::is_signed_v<DataType>) {
      for (std::size_t i = 0; i < width; ++i) test_vals[i] *= -1;
      host_check_shift_op_all_loaders<Abi>(shift_right(), test_vals, shift_by,
                                           num_cases);
      host_check_shift_op_all_loaders<Abi>(shift_left(), test_vals, shift_by,
                                           num_cases);
    }
  }
}

template <typename Abi, typename... DataTypes>
inline void host_check_shift_ops_all_types(
    flare::experimental::detail::data_types<DataTypes...>) {
  (host_check_shift_ops<Abi, DataTypes>(), ...);
}

template <typename... Abis>
inline void host_check_shift_ops_all_abis(
    flare::experimental::detail::abi_set<Abis...>) {
  using DataTypes = flare::experimental::detail::data_type_set;
  (host_check_shift_ops_all_types<Abis>(DataTypes()), ...);
}

template <typename Abi, typename Loader, typename ShiftOp, typename DataType>
FLARE_INLINE_FUNCTION void device_check_shift_on_one_loader(
    ShiftOp shift_op, DataType test_vals[], DataType shift_by[],
    std::size_t n) {
  using simd_type             = flare::experimental::simd<DataType, Abi>;
  constexpr std::size_t width = simd_type::size();
  Loader loader;

  for (std::size_t i = 0; i < n; ++i) {
    simd_type simd_vals;
    bool const loaded_arg = loader.device_load(test_vals, width, simd_vals);
    if (!loaded_arg) {
      continue;
    }

    simd_type expected_result;

    for (std::size_t lane = 0; lane < width; ++lane) {
      expected_result[lane] = shift_op.on_device(DataType(simd_vals[lane]),
                                                 static_cast<int>(shift_by[i]));
    }

    simd_type const computed_result =
        shift_op.on_device(simd_vals, static_cast<int>(shift_by[i]));
    device_check_equality(expected_result, computed_result, width);
  }
}

template <typename Abi, typename Loader, typename ShiftOp, typename DataType>
FLARE_INLINE_FUNCTION void device_check_shift_by_lanes_on_one_loader(
    ShiftOp shift_op, DataType test_vals[],
    flare::experimental::simd<DataType, Abi>& shift_by) {
  using simd_type             = flare::experimental::simd<DataType, Abi>;
  constexpr std::size_t width = simd_type::size();
  Loader loader;
  simd_type simd_vals;
  loader.device_load(test_vals, width, simd_vals);

  simd_type expected_result;

  for (std::size_t lane = 0; lane < width; ++lane) {
    expected_result[lane] = shift_op.on_device(
        DataType(simd_vals[lane]), static_cast<int>(shift_by[lane]));
  }
  simd_type const computed_result = shift_op.on_device(simd_vals, shift_by);
  device_check_equality(expected_result, computed_result, width);
}

template <typename Abi, typename ShiftOp, typename DataType>
FLARE_INLINE_FUNCTION void device_check_shift_op_all_loaders(
    ShiftOp shift_op, DataType test_vals[], DataType shift_by[],
    std::size_t n) {
  device_check_shift_on_one_loader<Abi, load_element_aligned>(
      shift_op, test_vals, shift_by, n);
  device_check_shift_on_one_loader<Abi, load_masked>(shift_op, test_vals,
                                                     shift_by, n);
  device_check_shift_on_one_loader<Abi, load_as_scalars>(shift_op, test_vals,
                                                         shift_by, n);

  flare::experimental::simd<DataType, Abi> shift_by_lanes;
  shift_by_lanes.copy_from(shift_by,
                           flare::experimental::element_aligned_tag());

  device_check_shift_by_lanes_on_one_loader<Abi, load_element_aligned>(
      shift_op, test_vals, shift_by_lanes);
  device_check_shift_by_lanes_on_one_loader<Abi, load_masked>(
      shift_op, test_vals, shift_by_lanes);
  device_check_shift_by_lanes_on_one_loader<Abi, load_as_scalars>(
      shift_op, test_vals, shift_by_lanes);
}

template <typename Abi, typename DataType>
FLARE_INLINE_FUNCTION void device_check_shift_ops() {
  if constexpr (std::is_integral_v<DataType>) {
    using simd_type                 = flare::experimental::simd<DataType, Abi>;
    constexpr std::size_t width     = simd_type::size();
    constexpr std::size_t num_cases = 8;

    DataType max = flare::reduction_identity<DataType>::max();

    DataType shift_by[num_cases] = {
        0, 1, 3, width / 2, width / 2 + 1, width - 1, width, width + 1};
    DataType test_vals[width];

    for (std::size_t i = 0; i < width; ++i) {
      DataType inc = max / width;
      test_vals[i] = i * inc + 1;
    }

    device_check_shift_op_all_loaders<Abi>(shift_right(), test_vals, shift_by,
                                           num_cases);
    device_check_shift_op_all_loaders<Abi>(shift_left(), test_vals, shift_by,
                                           num_cases);

    if constexpr (std::is_signed_v<DataType>) {
      for (std::size_t i = 0; i < width; ++i) test_vals[i] *= -1;
      device_check_shift_op_all_loaders<Abi>(shift_right(), test_vals, shift_by,
                                             num_cases);
      device_check_shift_op_all_loaders<Abi>(shift_left(), test_vals, shift_by,
                                             num_cases);
    }
  }
}

template <typename Abi, typename... DataTypes>
FLARE_INLINE_FUNCTION void device_check_shift_ops_all_types(
    flare::experimental::detail::data_types<DataTypes...>) {
  (device_check_shift_ops<Abi, DataTypes>(), ...);
}

template <typename... Abis>
FLARE_INLINE_FUNCTION void device_check_shift_ops_all_abis(
    flare::experimental::detail::abi_set<Abis...>) {
  using DataTypes = flare::experimental::detail::data_type_set;
  (device_check_shift_ops_all_types<Abis>(DataTypes()), ...);
}

class simd_device_shift_ops_functor {
 public:
  FLARE_INLINE_FUNCTION void operator()(int) const {
    device_check_shift_ops_all_abis(
        flare::experimental::detail::device_abi_set());
  }
};

TEST_CASE("simd, host_shift_ops") {
  host_check_shift_ops_all_abis(flare::experimental::detail::host_abi_set());
}

TEST_CASE("simd, device_shift_ops") {
  flare::parallel_for(flare::RangePolicy<flare::IndexType<int>>(0, 1),
                       simd_device_shift_ops_functor());
}

#endif  // SIMD_SHIFT_OPS_H_
