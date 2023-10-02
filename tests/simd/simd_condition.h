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

#ifndef SIMD_CONDITION_H_
#define SIMD_CONDITION_H_

#include <flare/simd/simd.h>
#include <simd_utilities.h>

template<typename Abi, typename DataType>
inline void host_check_condition() {
    using simd_type = typename flare::experimental::simd<DataType, Abi>;
    using mask_type = typename simd_type::mask_type;

    auto condition_op = [](mask_type const &mask, simd_type const &a,
                           simd_type const &b) {
        return flare::experimental::condition(mask, a, b);
    };

    simd_type value_a(16);
    simd_type value_b(20);

    auto condition_result = condition_op(mask_type(false), value_a, value_b);
    REQUIRE(all_of(condition_result == value_b));
    condition_result = condition_op(mask_type(true), value_a, value_b);
    REQUIRE(all_of(condition_result == value_a));
}

template<typename Abi, typename... DataTypes>
inline void host_check_condition_all_types(
        flare::experimental::detail::data_types<DataTypes...>) {
    (host_check_condition<Abi, DataTypes>(), ...);
}

template<typename... Abis>
inline void host_check_condition_all_abis(
        flare::experimental::detail::abi_set<Abis...>) {
    using DataTypes = flare::experimental::detail::data_type_set;
    (host_check_condition_all_types<Abis>(DataTypes()), ...);
}

template<typename Abi, typename DataType>
FLARE_INLINE_FUNCTION void device_check_condition() {
    using simd_type = typename flare::experimental::simd<DataType, Abi>;
    using mask_type = typename simd_type::mask_type;
    flare_checker checker;

    auto condition_op = [](mask_type const &mask, simd_type const &a,
                           simd_type const &b) {
        return flare::experimental::condition(mask, a, b);
    };

    simd_type value_a(16);
    simd_type value_b(20);

    auto condition_result = condition_op(mask_type(false), value_a, value_b);
    checker.truth(all_of(condition_result == value_b));
    condition_result = condition_op(mask_type(true), value_a, value_b);
    checker.truth(all_of(condition_result == value_a));
}

template<typename Abi, typename... DataTypes>
FLARE_INLINE_FUNCTION void device_check_condition_all_types(
        flare::experimental::detail::data_types<DataTypes...>) {
    (device_check_condition<Abi, DataTypes>(), ...);
}

template<typename... Abis>
FLARE_INLINE_FUNCTION void device_check_condition_all_abis(
        flare::experimental::detail::abi_set<Abis...>) {
    using DataTypes = flare::experimental::detail::data_type_set;
    (device_check_condition_all_types<Abis>(DataTypes()), ...);
}

class simd_device_condition_functor {
public:
    FLARE_INLINE_FUNCTION void operator()(int) const {
        device_check_condition_all_abis(
                flare::experimental::detail::device_abi_set());
    }
};

TEST_CASE("simd, host_condition") {
    host_check_condition_all_abis(flare::experimental::detail::host_abi_set());
}

TEST_CASE("simd, device_condition") {
    flare::parallel_for(flare::RangePolicy<flare::IndexType<int>>(0, 1),
                        simd_device_condition_functor());
}

#endif  // SIMD_CONDITION_H_
