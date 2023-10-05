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

#ifndef SIMD_MASK_OPS_H_
#define SIMD_MASK_OPS_H_

#include <flare/simd/simd.h>
#include <simd_utilities.h>

template<typename Abi, typename DataType>
inline void host_check_mask_ops() {
    using mask_type = flare::experimental::simd_mask<DataType, Abi>;

    REQUIRE_FALSE(none_of(mask_type(true)));
    REQUIRE(none_of(mask_type(false)));
    REQUIRE(all_of(mask_type(true)));
    REQUIRE_FALSE(all_of(mask_type(false)));
    REQUIRE(any_of(mask_type(true)));
    REQUIRE_FALSE(any_of(mask_type(false)));

    for (std::size_t i = 0; i < mask_type::size(); ++i) {
        mask_type test_mask(FLARE_LAMBDA(std::size_t j) { return i == j; });

        REQUIRE(any_of(test_mask));
        REQUIRE_FALSE(none_of(test_mask));

        if constexpr (mask_type::size() > 1) {
            REQUIRE_FALSE(all_of(test_mask));
        } else {
            REQUIRE(all_of(test_mask));
        }
    }
}

template<typename Abi, typename... DataTypes>
inline void host_check_mask_ops_all_types(
        flare::experimental::detail::data_types<DataTypes...>) {
    (host_check_mask_ops<Abi, DataTypes>(), ...);
}

template<typename... Abis>
inline void host_check_mask_ops_all_abis(
        flare::experimental::detail::abi_set<Abis...>) {
    using DataTypes = flare::experimental::detail::data_type_set;
    (host_check_mask_ops_all_types<Abis>(DataTypes()), ...);
}

template<typename Abi, typename DataType>
FLARE_INLINE_FUNCTION void device_check_mask_ops() {
    using mask_type = flare::experimental::simd_mask<DataType, Abi>;
    flare_checker checker;
    checker.truth(!none_of(mask_type(true)));
    checker.truth(none_of(mask_type(false)));
    checker.truth(all_of(mask_type(true)));
    checker.truth(!all_of(mask_type(false)));
    checker.truth(any_of(mask_type(true)));
    checker.truth(!any_of(mask_type(false)));

    for (std::size_t i = 0; i < mask_type::size(); ++i) {
        mask_type test_mask(FLARE_LAMBDA(std::size_t j) { return i == j; });

        checker.truth(any_of(test_mask));
        checker.truth(!none_of(test_mask));

        if constexpr (mask_type::size() > 1) {
            checker.truth(!all_of(test_mask));
        } else {
            checker.truth(all_of(test_mask));
        }
    }
}

template<typename Abi, typename... DataTypes>
FLARE_INLINE_FUNCTION void device_check_mask_ops_all_types(
        flare::experimental::detail::data_types<DataTypes...>) {
    (device_check_mask_ops<Abi, DataTypes>(), ...);
}

template<typename... Abis>
FLARE_INLINE_FUNCTION void device_check_mask_ops_all_abis(
        flare::experimental::detail::abi_set<Abis...>) {
    using DataTypes = flare::experimental::detail::data_type_set;
    (device_check_mask_ops_all_types<Abis>(DataTypes()), ...);
}

class simd_device_mask_ops_functor {
public:
    FLARE_INLINE_FUNCTION void operator()(int) const {
        device_check_mask_ops_all_abis(
                flare::experimental::detail::device_abi_set());
    }
};

TEST_CASE("simd, host_mask_ops") {
    host_check_mask_ops_all_abis(flare::experimental::detail::host_abi_set());
}

TEST_CASE("simd, device_mask_ops") {
    flare::parallel_for(flare::RangePolicy<flare::IndexType<int>>(0, 1),
                        simd_device_mask_ops_functor());
}

#endif  // SIMD_MASK_OPS_H_
