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

#ifndef SIMD_GENERATOR_CTORS_H_
#define SIMD_GENERATOR_CTORS_H_

#include <flare/simd/simd.h>
#include <simd_utilities.h>

template<typename Abi, typename DataType>
inline void host_check_gen_ctor() {
    using simd_type = flare::experimental::simd<DataType, Abi>;
    using mask_type = typename simd_type::mask_type;
    constexpr std::size_t lanes = simd_type::size();

    DataType init[lanes];
    DataType expected[lanes];
    mask_type init_mask(false);

    for (std::size_t i = 0; i < lanes; ++i) {
        if (i % 3 == 0) init_mask[i] = true;
        init[i] = 7;
        expected[i] = (init_mask[i]) ? init[i] * 9 : init[i];
    }

    simd_type basic(FLARE_LAMBDA(std::size_t i) { return init[i]; });
    mask_type mask(FLARE_LAMBDA(std::size_t i) { return init_mask[i]; });

    simd_type rhs;
    rhs.copy_from(init, flare::experimental::element_aligned_tag());
    host_check_equality(basic, rhs, lanes);

    simd_type lhs(FLARE_LAMBDA(std::size_t i) { return init[i] * 9; });
    simd_type result(
            FLARE_LAMBDA(std::size_t i) { return (mask[i]) ? lhs[i] : rhs[i]; });

    simd_type blend;
    blend.copy_from(expected, flare::experimental::element_aligned_tag());
    host_check_equality(blend, result, lanes);
}

template<typename Abi, typename... DataTypes>
inline void host_check_gen_ctors_all_types(
        flare::experimental::detail::data_types<DataTypes...>) {
    (host_check_gen_ctor<Abi, DataTypes>(), ...);
}

template<typename... Abis>
inline void host_check_gen_ctors_all_abis(
        flare::experimental::detail::abi_set<Abis...>) {
    using DataTypes = flare::experimental::detail::data_type_set;
    (host_check_gen_ctors_all_types<Abis>(DataTypes()), ...);
}

template<typename Abi, typename DataType>
FLARE_INLINE_FUNCTION void device_check_gen_ctor() {
    using simd_type = flare::experimental::simd<DataType, Abi>;
    using mask_type = typename simd_type::mask_type;
    constexpr std::size_t lanes = simd_type::size();

    DataType init[lanes];
    DataType expected[lanes];
    mask_type mask(false);

    for (std::size_t i = 0; i < lanes; ++i) {
        if (i % 3 == 0) mask[i] = true;
        init[i] = 7;
        expected[i] = (mask[i]) ? init[i] * 9 : init[i];
    }

    simd_type basic(FLARE_LAMBDA(std::size_t i) { return init[i]; });
    simd_type rhs;
    rhs.copy_from(init, flare::experimental::element_aligned_tag());
    device_check_equality(basic, rhs, lanes);

    simd_type lhs(FLARE_LAMBDA(std::size_t i) { return init[i] * 9; });
    simd_type result(
            FLARE_LAMBDA(std::size_t i) { return (mask[i]) ? lhs[i] : rhs[i]; });

    simd_type blend;
    blend.copy_from(expected, flare::experimental::element_aligned_tag());
    device_check_equality(result, blend, lanes);
}

template<typename Abi, typename... DataTypes>
FLARE_INLINE_FUNCTION void device_check_gen_ctors_all_types(
        flare::experimental::detail::data_types<DataTypes...>) {
    (device_check_gen_ctor<Abi, DataTypes>(), ...);
}

template<typename... Abis>
FLARE_INLINE_FUNCTION void device_check_gen_ctors_all_abis(
        flare::experimental::detail::abi_set<Abis...>) {
    using DataTypes = flare::experimental::detail::data_type_set;
    (device_check_gen_ctors_all_types<Abis>(DataTypes()), ...);
}

class simd_device_gen_ctor_functor {
public:
    FLARE_INLINE_FUNCTION void operator()(int) const {
        device_check_gen_ctors_all_abis(
                flare::experimental::detail::device_abi_set());
    }
};

TEST_CASE("simd, host_gen_ctors") {
    host_check_gen_ctors_all_abis(flare::experimental::detail::host_abi_set());
}

TEST_CASE("simd, device_gen_ctors") {
    flare::parallel_for(1, simd_device_gen_ctor_functor());
}

#endif  // SIMD_GENERATOR_CTORS_H_
