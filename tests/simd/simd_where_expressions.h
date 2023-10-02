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

#ifndef SIMD_WHERE_EXPRESSIONS_H_
#define SIMD_WHERE_EXPRESSIONS_H_

#include <flare/simd/simd.h>
#include <simd_utilities.h>

template<typename Abi, typename DataType>
inline void host_check_where_expr_scatter_to() {
    using simd_type = flare::experimental::simd<DataType, Abi>;
    using index_type = flare::experimental::simd<std::int32_t, Abi>;
    using mask_type = typename simd_type::mask_type;

    std::size_t nlanes = simd_type::size();
    DataType init[] = {11, 13, 17, 19, 23, 29, 31, 37};
    simd_type src;
    src.copy_from(init, flare::experimental::element_aligned_tag());

    for (std::size_t idx = 0; idx < nlanes; ++idx) {
        mask_type mask(true);
        mask[idx] = false;

        DataType dst[8] = {0};
        index_type index;
        simd_type expected_result;
        for (std::size_t i = 0; i < nlanes; ++i) {
            dst[i] = (2 + (i * 2));
            index[i] = i;
            expected_result[i] = (mask[i]) ? src[index[i]] : dst[i];
        }
        where(mask, src).scatter_to(dst, index);

        simd_type dst_simd;
        dst_simd.copy_from(dst, flare::experimental::element_aligned_tag());

        host_check_equality(expected_result, dst_simd, nlanes);
    }
}

template<typename Abi, typename DataType>
inline void host_check_where_expr_gather_from() {
    using simd_type = flare::experimental::simd<DataType, Abi>;
    using index_type = flare::experimental::simd<std::int32_t, Abi>;
    using mask_type = typename simd_type::mask_type;

    std::size_t nlanes = simd_type::size();
    DataType src[] = {11, 13, 17, 19, 23, 29, 31, 37};

    for (std::size_t idx = 0; idx < nlanes; ++idx) {
        mask_type mask(true);
        mask[idx] = false;

        simd_type dst;
        index_type index;
        simd_type expected_result;
        for (std::size_t i = 0; i < nlanes; ++i) {
            dst[i] = (2 + (i * 2));
            index[i] = i;
            expected_result[i] = (mask[i]) ? src[index[i]] : dst[i];
        }
        where(mask, dst).gather_from(src, index);

        host_check_equality(expected_result, dst, nlanes);
    }
}

template<class Abi, typename DataType>
inline void host_check_where_expr() {
    host_check_where_expr_scatter_to<Abi, DataType>();
    host_check_where_expr_gather_from<Abi, DataType>();
}

template<typename Abi, typename... DataTypes>
inline void host_check_where_expr_all_types(
        flare::experimental::detail::data_types<DataTypes...>) {
    (host_check_where_expr<Abi, DataTypes>(), ...);
}

template<typename... Abis>
inline void host_check_where_expr_all_abis(
        flare::experimental::detail::abi_set<Abis...>) {
    using DataTypes = flare::experimental::detail::data_type_set;
    (host_check_where_expr_all_types<Abis>(DataTypes()), ...);
}

template<typename Abi, typename DataType>
FLARE_INLINE_FUNCTION void device_check_where_expr_scatter_to() {
    using simd_type = flare::experimental::simd<DataType, Abi>;
    using index_type = flare::experimental::simd<std::int32_t, Abi>;
    using mask_type = typename simd_type::mask_type;

    std::size_t nlanes = simd_type::size();
    DataType init[] = {11, 13, 17, 19, 23, 29, 31, 37};
    simd_type src;
    src.copy_from(init, flare::experimental::element_aligned_tag());

    for (std::size_t idx = 0; idx < nlanes; ++idx) {
        mask_type mask(true);
        mask[idx] = false;

        DataType dst[8] = {0};
        index_type index;
        simd_type expected_result;
        for (std::size_t i = 0; i < nlanes; ++i) {
            dst[i] = (2 + (i * 2));
            index[i] = i;
            expected_result[i] = (mask[i]) ? src[index[i]] : dst[i];
        }
        where(mask, src).scatter_to(dst, index);

        simd_type dst_simd;
        dst_simd.copy_from(dst, flare::experimental::element_aligned_tag());

        device_check_equality(expected_result, dst_simd, nlanes);
    }
}

template<typename Abi, typename DataType>
FLARE_INLINE_FUNCTION void device_check_where_expr_gather_from() {
    using simd_type = flare::experimental::simd<DataType, Abi>;
    using index_type = flare::experimental::simd<std::int32_t, Abi>;
    using mask_type = typename simd_type::mask_type;

    std::size_t nlanes = simd_type::size();
    DataType src[] = {11, 13, 17, 19, 23, 29, 31, 37};

    for (std::size_t idx = 0; idx < nlanes; ++idx) {
        mask_type mask(true);
        mask[idx] = false;

        simd_type dst;
        index_type index;
        simd_type expected_result;
        for (std::size_t i = 0; i < nlanes; ++i) {
            dst[i] = (2 + (i * 2));
            index[i] = i;
            expected_result[i] = (mask[i]) ? src[index[i]] : dst[i];
        }
        where(mask, dst).gather_from(src, index);

        device_check_equality(expected_result, dst, nlanes);
    }
}

template<class Abi, typename DataType>
FLARE_INLINE_FUNCTION void device_check_where_expr() {
    device_check_where_expr_scatter_to<Abi, DataType>();
    device_check_where_expr_gather_from<Abi, DataType>();
}

template<typename Abi, typename... DataTypes>
FLARE_INLINE_FUNCTION void device_check_where_expr_all_types(
        flare::experimental::detail::data_types<DataTypes...>) {
    (device_check_where_expr<Abi, DataTypes>(), ...);
}

template<typename... Abis>
FLARE_INLINE_FUNCTION void device_check_where_expr_all_abis(
        flare::experimental::detail::abi_set<Abis...>) {
    using DataTypes = flare::experimental::detail::data_type_set;
    (device_check_where_expr_all_types<Abis>(DataTypes()), ...);
}

class simd_device_where_expr_functor {
public:
    FLARE_INLINE_FUNCTION void operator()(int) const {
        device_check_where_expr_all_abis(
                flare::experimental::detail::device_abi_set());
    }
};

TEST_CASE("simd, host_where_expressions") {
    host_check_where_expr_all_abis(flare::experimental::detail::host_abi_set());

}

TEST_CASE("simd, device_where_expressions") {
    flare::parallel_for(1,simd_device_where_expr_functor());
}

#endif  // SIMD_WHERE_EXPRESSIONS_H_
