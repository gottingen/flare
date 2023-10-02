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

#ifndef SIMD_MATH_OPS_H_
#define SIMD_MATH_OPS_H_

#include <flare/simd/simd.h>
#include <simd_utilities.h>

template<class Abi, class Loader, class BinaryOp, class T>
void host_check_math_op_one_loader(BinaryOp binary_op, std::size_t n,
                                   T const *first_args, T const *second_args) {
    Loader loader;
    using simd_type = flare::experimental::simd<T, Abi>;
    constexpr std::size_t width = simd_type::size();
    for (std::size_t i = 0; i < n; i += width) {
        std::size_t const nremaining = n - i;
        std::size_t const nlanes = flare::min(nremaining, width);
        simd_type first_arg;
        bool const loaded_first_arg =
                loader.host_load(first_args + i, nlanes, first_arg);
        simd_type second_arg;
        bool const loaded_second_arg =
                loader.host_load(second_args + i, nlanes, second_arg);
        if (!(loaded_first_arg && loaded_second_arg)) continue;
        simd_type expected_result;
        // gcc 8.4.0 warns if using nlanes as upper bound about first_arg and/or
        // second_arg being uninitialized
        for (std::size_t lane = 0; lane < simd_type::size(); ++lane) {
            if (lane < nlanes)
                expected_result[lane] =
                        binary_op.on_host(T(first_arg[lane]), T(second_arg[lane]));
        }
        simd_type const computed_result = binary_op.on_host(first_arg, second_arg);
        host_check_equality(expected_result, computed_result, nlanes);
    }
}

template<class Abi, class Loader, class UnaryOp, class T>
void host_check_math_op_one_loader(UnaryOp unary_op, std::size_t n,
                                   T const *args) {
    Loader loader;
    using simd_type = flare::experimental::simd<T, Abi>;
    constexpr std::size_t width = simd_type::size();
    for (std::size_t i = 0; i < n; i += width) {
        std::size_t const nremaining = n - i;
        std::size_t const nlanes = flare::min(nremaining, width);
        simd_type arg;
        bool const loaded_arg = loader.host_load(args + i, nlanes, arg);
        if (!loaded_arg) continue;
        simd_type expected_result;
        for (std::size_t lane = 0; lane < simd_type::size(); ++lane) {
            if (lane < nlanes)
                expected_result[lane] = unary_op.on_host_serial(T(arg[lane]));
        }
        simd_type const computed_result = unary_op.on_host(arg);
        host_check_equality(expected_result, computed_result, nlanes);
    }
}

template<class Abi, class Op, class... T>
inline void host_check_math_op_all_loaders(Op op, std::size_t n,
                                           T const *... args) {
    host_check_math_op_one_loader<Abi, load_element_aligned>(op, n, args...);
    host_check_math_op_one_loader<Abi, load_masked>(op, n, args...);
    host_check_math_op_one_loader<Abi, load_as_scalars>(op, n, args...);
}

template<typename Abi, typename DataType, size_t n>
inline void host_check_all_math_ops(const DataType (&first_args)[n],
                                    const DataType (&second_args)[n]) {
    host_check_math_op_all_loaders<Abi>(plus(), n, first_args, second_args);
    host_check_math_op_all_loaders<Abi>(minus(), n, first_args, second_args);
    host_check_math_op_all_loaders<Abi>(multiplies(), n, first_args, second_args);

    // TODO: Place fallback division implementations for all simd integer types
    if constexpr (std::is_same_v<DataType, double>)
        host_check_math_op_all_loaders<Abi>(divides(), n, first_args, second_args);

    host_check_math_op_all_loaders<Abi>(absolutes(), n, first_args);
}

template<typename Abi, typename DataType>
inline void host_check_abi_size() {
    using simd_type = flare::experimental::simd<DataType, Abi>;
    using mask_type = typename simd_type::mask_type;
    static_assert(simd_type::size() == mask_type::size());
}

template<class Abi, typename DataType>
inline void host_check_math_ops() {
    constexpr size_t n = 11;

    host_check_abi_size<Abi, DataType>();

    if constexpr (std::is_signed_v<DataType>) {
        DataType const first_args[n] = {1, 2, -1, 10, 0, 1, -2, 10, 0, 1, -2};
        DataType const second_args[n] = {1, 2, 1, 1, 1, -3, -2, 1, 13, -3, -2};
        host_check_all_math_ops<Abi>(first_args, second_args);
    } else {
        DataType const first_args[n] = {1, 2, 1, 10, 0, 1, 2, 10, 0, 1, 2};
        DataType const second_args[n] = {1, 2, 1, 1, 1, 3, 2, 1, 13, 3, 2};
        host_check_all_math_ops<Abi>(first_args, second_args);
    }
}

template<typename Abi, typename... DataTypes>
inline void host_check_math_ops_all_types(
        flare::experimental::detail::data_types<DataTypes...>) {
    (host_check_math_ops<Abi, DataTypes>(), ...);
}

template<typename... Abis>
inline void host_check_math_ops_all_abis(
        flare::experimental::detail::abi_set<Abis...>) {
    using DataTypes = flare::experimental::detail::data_type_set;
    (host_check_math_ops_all_types<Abis>(DataTypes()), ...);
}

template<typename Abi, typename Loader, typename BinaryOp, typename T>
FLARE_INLINE_FUNCTION void device_check_math_op_one_loader(
        BinaryOp binary_op, std::size_t n, T const *first_args,
        T const *second_args) {
    Loader loader;
    using simd_type = flare::experimental::simd<T, Abi>;
    constexpr std::size_t width = simd_type::size();
    for (std::size_t i = 0; i < n; i += width) {
        std::size_t const nremaining = n - i;
        std::size_t const nlanes = flare::min(nremaining, width);
        simd_type first_arg;
        bool const loaded_first_arg =
                loader.device_load(first_args + i, nlanes, first_arg);
        simd_type second_arg;
        bool const loaded_second_arg =
                loader.device_load(second_args + i, nlanes, second_arg);
        if (!(loaded_first_arg && loaded_second_arg)) continue;
        simd_type expected_result;
        for (std::size_t lane = 0; lane < nlanes; ++lane) {
            expected_result[lane] =
                    binary_op.on_device(first_arg[lane], second_arg[lane]);
        }
        simd_type const computed_result =
                binary_op.on_device(first_arg, second_arg);
        device_check_equality(expected_result, computed_result, nlanes);
    }
}

template<typename Abi, typename Loader, typename UnaryOp, typename T>
FLARE_INLINE_FUNCTION void device_check_math_op_one_loader(UnaryOp unary_op,
                                                           std::size_t n,
                                                           T const *args) {
    Loader loader;
    using simd_type = flare::experimental::simd<T, Abi>;
    constexpr std::size_t width = simd_type::size();
    for (std::size_t i = 0; i < n; i += width) {
        std::size_t const nremaining = n - i;
        std::size_t const nlanes = flare::min(nremaining, width);
        simd_type arg;
        bool const loaded_arg = loader.device_load(args + i, nlanes, arg);
        if (!loaded_arg) continue;
        simd_type expected_result;
        for (std::size_t lane = 0; lane < nlanes; ++lane) {
            expected_result[lane] = unary_op.on_device_serial(arg[lane]);
        }
        simd_type const computed_result = unary_op.on_device(arg);
        device_check_equality(expected_result, computed_result, nlanes);
    }
}

template<typename Abi, typename Op, typename... T>
FLARE_INLINE_FUNCTION void device_check_math_op_all_loaders(Op op,
                                                            std::size_t n,
                                                            T const *... args) {
    device_check_math_op_one_loader<Abi, load_element_aligned>(op, n, args...);
    device_check_math_op_one_loader<Abi, load_masked>(op, n, args...);
    device_check_math_op_one_loader<Abi, load_as_scalars>(op, n, args...);
}

template<typename Abi, typename DataType, size_t n>
FLARE_INLINE_FUNCTION void device_check_all_math_ops(
        const DataType (&first_args)[n], const DataType (&second_args)[n]) {
    device_check_math_op_all_loaders<Abi>(plus(), n, first_args, second_args);
    device_check_math_op_all_loaders<Abi>(minus(), n, first_args, second_args);
    device_check_math_op_all_loaders<Abi>(multiplies(), n, first_args,
                                          second_args);

    if constexpr (std::is_same_v<DataType, double>)
        device_check_math_op_all_loaders<Abi>(divides(), n, first_args,
                                              second_args);

    device_check_math_op_all_loaders<Abi>(absolutes(), n, first_args);
}

template<typename Abi, typename DataType>
FLARE_INLINE_FUNCTION void device_check_abi_size() {
    using simd_type = flare::experimental::simd<DataType, Abi>;
    using mask_type = typename simd_type::mask_type;
    static_assert(simd_type::size() == mask_type::size());
}

template<typename Abi, typename DataType>
FLARE_INLINE_FUNCTION void device_check_math_ops() {
    constexpr size_t n = 11;

    device_check_abi_size<Abi, DataType>();

    if constexpr (std::is_signed_v<DataType>) {
        DataType const first_args[n] = {1, 2, -1, 10, 0, 1, -2, 10, 0, 1, -2};
        DataType const second_args[n] = {1, 2, 1, 1, 1, -3, -2, 1, 13, -3, -2};
        device_check_all_math_ops<Abi>(first_args, second_args);
    } else {
        DataType const first_args[n] = {1, 2, 1, 10, 0, 1, 2, 10, 0, 1, 2};
        DataType const second_args[n] = {1, 2, 1, 1, 1, 3, 2, 1, 13, 3, 2};
        device_check_all_math_ops<Abi>(first_args, second_args);
    }
}

template<typename Abi, typename... DataTypes>
FLARE_INLINE_FUNCTION void device_check_math_ops_all_types(
        flare::experimental::detail::data_types<DataTypes...>) {
    (device_check_math_ops<Abi, DataTypes>(), ...);
}

template<typename... Abis>
FLARE_INLINE_FUNCTION void device_check_math_ops_all_abis(
        flare::experimental::detail::abi_set<Abis...>) {
    using DataTypes = flare::experimental::detail::data_type_set;
    (device_check_math_ops_all_types<Abis>(DataTypes()), ...);
}

class simd_device_math_ops_functor {
public:
    FLARE_INLINE_FUNCTION void operator()(int) const {
        device_check_math_ops_all_abis(
                flare::experimental::detail::device_abi_set());
    }
};

TEST_CASE("simd, host_math_ops") {
    host_check_math_ops_all_abis(flare::experimental::detail::host_abi_set());
}

TEST_CASE("simd, device_math_ops") {
    flare::parallel_for(flare::RangePolicy<flare::IndexType<int>>(0, 1),
                        simd_device_math_ops_functor());
}

#endif  // SIMD_MATH_OPS_H_
