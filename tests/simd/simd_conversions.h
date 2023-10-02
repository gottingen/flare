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

#ifndef SIMD_CONVERSIONS_H_
#define SIMD_CONVERSIONS_H_

#include <flare/simd/simd.h>
#include <simd_utilities.h>

template<typename Abi>
inline void host_check_conversions() {
    {
        auto a = flare::experimental::simd<std::uint64_t, Abi>(1);
        auto b = flare::experimental::simd<std::int64_t, Abi>(a);
        REQUIRE(all_of(b == decltype(b)(1)));
    }
    {
        auto a = flare::experimental::simd<std::int32_t, Abi>(1);
        auto b = flare::experimental::simd<std::uint64_t, Abi>(a);
        REQUIRE(all_of(b == decltype(b)(1)));
    }
    {
        auto a = flare::experimental::simd<std::uint64_t, Abi>(1);
        auto b = flare::experimental::simd<std::int32_t, Abi>(a);
        REQUIRE(all_of(b == decltype(b)(1)));
    }
    {
        auto a = flare::experimental::simd_mask<double, Abi>(true);
        auto b = flare::experimental::simd_mask<std::int32_t, Abi>(a);
        REQUIRE(b == decltype(b)(true));
    }
    {
        auto a = flare::experimental::simd_mask<std::int32_t, Abi>(true);
        auto b = flare::experimental::simd_mask<std::uint64_t, Abi>(a);
        REQUIRE(b == decltype(b)(true));
    }
    {
        auto a = flare::experimental::simd_mask<std::int32_t, Abi>(true);
        auto b = flare::experimental::simd_mask<std::int64_t, Abi>(a);
        REQUIRE(b == decltype(b)(true));
    }
    {
        auto a = flare::experimental::simd_mask<std::int32_t, Abi>(true);
        auto b = flare::experimental::simd_mask<double, Abi>(a);
        REQUIRE(b == decltype(b)(true));
    }
}

template<typename... Abis>
inline void host_check_conversions_all_abis(
        flare::experimental::detail::abi_set<Abis...>) {
    (host_check_conversions<Abis>(), ...);
}

template<typename Abi>
FLARE_INLINE_FUNCTION void device_check_conversions() {
    flare_checker checker;
    {
        auto a = flare::experimental::simd<std::uint64_t, Abi>(1);
        auto b = flare::experimental::simd<std::int64_t, Abi>(a);
        checker.truth(all_of(b == decltype(b)(1)));
    }
    {
        auto a = flare::experimental::simd<std::int32_t, Abi>(1);
        auto b = flare::experimental::simd<std::uint64_t, Abi>(a);
        checker.truth(all_of(b == decltype(b)(1)));
    }
    {
        auto a = flare::experimental::simd<std::uint64_t, Abi>(1);
        auto b = flare::experimental::simd<std::int32_t, Abi>(a);
        checker.truth(all_of(b == decltype(b)(1)));
    }
    {
        auto a = flare::experimental::simd_mask<double, Abi>(true);
        auto b = flare::experimental::simd_mask<std::int32_t, Abi>(a);
        checker.truth(b == decltype(b)(true));
    }
    {
        auto a = flare::experimental::simd_mask<std::int32_t, Abi>(true);
        auto b = flare::experimental::simd_mask<std::uint64_t, Abi>(a);
        checker.truth(b == decltype(b)(true));
    }
    {
        auto a = flare::experimental::simd_mask<std::int32_t, Abi>(true);
        auto b = flare::experimental::simd_mask<std::int64_t, Abi>(a);
        checker.truth(b == decltype(b)(true));
    }
    {
        auto a = flare::experimental::simd_mask<std::int32_t, Abi>(true);
        auto b = flare::experimental::simd_mask<double, Abi>(a);
        checker.truth(b == decltype(b)(true));
    }
}

template<typename... Abis>
FLARE_INLINE_FUNCTION void device_check_conversions_all_abis(
        flare::experimental::detail::abi_set<Abis...>) {
    (device_check_conversions<Abis>(), ...);
}

class simd_device_conversions_functor {
public:
    FLARE_INLINE_FUNCTION void operator()(int) const {
        device_check_conversions_all_abis(
                flare::experimental::detail::device_abi_set());
    }
};

TEST_CASE("simd, host_conversions") {
    host_check_conversions_all_abis(flare::experimental::detail::host_abi_set());
}

TEST_CASE("simd, device_conversions") {
    flare::parallel_for(flare::RangePolicy<flare::IndexType<int>>(0, 1),
                        simd_device_conversions_functor());
}

#endif  // SIMD_CONVERSIONS_H_
