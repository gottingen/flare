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

#ifndef FLARE_SIMD_TESTING_UTILITIES_HPP
#define FLARE_SIMD_TESTING_UTILITIES_HPP

#include <doctest.h>
#include <flare/simd/simd.h>
#include <simd_ops.h>

class gtest_checker {
public:
    void truth(bool x) const { REQUIRE(x); }

    template<class T>
    void equality(T const &a, T const &b) const {
        REQUIRE_EQ(a, b);
    }
};

class flare_checker {
public:
    FLARE_INLINE_FUNCTION void truth(bool x) const {
        if (!x) flare::abort("SIMD unit test truth condition failed on device");
    }

    template<class T>
    FLARE_INLINE_FUNCTION void equality(T const &a, T const &b) const {
        if (a != b)
            flare::abort("SIMD unit test equality condition failed on device");
    }
};

template<class T, class Abi>
inline void host_check_equality(
        flare::experimental::simd<T, Abi> const &expected_result,
        flare::experimental::simd<T, Abi> const &computed_result,
        std::size_t nlanes) {
    gtest_checker checker;
    for (std::size_t i = 0; i < nlanes; ++i) {
        checker.equality(expected_result[i], computed_result[i]);
    }
    using mask_type = typename flare::experimental::simd<T, Abi>::mask_type;
    mask_type mask(false);
    for (std::size_t i = 0; i < nlanes; ++i) {
        mask[i] = true;
    }
    checker.equality((expected_result == computed_result) && mask, mask);
}

template<class T, class Abi>
FLARE_INLINE_FUNCTION void device_check_equality(
        flare::experimental::simd<T, Abi> const &expected_result,
        flare::experimental::simd<T, Abi> const &computed_result,
        std::size_t nlanes) {
    flare_checker checker;
    for (std::size_t i = 0; i < nlanes; ++i) {
        checker.equality(expected_result[i], computed_result[i]);
    }
    using mask_type = typename flare::experimental::simd<T, Abi>::mask_type;
    mask_type mask(false);
    for (std::size_t i = 0; i < nlanes; ++i) {
        mask[i] = true;
    }
    checker.equality((expected_result == computed_result) && mask, mask);
}

template<typename T, typename Abi>
FLARE_INLINE_FUNCTION void check_equality(
        flare::experimental::simd<T, Abi> const &expected_result,
        flare::experimental::simd<T, Abi> const &computed_result,
        std::size_t nlanes) {
    FLARE_IF_ON_HOST(
            (host_check_equality(expected_result, computed_result, nlanes);))
    FLARE_IF_ON_DEVICE(
            (device_check_equality(expected_result, computed_result, nlanes);))
}

class load_element_aligned {
public:
    template<class T, class Abi>
    bool host_load(T const *mem, std::size_t n,
                   flare::experimental::simd<T, Abi> &result) const {
        if (n < result.size()) return false;
        result.copy_from(mem, flare::experimental::element_aligned_tag());
        return true;
    }

    template<class T, class Abi>
    FLARE_INLINE_FUNCTION bool device_load(
            T const *mem, std::size_t n,
            flare::experimental::simd<T, Abi> &result) const {
        if (n < result.size()) return false;
        result.copy_from(mem, flare::experimental::element_aligned_tag());
        return true;
    }
};

class load_masked {
public:
    template<class T, class Abi>
    bool host_load(T const *mem, std::size_t n,
                   flare::experimental::simd<T, Abi> &result) const {
        using mask_type = typename flare::experimental::simd<T, Abi>::mask_type;
        mask_type mask(false);
        for (std::size_t i = 0; i < n; ++i) {
            mask[i] = true;
        }
        where(mask, result)
                .copy_from(mem, flare::experimental::element_aligned_tag());
        where(!mask, result) = 0;
        return true;
    }

    template<class T, class Abi>
    FLARE_INLINE_FUNCTION bool device_load(
            T const *mem, std::size_t n,
            flare::experimental::simd<T, Abi> &result) const {
        using mask_type = typename flare::experimental::simd<T, Abi>::mask_type;
        mask_type mask(false);
        for (std::size_t i = 0; i < n; ++i) {
            mask[i] = true;
        }
        where(mask, result)
                .copy_from(mem, flare::experimental::element_aligned_tag());
        where(!mask, result) = T(0);
        return true;
    }
};

class load_as_scalars {
public:
    template<class T, class Abi>
    bool host_load(T const *mem, std::size_t n,
                   flare::experimental::simd<T, Abi> &result) const {
        for (std::size_t i = 0; i < n; ++i) {
            result[i] = mem[i];
        }
        for (std::size_t i = n; i < result.size(); ++i) {
            result[i] = T(0);
        }
        return true;
    }

    template<class T, class Abi>
    FLARE_INLINE_FUNCTION bool device_load(
            T const *mem, std::size_t n,
            flare::experimental::simd<T, Abi> &result) const {
        for (std::size_t i = 0; i < n; ++i) {
            result[i] = mem[i];
        }
        for (std::size_t i = n; i < result.size(); ++i) {
            result[i] = T(0);
        }
        return true;
    }
};

#endif
