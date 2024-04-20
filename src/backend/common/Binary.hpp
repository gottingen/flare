// Copyright 2023 The EA Authors.
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

#pragma once
#include <backend.hpp>
#include <math.hpp>
#include <types.hpp>

#ifndef __DH__
#define __DH__
#endif

#include "optypes.hpp"

namespace flare {
namespace common {

using namespace detail;  // NOLINT

// Because isnan(cfloat) and isnan(cdouble) is not defined
#define IS_NAN(val) !((val) == (val))

template<typename T, fly_op_t op>
struct Binary {
    static __DH__ T init();

    __DH__ T operator()(T lhs, T rhs);
};

template<typename T>
struct Binary<T, fly_add_t> {
    static __DH__ T init() { return scalar<T>(0); }

    __DH__ T operator()(T lhs, T rhs) { return lhs + rhs; }
};

template<typename T>
struct Binary<T, fly_sub_t> {
    static __DH__ T init() { return scalar<T>(0); }

    __DH__ T operator()(T lhs, T rhs) { return lhs - rhs; }
};

template<typename T>
struct Binary<T, fly_mul_t> {
    static __DH__ T init() { return scalar<T>(1); }

    __DH__ T operator()(T lhs, T rhs) { return lhs * rhs; }
};

template<typename T>
struct Binary<T, fly_div_t> {
    static __DH__ T init() { return scalar<T>(1); }

    __DH__ T operator()(T lhs, T rhs) { return lhs / rhs; }
};

template<typename T>
struct Binary<T, fly_or_t> {
    static __DH__ T init() { return scalar<T>(0); }

    __DH__ T operator()(T lhs, T rhs) { return lhs || rhs; }
};

template<typename T>
struct Binary<T, fly_and_t> {
    static __DH__ T init() { return scalar<T>(1); }

    __DH__ T operator()(T lhs, T rhs) { return lhs && rhs; }
};

template<typename T>
struct Binary<T, fly_notzero_t> {
    static __DH__ T init() { return scalar<T>(0); }

    __DH__ T operator()(T lhs, T rhs) { return lhs + rhs; }
};

template<typename T>
struct Binary<T, fly_min_t> {
    static __DH__ T init() { return maxval<T>(); }

    __DH__ T operator()(T lhs, T rhs) { return detail::min(lhs, rhs); }
};

template<>
struct Binary<char, fly_min_t> {
    static __DH__ char init() { return 1; }

    __DH__ char operator()(char lhs, char rhs) {
        return detail::min(lhs > 0, rhs > 0);
    }
};

#define SPECIALIZE_COMPLEX_MIN(T, Tr)                                       \
    template<>                                                              \
    struct Binary<T, fly_min_t> {                                            \
        static __DH__ T init() { return scalar<T>(maxval<Tr>()); }          \
                                                                            \
        __DH__ T operator()(T lhs, T rhs) { return detail::min(lhs, rhs); } \
    };

SPECIALIZE_COMPLEX_MIN(cfloat, float)
SPECIALIZE_COMPLEX_MIN(cdouble, double)

#undef SPECIALIZE_COMPLEX_MIN

template<typename T>
struct Binary<T, fly_max_t> {
    static __DH__ T init() { return minval<T>(); }

    __DH__ T operator()(T lhs, T rhs) { return detail::max(lhs, rhs); }
};

template<>
struct Binary<char, fly_max_t> {
    static __DH__ char init() { return 0; }

    __DH__ char operator()(char lhs, char rhs) { return max(lhs > 0, rhs > 0); }
};

#define SPECIALIZE_COMPLEX_MAX(T, Tr)                                       \
    template<>                                                              \
    struct Binary<T, fly_max_t> {                                            \
        static __DH__ T init() { return scalar<T>(detail::scalar<Tr>(0)); } \
                                                                            \
        __DH__ T operator()(T lhs, T rhs) { return detail::max(lhs, rhs); } \
    };

SPECIALIZE_COMPLEX_MAX(cfloat, float)
SPECIALIZE_COMPLEX_MAX(cdouble, double)

#undef SPECIALIZE_COMPLEX_MAX

}  // namespace common
}  // namespace flare
