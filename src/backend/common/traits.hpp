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

#include <common/err_common.hpp>
#include <fly/defines.h>

namespace fly {
template<typename T>
struct dtype_traits;
}

namespace flare {
namespace common {
class half;

namespace {

inline size_t dtypeSize(fly::dtype type) {
    switch (type) {
        case u8:
        case b8: return 1;
        case s16:
        case u16:
        case f16: return 2;
        case s32:
        case u32:
        case f32: return 4;
        case u64:
        case s64:
        case c32:
        case f64: return 8;
        case c64: return 16;
        default: FLY_RETURN_ERROR("Unsupported type", FLY_ERR_INTERNAL);
    }
}

constexpr bool isComplex(fly::dtype type) {
    return ((type == c32) || (type == c64));
}

constexpr bool isReal(fly::dtype type) { return !isComplex(type); }

constexpr bool isDouble(fly::dtype type) { return (type == f64 || type == c64); }

constexpr bool isSingle(fly::dtype type) { return (type == f32 || type == c32); }

constexpr bool isHalf(fly::dtype type) { return (type == f16); }

constexpr bool isRealFloating(fly::dtype type) {
    return (type == f64 || type == f32 || type == f16);
}

constexpr bool isInteger(fly::dtype type) {
    return (type == s32 || type == u32 || type == s64 || type == u64 ||
            type == s16 || type == u16 || type == u8);
}

constexpr bool isBool(fly::dtype type) { return (type == b8); }

constexpr bool isFloating(fly::dtype type) {
    return (!isInteger(type) && !isBool(type));
}

template<typename T, typename U, typename... Args>
constexpr bool is_any_of() {
    FLY_IF_CONSTEXPR(!sizeof...(Args)) { return std::is_same<T, U>::value; }
    else { return std::is_same<T, U>::value || is_any_of<T, Args...>(); }
}

}  // namespace
}  // namespace common
}  // namespace flare

namespace fly {
template<>
struct dtype_traits<flare::common::half> {
    enum { fly_type = f16, ctype = f16 };
    typedef flare::common::half base_type;
    static const char *getName() { return "half"; }
};
}  // namespace fly
