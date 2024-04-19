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
#include <types.hpp>

#include <type_traits>

namespace flare {
namespace common {

// The value returns true if the type is a complex type. False otherwise
template<typename T>
struct is_complex {
    static const bool value = false;
};
template<>
struct is_complex<detail::cfloat> {
    static const bool value = true;
};
template<>
struct is_complex<detail::cdouble> {
    static const bool value = true;
};

/// This is an enable_if for complex types.
template<typename T, typename TYPE = void>
using if_complex = typename std::enable_if<is_complex<T>::value, TYPE>::type;

/// This is an enable_if for real types.
template<typename T, typename TYPE = void>
using if_real =
    typename std::enable_if<is_complex<T>::value == false, TYPE>::type;

}  // namespace common
}  // namespace flare
