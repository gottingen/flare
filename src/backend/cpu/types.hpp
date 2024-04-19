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
#include <common/kernel_type.hpp>
#include <complex>

namespace flare {
namespace cpu {

namespace {
template<typename T>
const char *shortname(bool caps = false) {
    return caps ? "?" : "?";
}

template<typename T>
const char *getFullName() {
    return "N/A";
}

}  // namespace

using cdouble = std::complex<double>;
using cfloat  = std::complex<float>;
using intl    = long long;
using uint    = unsigned int;
using uchar   = unsigned char;
using uintl   = unsigned long long;
using ushort  = unsigned short;

template<typename T>
using compute_t = typename common::kernel_type<T>::compute;

template<typename T>
using data_t = typename common::kernel_type<T>::data;

}  // namespace cpu

namespace common {
template<typename T>
struct kernel_type;

class half;

template<>
struct kernel_type<flare::common::half> {
    using data = flare::common::half;

    // These are the types within a kernel
    using native = float;

    using compute = float;
};
}  // namespace common

}  // namespace flare
