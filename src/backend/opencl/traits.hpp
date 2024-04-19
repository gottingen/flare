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

#include <common/defines.hpp>
#include <common/traits.hpp>
#include <types.hpp>

#include <sstream>
#include <string>

namespace fly {

template<>
struct dtype_traits<flare::opencl::cfloat> {
    enum { fly_type = c32 };
    typedef float base_type;
    static const char *getName() { return "float2"; }
};

template<>
struct dtype_traits<flare::opencl::cdouble> {
    enum { fly_type = c64 };
    typedef double base_type;
    static const char *getName() { return "double2"; }
};
}  // namespace fly

namespace flare {
namespace opencl {

template<typename T>
static bool iscplx() {
    return false;
}
template<>
inline bool iscplx<cfloat>() {
    return true;
}
template<>
inline bool iscplx<cdouble>() {
    return true;
}

template<typename T>
inline std::string scalar_to_option(const T &val) {
    using namespace flare::common;
    using std::to_string;
    return to_string(+val);
}

template<>
inline std::string scalar_to_option<cl_float2>(const cl_float2 &val) {
    std::ostringstream ss;
    ss << val.s[0] << "," << val.s[1];
    return ss.str();
}

template<>
inline std::string scalar_to_option<cl_double2>(const cl_double2 &val) {
    std::ostringstream ss;
    ss << val.s[0] << "," << val.s[1];
    return ss.str();
}

using fly::dtype_traits;
}  // namespace opencl
}  // namespace flare
