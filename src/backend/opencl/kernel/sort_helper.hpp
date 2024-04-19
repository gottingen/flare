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
#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <debug_opencl.hpp>
#include <type_traits>

namespace flare {
namespace opencl {
namespace kernel {

template<typename T>
using htype_t = typename std::conditional<std::is_same<T, common::half>::value,
                                          cl_half, T>::type;

// If type is cdouble, return std::complex<double>, else return T
template<typename T>
using ztype_t =
    typename std::conditional<std::is_same<T, cdouble>::value,
                              std::complex<double>, htype_t<T>>::type;

// If type is cfloat, return std::complex<float>, else return ztype_t
template<typename T>
using ctype_t =
    typename std::conditional<std::is_same<T, cfloat>::value,
                              std::complex<float>, ztype_t<T>>::type;

// If type is intl, return cl_long, else return ctype_t
template<typename T>
using ltype_t = typename std::conditional<std::is_same<T, intl>::value, cl_long,
                                          ctype_t<T>>::type;

// If type is uintl, return cl_ulong, else return ltype_t
template<typename T>
using type_t = typename std::conditional<std::is_same<T, uintl>::value,
                                         cl_ulong, ltype_t<T>>::type;
}  // namespace kernel
}  // namespace opencl
}  // namespace flare
