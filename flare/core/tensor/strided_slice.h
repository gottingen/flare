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


#ifndef FLARE_CORE_TENSOR_STRIDED_SLICE_H_
#define FLARE_CORE_TENSOR_STRIDED_SLICE_H_

#include <type_traits>

namespace flare {
namespace experimental {

namespace {
  template<class T>
  struct __mdspan_is_integral_constant: std::false_type {};

  template<class T, T val>
  struct __mdspan_is_integral_constant<std::integral_constant<T,val>>: std::true_type {};
}
// Slice Specifier allowing for strides and compile time extent
template <class OffsetType, class ExtentType, class StrideType>
struct strided_slice {
  using offset_type = OffsetType;
  using extent_type = ExtentType;
  using stride_type = StrideType;

  OffsetType offset;
  ExtentType extent;
  StrideType stride;

  static_assert(std::is_integral_v<OffsetType> || __mdspan_is_integral_constant<OffsetType>::value);
  static_assert(std::is_integral_v<ExtentType> || __mdspan_is_integral_constant<ExtentType>::value);
  static_assert(std::is_integral_v<StrideType> || __mdspan_is_integral_constant<StrideType>::value);
};

} // experimental
} // flare
#endif  // FLARE_CORE_TENSOR_STRIDED_SLICE_H_
