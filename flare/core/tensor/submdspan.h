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


#ifndef FLARE_CORE_TENSOR_SUBMDSPAN_H_
#define FLARE_CORE_TENSOR_SUBMDSPAN_H_

#include <flare/core/tensor/submdspan_extents.h>
#include <flare/core/tensor/submdspan_mapping.h>

namespace flare {
namespace experimental {
template <class ElementType, class Extents, class LayoutPolicy,
          class AccessorPolicy, class... SliceSpecifiers>
MDSPAN_INLINE_FUNCTION
constexpr auto
submdspan(const mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy> &src,
          SliceSpecifiers... slices) {
  const auto sub_mapping_offset = submdspan_mapping(src.mapping(), slices...);
  // NVCC has a problem with the deduction so lets figure out the type
  using sub_mapping_t = std::remove_cv_t<decltype(sub_mapping_offset.mapping)>;
  using sub_extents_t = typename sub_mapping_t::extents_type;
  using sub_layout_t = typename sub_mapping_t::layout_type;
  using sub_accessor_t = typename AccessorPolicy::offset_policy;
  return mdspan<ElementType, sub_extents_t, sub_layout_t, sub_accessor_t>(
      src.accessor().offset(src.data_handle(), sub_mapping_offset.offset),
      sub_mapping_offset.mapping,
      sub_accessor_t(src.accessor()));
}
} // namespace experimental
} // namespace flare
#endif  // FLARE_CORE_TENSOR_SUBMDSPAN_H_
