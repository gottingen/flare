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

#ifndef FLARE_ALGORITHM_NESTED_SORT_PUBLIC_API_H_
#define FLARE_ALGORITHM_NESTED_SORT_PUBLIC_API_H_

#include <flare/algorithm/nested_sort_impl.h>
#include <flare/core.h>
#include <flare/algorithm/helper_predicates_impl.h>

namespace flare {
namespace experimental {

template <class TeamMember, class TensorType>
FLARE_INLINE_FUNCTION void sort_team(const TeamMember& t,
                                      const TensorType& tensor) {
  detail::sort_nested_impl(t, tensor, nullptr,
                         experimental::detail::StdAlgoLessThanBinaryPredicate<
                             typename TensorType::non_const_value_type>(),
                         detail::NestedRange<true>());
}

template <class TeamMember, class TensorType, class Comparator>
FLARE_INLINE_FUNCTION void sort_team(const TeamMember& t, const TensorType& tensor,
                                      const Comparator& comp) {
  detail::sort_nested_impl(t, tensor, nullptr, comp, detail::NestedRange<true>());
}

template <class TeamMember, class KeyTensorType, class ValueTensorType>
FLARE_INLINE_FUNCTION void sort_by_key_team(const TeamMember& t,
                                             const KeyTensorType& keyTensor,
                                             const ValueTensorType& valueTensor) {
  detail::sort_nested_impl(t, keyTensor, valueTensor,
                         experimental::detail::StdAlgoLessThanBinaryPredicate<
                             typename KeyTensorType::non_const_value_type>(),
                         detail::NestedRange<true>());
}

template <class TeamMember, class KeyTensorType, class ValueTensorType,
          class Comparator>
FLARE_INLINE_FUNCTION void sort_by_key_team(const TeamMember& t,
                                             const KeyTensorType& keyTensor,
                                             const ValueTensorType& valueTensor,
                                             const Comparator& comp) {
  detail::sort_nested_impl(t, keyTensor, valueTensor, comp,
                         detail::NestedRange<true>());
}

template <class TeamMember, class TensorType>
FLARE_INLINE_FUNCTION void sort_thread(const TeamMember& t,
                                        const TensorType& tensor) {
  detail::sort_nested_impl(t, tensor, nullptr,
                         experimental::detail::StdAlgoLessThanBinaryPredicate<
                             typename TensorType::non_const_value_type>(),
                         detail::NestedRange<false>());
}

template <class TeamMember, class TensorType, class Comparator>
FLARE_INLINE_FUNCTION void sort_thread(const TeamMember& t,
                                        const TensorType& tensor,
                                        const Comparator& comp) {
  detail::sort_nested_impl(t, tensor, nullptr, comp, detail::NestedRange<false>());
}

template <class TeamMember, class KeyTensorType, class ValueTensorType>
FLARE_INLINE_FUNCTION void sort_by_key_thread(const TeamMember& t,
                                               const KeyTensorType& keyTensor,
                                               const ValueTensorType& valueTensor) {
  detail::sort_nested_impl(t, keyTensor, valueTensor,
                         experimental::detail::StdAlgoLessThanBinaryPredicate<
                             typename KeyTensorType::non_const_value_type>(),
                         detail::NestedRange<false>());
}

template <class TeamMember, class KeyTensorType, class ValueTensorType,
          class Comparator>
FLARE_INLINE_FUNCTION void sort_by_key_thread(const TeamMember& t,
                                               const KeyTensorType& keyTensor,
                                               const ValueTensorType& valueTensor,
                                               const Comparator& comp) {
  detail::sort_nested_impl(t, keyTensor, valueTensor, comp,
                         detail::NestedRange<false>());
}

}  // namespace experimental
}  // namespace flare
#endif  // FLARE_ALGORITHM_NESTED_SORT_PUBLIC_API_H_
