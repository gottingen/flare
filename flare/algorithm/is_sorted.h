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

#ifndef FLARE_ALGORITHM_IS_SORTED_H_
#define FLARE_ALGORITHM_IS_SORTED_H_

#include <flare/algorithm/is_sorted_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

template <class ExecutionSpace, class IteratorType>
bool is_sorted(const ExecutionSpace& ex, IteratorType first,
               IteratorType last) {
  return detail::is_sorted_impl("flare::is_sorted_iterator_api_default", ex,
                              first, last);
}

template <class ExecutionSpace, class IteratorType>
bool is_sorted(const std::string& label, const ExecutionSpace& ex,
               IteratorType first, IteratorType last) {
  return detail::is_sorted_impl(label, ex, first, last);
}

template <class ExecutionSpace, class DataType, class... Properties>
bool is_sorted(const ExecutionSpace& ex,
               const ::flare::Tensor<DataType, Properties...>& tensor) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);

  namespace KE = ::flare::experimental;
  return detail::is_sorted_impl("flare::is_sorted_tensor_api_default", ex,
                              KE::cbegin(tensor), KE::cend(tensor));
}

template <class ExecutionSpace, class DataType, class... Properties>
bool is_sorted(const std::string& label, const ExecutionSpace& ex,
               const ::flare::Tensor<DataType, Properties...>& tensor) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);

  namespace KE = ::flare::experimental;
  return detail::is_sorted_impl(label, ex, KE::cbegin(tensor), KE::cend(tensor));
}

template <class ExecutionSpace, class IteratorType, class ComparatorType>
bool is_sorted(const ExecutionSpace& ex, IteratorType first, IteratorType last,
               ComparatorType comp) {
  return detail::is_sorted_impl("flare::is_sorted_iterator_api_default", ex,
                              first, last, std::move(comp));
}

template <class ExecutionSpace, class IteratorType, class ComparatorType>
bool is_sorted(const std::string& label, const ExecutionSpace& ex,
               IteratorType first, IteratorType last, ComparatorType comp) {
  return detail::is_sorted_impl(label, ex, first, last, std::move(comp));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ComparatorType>
bool is_sorted(const ExecutionSpace& ex,
               const ::flare::Tensor<DataType, Properties...>& tensor,
               ComparatorType comp) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);

  namespace KE = ::flare::experimental;
  return detail::is_sorted_impl("flare::is_sorted_tensor_api_default", ex,
                              KE::cbegin(tensor), KE::cend(tensor),
                              std::move(comp));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ComparatorType>
bool is_sorted(const std::string& label, const ExecutionSpace& ex,
               const ::flare::Tensor<DataType, Properties...>& tensor,
               ComparatorType comp) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);

  namespace KE = ::flare::experimental;
  return detail::is_sorted_impl(label, ex, KE::cbegin(tensor), KE::cend(tensor),
                              std::move(comp));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_IS_SORTED_H_
