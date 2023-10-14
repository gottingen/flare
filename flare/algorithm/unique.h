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

#ifndef FLARE_ALGORITHM_UNIQUE_H_
#define FLARE_ALGORITHM_UNIQUE_H_

#include <flare/algorithm/unique_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

// note: the enable_if below is to avoid "call to ... is ambiguous"
// for example in the unit test when using a variadic function

// overload set1
template <class ExecutionSpace, class IteratorType>
std::enable_if_t<!::flare::is_tensor<IteratorType>::value, IteratorType> unique(
    const ExecutionSpace& ex, IteratorType first, IteratorType last) {
  return detail::unique_impl("flare::unique_iterator_api_default", ex, first,
                           last);
}

template <class ExecutionSpace, class IteratorType>
std::enable_if_t<!::flare::is_tensor<IteratorType>::value, IteratorType> unique(
    const std::string& label, const ExecutionSpace& ex, IteratorType first,
    IteratorType last) {
  return detail::unique_impl(label, ex, first, last);
}

template <class ExecutionSpace, class DataType, class... Properties>
auto unique(const ExecutionSpace& ex,
            const ::flare::Tensor<DataType, Properties...>& tensor) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);
  return ::flare::experimental::unique("flare::unique_tensor_api_default", ex,
                                        begin(tensor), end(tensor));
}

template <class ExecutionSpace, class DataType, class... Properties>
auto unique(const std::string& label, const ExecutionSpace& ex,
            const ::flare::Tensor<DataType, Properties...>& tensor) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);
  return ::flare::experimental::unique(label, ex, begin(tensor), end(tensor));
}

// overload set2
template <class ExecutionSpace, class IteratorType, class BinaryPredicate>
IteratorType unique(const ExecutionSpace& ex, IteratorType first,
                    IteratorType last, BinaryPredicate pred) {
  return detail::unique_impl("flare::unique_iterator_api_default", ex, first,
                           last, pred);
}

template <class ExecutionSpace, class IteratorType, class BinaryPredicate>
IteratorType unique(const std::string& label, const ExecutionSpace& ex,
                    IteratorType first, IteratorType last,
                    BinaryPredicate pred) {
  return detail::unique_impl(label, ex, first, last, pred);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class BinaryPredicate>
auto unique(const ExecutionSpace& ex,
            const ::flare::Tensor<DataType, Properties...>& tensor,
            BinaryPredicate pred) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);
  return detail::unique_impl("flare::unique_tensor_api_default", ex, begin(tensor),
                           end(tensor), std::move(pred));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class BinaryPredicate>
auto unique(const std::string& label, const ExecutionSpace& ex,
            const ::flare::Tensor<DataType, Properties...>& tensor,
            BinaryPredicate pred) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);
  return detail::unique_impl(label, ex, begin(tensor), end(tensor), std::move(pred));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_UNIQUE_H_
