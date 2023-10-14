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

#ifndef FLARE_ALGORITHM_PARTITION_COPY_H_
#define FLARE_ALGORITHM_PARTITION_COPY_H_

#include <flare/algorithm/partition_copy_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorTrueType, class OutputIteratorFalseType,
          class PredicateType>
::flare::pair<OutputIteratorTrueType, OutputIteratorFalseType> partition_copy(
    const ExecutionSpace& ex, InputIteratorType from_first,
    InputIteratorType from_last, OutputIteratorTrueType to_first_true,
    OutputIteratorFalseType to_first_false, PredicateType p) {
  return detail::partition_copy_impl(
      "flare::partition_copy_iterator_api_default", ex, from_first, from_last,
      to_first_true, to_first_false, std::move(p));
}

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorTrueType, class OutputIteratorFalseType,
          class PredicateType>
::flare::pair<OutputIteratorTrueType, OutputIteratorFalseType> partition_copy(
    const std::string& label, const ExecutionSpace& ex,
    InputIteratorType from_first, InputIteratorType from_last,
    OutputIteratorTrueType to_first_true,
    OutputIteratorFalseType to_first_false, PredicateType p) {
  return detail::partition_copy_impl(label, ex, from_first, from_last,
                                   to_first_true, to_first_false, std::move(p));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class DataType3,
          class... Properties3, class PredicateType>
auto partition_copy(
    const ExecutionSpace& ex,
    const ::flare::Tensor<DataType1, Properties1...>& tensor_from,
    const ::flare::Tensor<DataType2, Properties2...>& tensor_dest_true,
    const ::flare::Tensor<DataType3, Properties3...>& tensor_dest_false,
    PredicateType p) {
  return detail::partition_copy_impl("flare::partition_copy_tensor_api_default",
                                   ex, cbegin(tensor_from), cend(tensor_from),
                                   begin(tensor_dest_true),
                                   begin(tensor_dest_false), std::move(p));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class DataType3,
          class... Properties3, class PredicateType>
auto partition_copy(
    const std::string& label, const ExecutionSpace& ex,
    const ::flare::Tensor<DataType1, Properties1...>& tensor_from,
    const ::flare::Tensor<DataType2, Properties2...>& tensor_dest_true,
    const ::flare::Tensor<DataType3, Properties3...>& tensor_dest_false,
    PredicateType p) {
  return detail::partition_copy_impl(label, ex, cbegin(tensor_from),
                                   cend(tensor_from), begin(tensor_dest_true),
                                   begin(tensor_dest_false), std::move(p));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_PARTITION_COPY_H_
