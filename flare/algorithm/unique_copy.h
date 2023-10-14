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

#ifndef FLARE_ALGORITHM_UNIQUE_COPY_H_
#define FLARE_ALGORITHM_UNIQUE_COPY_H_

#include <flare/algorithm/unique_copy_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

// overload set1
template <class ExecutionSpace, class InputIterator, class OutputIterator>
std::enable_if_t<!::flare::is_tensor<InputIterator>::value, OutputIterator>
unique_copy(const ExecutionSpace& ex, InputIterator first, InputIterator last,
            OutputIterator d_first) {
  return detail::unique_copy_impl("flare::unique_copy_iterator_api_default", ex,
                                first, last, d_first);
}

template <class ExecutionSpace, class InputIterator, class OutputIterator>
std::enable_if_t<!::flare::is_tensor<InputIterator>::value, OutputIterator>
unique_copy(const std::string& label, const ExecutionSpace& ex,
            InputIterator first, InputIterator last, OutputIterator d_first) {
  return detail::unique_copy_impl(label, ex, first, last, d_first);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto unique_copy(const ExecutionSpace& ex,
                 const ::flare::Tensor<DataType1, Properties1...>& source,
                 const ::flare::Tensor<DataType2, Properties2...>& dest) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  return ::flare::experimental::unique_copy(
      "flare::unique_copy_tensor_api_default", ex, cbegin(source), cend(source),
      begin(dest));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto unique_copy(const std::string& label, const ExecutionSpace& ex,
                 const ::flare::Tensor<DataType1, Properties1...>& source,
                 const ::flare::Tensor<DataType2, Properties2...>& dest) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  return ::flare::experimental::unique_copy(label, ex, cbegin(source),
                                             cend(source), begin(dest));
}

// overload set2
template <class ExecutionSpace, class InputIterator, class OutputIterator,
          class BinaryPredicate>
OutputIterator unique_copy(const ExecutionSpace& ex, InputIterator first,
                           InputIterator last, OutputIterator d_first,
                           BinaryPredicate pred) {
  return detail::unique_copy_impl("flare::unique_copy_iterator_api_default", ex,
                                first, last, d_first, pred);
}

template <class ExecutionSpace, class InputIterator, class OutputIterator,
          class BinaryPredicate>
OutputIterator unique_copy(const std::string& label, const ExecutionSpace& ex,
                           InputIterator first, InputIterator last,
                           OutputIterator d_first, BinaryPredicate pred) {
  return detail::unique_copy_impl(label, ex, first, last, d_first, pred);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicate>
auto unique_copy(const ExecutionSpace& ex,
                 const ::flare::Tensor<DataType1, Properties1...>& source,
                 const ::flare::Tensor<DataType2, Properties2...>& dest,
                 BinaryPredicate pred) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  return detail::unique_copy_impl("flare::unique_copy_tensor_api_default", ex,
                                cbegin(source), cend(source), begin(dest),
                                std::move(pred));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicate>
auto unique_copy(const std::string& label, const ExecutionSpace& ex,
                 const ::flare::Tensor<DataType1, Properties1...>& source,
                 const ::flare::Tensor<DataType2, Properties2...>& dest,
                 BinaryPredicate pred) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  return detail::unique_copy_impl(label, ex, cbegin(source), cend(source),
                                begin(dest), std::move(pred));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_UNIQUE_COPY_H_
