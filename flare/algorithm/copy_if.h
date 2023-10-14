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

#ifndef FLARE_ALGORITHM_COPY_IF_H_
#define FLARE_ALGORITHM_COPY_IF_H_

#include <flare/algorithm/copy_if_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

template <class ExecutionSpace, class InputIterator, class OutputIterator,
          class Predicate>
OutputIterator copy_if(const ExecutionSpace& ex, InputIterator first,
                       InputIterator last, OutputIterator d_first,
                       Predicate pred) {
  return detail::copy_if_impl("flare::copy_if_iterator_api_default", ex, first,
                            last, d_first, std::move(pred));
}

template <class ExecutionSpace, class InputIterator, class OutputIterator,
          class Predicate>
OutputIterator copy_if(const std::string& label, const ExecutionSpace& ex,
                       InputIterator first, InputIterator last,
                       OutputIterator d_first, Predicate pred) {
  return detail::copy_if_impl(label, ex, first, last, d_first, std::move(pred));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class Predicate>
auto copy_if(const ExecutionSpace& ex,
             const ::flare::Tensor<DataType1, Properties1...>& source,
             ::flare::Tensor<DataType2, Properties2...>& dest, Predicate pred) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  return detail::copy_if_impl("flare::copy_if_tensor_api_default", ex,
                            cbegin(source), cend(source), begin(dest),
                            std::move(pred));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class Predicate>
auto copy_if(const std::string& label, const ExecutionSpace& ex,
             const ::flare::Tensor<DataType1, Properties1...>& source,
             ::flare::Tensor<DataType2, Properties2...>& dest, Predicate pred) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  return detail::copy_if_impl(label, ex, cbegin(source), cend(source),
                            begin(dest), std::move(pred));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_COPY_IF_H_
