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

#ifndef FLARE_ALGORITHM_COPY_H_
#define FLARE_ALGORITHM_COPY_H_

#include <flare/algorithm/copy_n_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator copy(const ExecutionSpace& ex, InputIterator first,
                    InputIterator last, OutputIterator d_first) {
  return detail::copy_impl("flare::copy_iterator_api_default", ex, first, last,
                         d_first);
}

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator copy(const std::string& label, const ExecutionSpace& ex,
                    InputIterator first, InputIterator last,
                    OutputIterator d_first) {
  return detail::copy_impl(label, ex, first, last, d_first);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto copy(const ExecutionSpace& ex,
          const ::flare::Tensor<DataType1, Properties1...>& source,
          ::flare::Tensor<DataType2, Properties2...>& dest) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  namespace KE = ::flare::experimental;
  return detail::copy_impl("flare::copy_tensor_api_default", ex,
                         KE::cbegin(source), KE::cend(source), KE::begin(dest));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto copy(const std::string& label, const ExecutionSpace& ex,
          const ::flare::Tensor<DataType1, Properties1...>& source,
          ::flare::Tensor<DataType2, Properties2...>& dest) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(source);
  detail::static_assert_is_admissible_to_flare_std_algorithms(dest);

  namespace KE = ::flare::experimental;
  return detail::copy_impl(label, ex, KE::cbegin(source), KE::cend(source),
                         KE::begin(dest));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_COPY_H_
