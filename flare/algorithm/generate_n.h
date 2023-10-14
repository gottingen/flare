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

#ifndef FLARE_ALGORITHM_GENERATE_N_H_
#define FLARE_ALGORITHM_GENERATE_N_H_

#include <flare/algorithm/generate_n_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

template <class ExecutionSpace, class IteratorType, class Size, class Generator>
IteratorType generate_n(const ExecutionSpace& ex, IteratorType first,
                        Size count, Generator g) {
  detail::generate_n_impl("flare::generate_n_iterator_api_default", ex, first,
                        count, std::move(g));
  return first + count;
}

template <class ExecutionSpace, class IteratorType, class Size, class Generator>
IteratorType generate_n(const std::string& label, const ExecutionSpace& ex,
                        IteratorType first, Size count, Generator g) {
  detail::generate_n_impl(label, ex, first, count, std::move(g));
  return first + count;
}

template <class ExecutionSpace, class DataType, class... Properties, class Size,
          class Generator>
auto generate_n(const ExecutionSpace& ex,
                const ::flare::Tensor<DataType, Properties...>& tensor, Size count,
                Generator g) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);

  return detail::generate_n_impl("flare::generate_n_tensor_api_default", ex,
                               begin(tensor), count, std::move(g));
}

template <class ExecutionSpace, class DataType, class... Properties, class Size,
          class Generator>
auto generate_n(const std::string& label, const ExecutionSpace& ex,
                const ::flare::Tensor<DataType, Properties...>& tensor, Size count,
                Generator g) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);

  return detail::generate_n_impl(label, ex, begin(tensor), count, std::move(g));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_GENERATE_N_H_
