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

#ifndef FLARE_ALGORITHM_GENERATE_H_
#define FLARE_ALGORITHM_GENERATE_H_

#include <flare/algorithm/generate_n_impl.h>
#include <flare/algorithm/begin_end.h>

namespace flare {
namespace experimental {

template <class ExecutionSpace, class IteratorType, class Generator>
void generate(const ExecutionSpace& ex, IteratorType first, IteratorType last,
              Generator g) {
  detail::generate_impl("flare::generate_iterator_api_default", ex, first, last,
                      std::move(g));
}

template <class ExecutionSpace, class IteratorType, class Generator>
void generate(const std::string& label, const ExecutionSpace& ex,
              IteratorType first, IteratorType last, Generator g) {
  detail::generate_impl(label, ex, first, last, std::move(g));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class Generator>
void generate(const ExecutionSpace& ex,
              const ::flare::Tensor<DataType, Properties...>& tensor,
              Generator g) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);

  detail::generate_impl("flare::generate_tensor_api_default", ex, begin(tensor),
                      end(tensor), std::move(g));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class Generator>
void generate(const std::string& label, const ExecutionSpace& ex,
              const ::flare::Tensor<DataType, Properties...>& tensor,
              Generator g) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(tensor);

  detail::generate_impl(label, ex, begin(tensor), end(tensor), std::move(g));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_GENERATE_H_
