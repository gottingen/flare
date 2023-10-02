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

#ifndef FLARE_ALGORITHM_GENERATE_IMPL_H_
#define FLARE_ALGORITHM_GENERATE_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IteratorType, class Generator>
struct StdGenerateFunctor {
  using index_type = typename IteratorType::difference_type;
  IteratorType m_first;
  Generator m_generator;

  FLARE_FUNCTION
  void operator()(index_type i) const { m_first[i] = m_generator(); }

  FLARE_FUNCTION
  StdGenerateFunctor(IteratorType _first, Generator _g)
      : m_first(std::move(_first)), m_generator(std::move(_g)) {}
};

template <class ExecutionSpace, class IteratorType, class Generator>
void generate_impl(const std::string& label, const ExecutionSpace& ex,
                   IteratorType first, IteratorType last, Generator g) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);

  // aliases
  using func_t = StdGenerateFunctor<IteratorType, Generator>;

  // run
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_for(label,
                         RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                         func_t(first, g));
  ex.fence("flare::generate: fence after operation");
}

template <class ExecutionSpace, class IteratorType, class Size, class Generator>
IteratorType generate_n_impl(const std::string& label, const ExecutionSpace& ex,
                             IteratorType first, Size count, Generator g) {
  if (count <= 0) {
    return first;
  }

  generate_impl(label, ex, first, first + count, g);
  return first + count;
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_GENERATE_IMPL_H_
