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

#ifndef FLARE_ALGORITHM_FOR_EACH_IMPL_H_
#define FLARE_ALGORITHM_FOR_EACH_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IteratorType, class UnaryFunctorType>
struct StdForEachFunctor {
  using index_type = typename IteratorType::difference_type;
  IteratorType m_first;
  UnaryFunctorType m_functor;

  FLARE_FUNCTION
  void operator()(index_type i) const { m_functor(m_first[i]); }

  FLARE_FUNCTION
  StdForEachFunctor(IteratorType _first, UnaryFunctorType _functor)
      : m_first(std::move(_first)), m_functor(std::move(_functor)) {}
};

template <class HandleType, class IteratorType, class UnaryFunctorType>
UnaryFunctorType for_each_exespace_impl(const std::string& label,
                                        const HandleType& handle,
                                        IteratorType first, IteratorType last,
                                        UnaryFunctorType functor) {
  // checks
  detail::static_assert_random_access_and_accessible(handle, first);
  detail::expect_valid_range(first, last);

  // run
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_for(
      label, RangePolicy<HandleType>(handle, 0, num_elements),
      StdForEachFunctor<IteratorType, UnaryFunctorType>(first, functor));
  handle.fence("flare::for_each: fence after operation");

  return functor;
}

template <class ExecutionSpace, class IteratorType, class SizeType,
          class UnaryFunctorType>
IteratorType for_each_n_exespace_impl(const std::string& label,
                                      const ExecutionSpace& ex,
                                      IteratorType first, SizeType n,
                                      UnaryFunctorType functor) {
  auto last = first + n;
  detail::static_assert_random_access_and_accessible(ex, first, last);
  detail::expect_valid_range(first, last);

  if (n == 0) {
    return first;
  }

  for_each_exespace_impl(label, ex, first, last, std::move(functor));
  // no neeed to fence since for_each_exespace_impl fences already

  return last;
}

//
// team impl
//
template <class TeamHandleType, class IteratorType, class UnaryFunctorType>
FLARE_FUNCTION UnaryFunctorType
for_each_team_impl(const TeamHandleType& teamHandle, IteratorType first,
                   IteratorType last, UnaryFunctorType functor) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first);
  detail::expect_valid_range(first, last);
  // run
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_for(
      TeamThreadRange(teamHandle, 0, num_elements),
      StdForEachFunctor<IteratorType, UnaryFunctorType>(first, functor));
  teamHandle.team_barrier();
  return functor;
}

template <class TeamHandleType, class IteratorType, class SizeType,
          class UnaryFunctorType>
FLARE_FUNCTION IteratorType
for_each_n_team_impl(const TeamHandleType& teamHandle, IteratorType first,
                     SizeType n, UnaryFunctorType functor) {
  auto last = first + n;
  detail::static_assert_random_access_and_accessible(teamHandle, first, last);
  detail::expect_valid_range(first, last);

  if (n == 0) {
    return first;
  }

  for_each_team_impl(teamHandle, first, last, std::move(functor));
  // no neeed to fence since for_each_team_impl fences already

  return last;
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_FOR_EACH_IMPL_H_
