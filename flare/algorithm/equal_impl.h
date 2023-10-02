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

#ifndef FLARE_ALGORITHM_EQUAL_IMPL_H_
#define FLARE_ALGORITHM_EQUAL_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IteratorType1, class IteratorType2, class BinaryPredicateType>
struct StdEqualFunctor {
  using index_type = typename IteratorType1::difference_type;

  IteratorType1 m_first1;
  IteratorType2 m_first2;
  BinaryPredicateType m_predicate;

  FLARE_FUNCTION
  void operator()(index_type i, std::size_t& lsum) const {
    if (!m_predicate(m_first1[i], m_first2[i])) {
      lsum = 1;
    }
  }

  FLARE_FUNCTION
  StdEqualFunctor(IteratorType1 _first1, IteratorType2 _first2,
                  BinaryPredicateType _predicate)
      : m_first1(std::move(_first1)),
        m_first2(std::move(_first2)),
        m_predicate(std::move(_predicate)) {}
};

//
// exespace impl
//
template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
bool equal_exespace_impl(const std::string& label, const ExecutionSpace& ex,
                         IteratorType1 first1, IteratorType1 last1,
                         IteratorType2 first2, BinaryPredicateType predicate) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first1, first2);
  detail::static_assert_iterators_have_matching_difference_type(first1, first2);
  detail::expect_valid_range(first1, last1);

  // run
  const auto num_elements = flare::experimental::distance(first1, last1);
  std::size_t different   = 0;
  ::flare::parallel_reduce(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements),
      StdEqualFunctor(first1, first2, predicate), different);
  ex.fence("flare::equal: fence after operation");

  return !different;
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
bool equal_exespace_impl(const std::string& label, const ExecutionSpace& ex,
                         IteratorType1 first1, IteratorType1 last1,
                         IteratorType2 first2) {
  using value_type1 = typename IteratorType1::value_type;
  using value_type2 = typename IteratorType2::value_type;
  using pred_t      = StdAlgoEqualBinaryPredicate<value_type1, value_type2>;
  return equal_exespace_impl(label, ex, first1, last1, first2, pred_t());
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
bool equal_exespace_impl(const std::string& label, const ExecutionSpace& ex,
                         IteratorType1 first1, IteratorType1 last1,
                         IteratorType2 first2, IteratorType2 last2,
                         BinaryPredicateType predicate) {
  const auto d1 = ::flare::experimental::distance(first1, last1);
  const auto d2 = ::flare::experimental::distance(first2, last2);
  if (d1 != d2) {
    return false;
  }

  return equal_exespace_impl(label, ex, first1, last1, first2, predicate);
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
bool equal_exespace_impl(const std::string& label, const ExecutionSpace& ex,
                         IteratorType1 first1, IteratorType1 last1,
                         IteratorType2 first2, IteratorType2 last2) {
  detail::expect_valid_range(first1, last1);
  detail::expect_valid_range(first2, last2);

  using value_type1 = typename IteratorType1::value_type;
  using value_type2 = typename IteratorType2::value_type;
  using pred_t      = StdAlgoEqualBinaryPredicate<value_type1, value_type2>;
  return equal_exespace_impl(label, ex, first1, last1, first2, last2, pred_t());
}

//
// team impl
//
template <class TeamHandleType, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
FLARE_FUNCTION bool equal_team_impl(const TeamHandleType& teamHandle,
                                     IteratorType1 first1, IteratorType1 last1,
                                     IteratorType2 first2,
                                     BinaryPredicateType predicate) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first1, first2);
  detail::static_assert_iterators_have_matching_difference_type(first1, first2);
  detail::expect_valid_range(first1, last1);

  // run
  const auto num_elements = flare::experimental::distance(first1, last1);
  std::size_t different   = 0;
  ::flare::parallel_reduce(TeamThreadRange(teamHandle, 0, num_elements),
                            StdEqualFunctor(first1, first2, predicate),
                            different);
  teamHandle.team_barrier();

  return !different;
}

template <class TeamHandleType, class IteratorType1, class IteratorType2>
FLARE_FUNCTION bool equal_team_impl(const TeamHandleType& teamHandle,
                                     IteratorType1 first1, IteratorType1 last1,
                                     IteratorType2 first2) {
  using value_type1 = typename IteratorType1::value_type;
  using value_type2 = typename IteratorType2::value_type;
  using pred_t      = StdAlgoEqualBinaryPredicate<value_type1, value_type2>;
  return equal_team_impl(teamHandle, first1, last1, first2, pred_t());
}

template <class TeamHandleType, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
FLARE_FUNCTION bool equal_team_impl(const TeamHandleType& teamHandle,
                                     IteratorType1 first1, IteratorType1 last1,
                                     IteratorType2 first2, IteratorType2 last2,
                                     BinaryPredicateType predicate) {
  const auto d1 = ::flare::experimental::distance(first1, last1);
  const auto d2 = ::flare::experimental::distance(first2, last2);
  if (d1 != d2) {
    return false;
  }

  return equal_team_impl(teamHandle, first1, last1, first2, predicate);
}

template <class TeamHandleType, class IteratorType1, class IteratorType2>
FLARE_FUNCTION bool equal_team_impl(const TeamHandleType& teamHandle,
                                     IteratorType1 first1, IteratorType1 last1,
                                     IteratorType2 first2,
                                     IteratorType2 last2) {
  detail::expect_valid_range(first1, last1);
  detail::expect_valid_range(first2, last2);

  using value_type1 = typename IteratorType1::value_type;
  using value_type2 = typename IteratorType2::value_type;
  using pred_t      = StdAlgoEqualBinaryPredicate<value_type1, value_type2>;
  return equal_team_impl(teamHandle, first1, last1, first2, last2, pred_t());
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_EQUAL_IMPL_H_
