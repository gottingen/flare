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

#ifndef FLARE_ALGORITHM_MISMATCH_IMPL_H_
#define FLARE_ALGORITHM_MISMATCH_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IteratorType1, class IteratorType2, class ReducerType,
          class BinaryPredicateType>
struct StdMismatchRedFunctor {
  using index_type     = typename IteratorType1::difference_type;
  using red_value_type = typename ReducerType::value_type;

  IteratorType1 m_first1;
  IteratorType2 m_first2;
  ReducerType m_reducer;
  BinaryPredicateType m_predicate;

  FLARE_FUNCTION
  void operator()(const index_type i, red_value_type& red_value) const {
    const auto& my_value1 = m_first1[i];
    const auto& my_value2 = m_first2[i];

    // FIXME_NVHPC using a ternary operator causes problems
    red_value_type rv = {i};
    if (m_predicate(my_value1, my_value2)) {
      rv = {::flare::reduction_identity<index_type>::min()};
    }

    m_reducer.join(red_value, rv);
  }

  FLARE_FUNCTION
  StdMismatchRedFunctor(IteratorType1 first1, IteratorType2 first2,
                        ReducerType reducer, BinaryPredicateType predicate)
      : m_first1(std::move(first1)),
        m_first2(std::move(first2)),
        m_reducer(std::move(reducer)),
        m_predicate(std::move(predicate)) {}
};

//
// exespace impl
//
template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
::flare::pair<IteratorType1, IteratorType2> mismatch_exespace_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType1 first1,
    IteratorType1 last1, IteratorType2 first2, IteratorType2 last2,
    BinaryPredicateType predicate) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first1, first2);
  detail::static_assert_iterators_have_matching_difference_type(first1, first2);
  detail::expect_valid_range(first1, last1);
  detail::expect_valid_range(first2, last2);

  // aliases
  using return_type          = ::flare::pair<IteratorType1, IteratorType2>;
  using index_type           = typename IteratorType1::difference_type;
  using reducer_type         = FirstLoc<index_type>;
  using reduction_value_type = typename reducer_type::value_type;

  const auto num_e1 = last1 - first1;
  const auto num_e2 = last2 - first2;
  if (num_e1 == 0 || num_e2 == 0) {
    return return_type(first1, first2);
  }

  // run
  const auto num_elemen_par_reduce = (num_e1 <= num_e2) ? num_e1 : num_e2;
  reduction_value_type red_result;
  reducer_type reducer(red_result);
  ::flare::parallel_reduce(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elemen_par_reduce),
      // use CTAD
      StdMismatchRedFunctor(first1, first2, reducer, std::move(predicate)),
      reducer);

  // fence not needed because reducing into scalar

  // decide and return
  constexpr auto red_min = ::flare::reduction_identity<index_type>::min();
  if (red_result.min_loc_true == red_min) {
    // in here means mismatch has not been found
    if (num_e1 == num_e2) {
      return return_type(last1, last2);
    } else if (num_e1 < num_e2) {
      return return_type(last1, first2 + num_e1);
    } else {
      return return_type(first1 + num_e2, last2);
    }
  } else {
    // in here means mismatch has been found
    return return_type(first1 + red_result.min_loc_true,
                       first2 + red_result.min_loc_true);
  }
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
::flare::pair<IteratorType1, IteratorType2> mismatch_exespace_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType1 first1,
    IteratorType1 last1, IteratorType2 first2, IteratorType2 last2) {
  using value_type1 = typename IteratorType1::value_type;
  using value_type2 = typename IteratorType2::value_type;
  using pred_t      = StdAlgoEqualBinaryPredicate<value_type1, value_type2>;
  return mismatch_exespace_impl(label, ex, first1, last1, first2, last2,
                                pred_t());
}

//
// team impl
//
template <class TeamHandleType, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
FLARE_FUNCTION ::flare::pair<IteratorType1, IteratorType2> mismatch_team_impl(
    const TeamHandleType& teamHandle, IteratorType1 first1, IteratorType1 last1,
    IteratorType2 first2, IteratorType2 last2, BinaryPredicateType predicate) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first1, first2);
  detail::static_assert_iterators_have_matching_difference_type(first1, first2);
  detail::expect_valid_range(first1, last1);
  detail::expect_valid_range(first2, last2);

  // aliases
  using return_type          = ::flare::pair<IteratorType1, IteratorType2>;
  using index_type           = typename IteratorType1::difference_type;
  using reducer_type         = FirstLoc<index_type>;
  using reduction_value_type = typename reducer_type::value_type;

  // trivial case: note that this is important,
  // for OpenMPTarget, omitting special handling of
  // the trivial case was giving all sorts of strange stuff.
  const auto num_e1 = last1 - first1;
  const auto num_e2 = last2 - first2;
  if (num_e1 == 0 || num_e2 == 0) {
    return return_type(first1, first2);
  }

  // run
  const auto num_elemen_par_reduce = (num_e1 <= num_e2) ? num_e1 : num_e2;
  reduction_value_type red_result;
  reducer_type reducer(red_result);
  ::flare::parallel_reduce(
      TeamThreadRange(teamHandle, 0, num_elemen_par_reduce),
      // use CTAD
      StdMismatchRedFunctor(first1, first2, reducer, std::move(predicate)),
      reducer);

  teamHandle.team_barrier();

  // decide and return
  constexpr auto red_min = ::flare::reduction_identity<index_type>::min();
  if (red_result.min_loc_true == red_min) {
    // in here means mismatch has not been found
    if (num_e1 == num_e2) {
      return return_type(last1, last2);
    } else if (num_e1 < num_e2) {
      return return_type(last1, first2 + num_e1);
    } else {
      return return_type(first1 + num_e2, last2);
    }
  } else {
    // in here means mismatch has been found
    return return_type(first1 + red_result.min_loc_true,
                       first2 + red_result.min_loc_true);
  }
}

template <class TeamHandleType, class IteratorType1, class IteratorType2>
FLARE_FUNCTION ::flare::pair<IteratorType1, IteratorType2> mismatch_team_impl(
    const TeamHandleType& teamHandle, IteratorType1 first1, IteratorType1 last1,
    IteratorType2 first2, IteratorType2 last2) {
  using value_type1 = typename IteratorType1::value_type;
  using value_type2 = typename IteratorType2::value_type;
  using pred_t      = StdAlgoEqualBinaryPredicate<value_type1, value_type2>;
  return mismatch_team_impl(teamHandle, first1, last1, first2, last2, pred_t());
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_MISMATCH_IMPL_H_
