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

#ifndef FLARE_ALGORITHM_FIND_FIRST_OF_IMPL_H_
#define FLARE_ALGORITHM_FIND_FIRST_OF_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IndexType, class IteratorType1, class IteratorType2,
          class ReducerType, class PredicateType>
struct StdFindFirstOfFunctor {
  using red_value_type = typename ReducerType::value_type;

  IteratorType1 m_first;
  IteratorType2 m_s_first;
  IteratorType2 m_s_last;
  ReducerType m_reducer;
  PredicateType m_p;

  FLARE_FUNCTION
  void operator()(const IndexType i, red_value_type& red_value) const {
    namespace KE        = ::flare::experimental;
    const auto& myvalue = m_first[i];
    bool found          = false;

    const auto search_count = KE::distance(m_s_first, m_s_last);
    for (IndexType k = 0; k < search_count; ++k) {
      if (m_p(myvalue, m_s_first[k])) {
        found = true;
        break;
      }
    }

    // FIXME_NVHPC using a ternary operator causes problems
    red_value_type rv = {::flare::reduction_identity<IndexType>::min()};
    if (found) {
      rv.min_loc_true = i;
    }
    m_reducer.join(red_value, rv);
  }

  FLARE_FUNCTION
  StdFindFirstOfFunctor(IteratorType1 first, IteratorType2 s_first,
                        IteratorType2 s_last, ReducerType reducer,
                        PredicateType p)
      : m_first(std::move(first)),
        m_s_first(std::move(s_first)),
        m_s_last(std::move(s_last)),
        m_reducer(std::move(reducer)),
        m_p(std::move(p)) {}
};

//
// exespace impl
//
template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
IteratorType1 find_first_of_exespace_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType1 first,
    IteratorType1 last, IteratorType2 s_first, IteratorType2 s_last,
    const BinaryPredicateType& pred) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first, s_first);
  detail::static_assert_iterators_have_matching_difference_type(first, s_first);
  detail::expect_valid_range(first, last);
  detail::expect_valid_range(s_first, s_last);

  if ((s_first == s_last) || (first == last)) {
    return last;
  }

  using index_type           = typename IteratorType1::difference_type;
  using reducer_type         = FirstLoc<index_type>;
  using reduction_value_type = typename reducer_type::value_type;
  using func_t = StdFindFirstOfFunctor<index_type, IteratorType1, IteratorType2,
                                       reducer_type, BinaryPredicateType>;

  // run
  reduction_value_type red_result;
  reducer_type reducer(red_result);
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_reduce(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements),
      func_t(first, s_first, s_last, reducer, pred), reducer);

  // fence not needed because reducing into scalar

  // decide and return
  if (red_result.min_loc_true ==
      ::flare::reduction_identity<index_type>::min()) {
    // if here, nothing found
    return last;
  } else {
    // a location has been found
    return first + red_result.min_loc_true;
  }
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
IteratorType1 find_first_of_exespace_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType1 first,
    IteratorType1 last, IteratorType2 s_first, IteratorType2 s_last) {
  using value_type1    = typename IteratorType1::value_type;
  using value_type2    = typename IteratorType2::value_type;
  using predicate_type = StdAlgoEqualBinaryPredicate<value_type1, value_type2>;
  return find_first_of_exespace_impl(label, ex, first, last, s_first, s_last,
                                     predicate_type());
}

//
// team impl
//
template <class TeamHandleType, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
FLARE_FUNCTION IteratorType1
find_first_of_team_impl(const TeamHandleType& teamHandle, IteratorType1 first,
                        IteratorType1 last, IteratorType2 s_first,
                        IteratorType2 s_last, const BinaryPredicateType& pred) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first, s_first);
  detail::static_assert_iterators_have_matching_difference_type(first, s_first);
  detail::expect_valid_range(first, last);
  detail::expect_valid_range(s_first, s_last);

  if ((s_first == s_last) || (first == last)) {
    return last;
  }

  using index_type           = typename IteratorType1::difference_type;
  using reducer_type         = FirstLoc<index_type>;
  using reduction_value_type = typename reducer_type::value_type;
  using func_t = StdFindFirstOfFunctor<index_type, IteratorType1, IteratorType2,
                                       reducer_type, BinaryPredicateType>;

  // run
  reduction_value_type red_result;
  reducer_type reducer(red_result);
  const auto num_elements = flare::experimental::distance(first, last);
  ::flare::parallel_reduce(TeamThreadRange(teamHandle, 0, num_elements),
                            func_t(first, s_first, s_last, reducer, pred),
                            reducer);

  teamHandle.team_barrier();

  // decide and return
  if (red_result.min_loc_true ==
      ::flare::reduction_identity<index_type>::min()) {
    // if here, nothing found
    return last;
  } else {
    // a location has been found
    return first + red_result.min_loc_true;
  }
}

template <class TeamHandleType, class IteratorType1, class IteratorType2>
FLARE_FUNCTION IteratorType1 find_first_of_team_impl(
    const TeamHandleType& teamHandle, IteratorType1 first, IteratorType1 last,
    IteratorType2 s_first, IteratorType2 s_last) {
  using value_type1    = typename IteratorType1::value_type;
  using value_type2    = typename IteratorType2::value_type;
  using predicate_type = StdAlgoEqualBinaryPredicate<value_type1, value_type2>;
  return find_first_of_team_impl(teamHandle, first, last, s_first, s_last,
                                 predicate_type());
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_FIND_FIRST_OF_IMPL_H_
