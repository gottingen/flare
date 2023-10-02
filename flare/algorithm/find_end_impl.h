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

#ifndef FLARE_ALGORITHM_FIND_END_IMPL_H_
#define FLARE_ALGORITHM_FIND_END_IMPL_H_

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
struct StdFindEndFunctor {
  using red_value_type = typename ReducerType::value_type;

  IteratorType1 m_first;
  IteratorType1 m_last;
  IteratorType2 m_s_first;
  IteratorType2 m_s_last;
  ReducerType m_reducer;
  PredicateType m_p;

  FLARE_FUNCTION
  void operator()(const IndexType i, red_value_type& red_value) const {
    namespace KE = ::flare::experimental;
    auto myit    = m_first + i;
    bool found   = true;

    const auto search_count = KE::distance(m_s_first, m_s_last);
    for (IndexType k = 0; k < search_count; ++k) {
      // note that we add this EXPECT to check if we are in a valid range
      // but I think we can remvoe this beceause the guarantee we don't go
      // out of bounds is taken care of at the calling site
      // where we launch the par-reduce.
      FLARE_EXPECTS((myit + k) < m_last);

      if (!m_p(myit[k], m_s_first[k])) {
        found = false;
        break;
      }
    }

    // FIXME_NVHPC using a ternary operator causes problems
    red_value_type rv = {::flare::reduction_identity<IndexType>::max()};
    if (found) {
      rv.max_loc_true = i;
    }

    m_reducer.join(red_value, rv);
  }

  FLARE_FUNCTION
  StdFindEndFunctor(IteratorType1 first, IteratorType1 last,
                    IteratorType2 s_first, IteratorType2 s_last,
                    ReducerType reducer, PredicateType p)
      : m_first(std::move(first)),
        m_last(std::move(last)),
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
IteratorType1 find_end_exespace_impl(const std::string& label,
                                     const ExecutionSpace& ex,
                                     IteratorType1 first, IteratorType1 last,
                                     IteratorType2 s_first,
                                     IteratorType2 s_last,
                                     const BinaryPredicateType& pred) {
  // checks
  detail::static_assert_random_access_and_accessible(ex, first, s_first);
  detail::static_assert_iterators_have_matching_difference_type(first, s_first);
  detail::expect_valid_range(first, last);
  detail::expect_valid_range(s_first, s_last);

  // the target sequence should not be larger than the range [first, last)
  namespace KE            = ::flare::experimental;
  const auto num_elements = KE::distance(first, last);
  const auto s_count      = KE::distance(s_first, s_last);
  FLARE_EXPECTS(num_elements >= s_count);

  if (s_first == s_last) {
    return last;
  }

  if (first == last) {
    return last;
  }

  // special case where the two ranges have equal size
  if (num_elements == s_count) {
    const auto equal_result =
        equal_exespace_impl(label, ex, first, last, s_first, pred);
    return (equal_result) ? first : last;
  } else {
    using index_type           = typename IteratorType1::difference_type;
    using reducer_type         = LastLoc<index_type>;
    using reduction_value_type = typename reducer_type::value_type;
    using func_t = StdFindEndFunctor<index_type, IteratorType1, IteratorType2,
                                     reducer_type, BinaryPredicateType>;

    // run
    reduction_value_type red_result;
    reducer_type reducer(red_result);

    // decide the size of the range policy of the par_red:
    // note that the last feasible index to start looking is the index
    // whose distance from the "last" is equal to the sequence count.
    // the +1 is because we need to include that location too.
    const auto range_size = num_elements - s_count + 1;

    // run par reduce
    ::flare::parallel_reduce(
        label, RangePolicy<ExecutionSpace>(ex, 0, range_size),
        func_t(first, last, s_first, s_last, reducer, pred), reducer);

    // fence not needed because reducing into scalar

    // decide and return
    if (red_result.max_loc_true ==
        ::flare::reduction_identity<index_type>::max()) {
      // if here, a subrange has not been found
      return last;
    } else {
      // a location has been found
      return first + red_result.max_loc_true;
    }
  }
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
IteratorType1 find_end_exespace_impl(const std::string& label,
                                     const ExecutionSpace& ex,
                                     IteratorType1 first, IteratorType1 last,
                                     IteratorType2 s_first,
                                     IteratorType2 s_last) {
  using value_type1    = typename IteratorType1::value_type;
  using value_type2    = typename IteratorType2::value_type;
  using predicate_type = StdAlgoEqualBinaryPredicate<value_type1, value_type2>;
  return find_end_exespace_impl(label, ex, first, last, s_first, s_last,
                                predicate_type());
}

//
// team impl
//
template <class TeamHandleType, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
FLARE_FUNCTION IteratorType1
find_end_team_impl(const TeamHandleType& teamHandle, IteratorType1 first,
                   IteratorType1 last, IteratorType2 s_first,
                   IteratorType2 s_last, const BinaryPredicateType& pred) {
  // checks
  detail::static_assert_random_access_and_accessible(teamHandle, first, s_first);
  detail::static_assert_iterators_have_matching_difference_type(first, s_first);
  detail::expect_valid_range(first, last);
  detail::expect_valid_range(s_first, s_last);

  // the target sequence should not be larger than the range [first, last)
  namespace KE            = ::flare::experimental;
  const auto num_elements = KE::distance(first, last);
  const auto s_count      = KE::distance(s_first, s_last);
  FLARE_EXPECTS(num_elements >= s_count);

  if (s_first == s_last) {
    return last;
  }

  if (first == last) {
    return last;
  }

  // special case where the two ranges have equal size
  if (num_elements == s_count) {
    const auto equal_result =
        equal_team_impl(teamHandle, first, last, s_first, pred);
    return (equal_result) ? first : last;
  } else {
    using index_type           = typename IteratorType1::difference_type;
    using reducer_type         = LastLoc<index_type>;
    using reduction_value_type = typename reducer_type::value_type;
    using func_t = StdFindEndFunctor<index_type, IteratorType1, IteratorType2,
                                     reducer_type, BinaryPredicateType>;

    // run
    reduction_value_type red_result;
    reducer_type reducer(red_result);

    // decide the size of the range policy of the par_red:
    // note that the last feasible index to start looking is the index
    // whose distance from the "last" is equal to the sequence count.
    // the +1 is because we need to include that location too.
    const auto range_size = num_elements - s_count + 1;

    // run par reduce
    ::flare::parallel_reduce(
        TeamThreadRange(teamHandle, 0, range_size),
        func_t(first, last, s_first, s_last, reducer, pred), reducer);

    teamHandle.team_barrier();

    // decide and return
    if (red_result.max_loc_true ==
        ::flare::reduction_identity<index_type>::max()) {
      // if here, a subrange has not been found
      return last;
    } else {
      // a location has been found
      return first + red_result.max_loc_true;
    }
  }
}

template <class TeamHandleType, class IteratorType1, class IteratorType2>
FLARE_FUNCTION IteratorType1 find_end_team_impl(
    const TeamHandleType& teamHandle, IteratorType1 first, IteratorType1 last,
    IteratorType2 s_first, IteratorType2 s_last) {
  using value_type1    = typename IteratorType1::value_type;
  using value_type2    = typename IteratorType2::value_type;
  using predicate_type = StdAlgoEqualBinaryPredicate<value_type1, value_type2>;
  return find_end_team_impl(teamHandle, first, last, s_first, s_last,
                            predicate_type());
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_FIND_END_IMPL_H_
