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

#ifndef FLARE_ALGORITHM_SEARCH_N_IMPL_H_
#define FLARE_ALGORITHM_SEARCH_N_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/all_of_any_of_none_of_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IndexType, class IteratorType, class SizeType, class ValueType,
          class ReducerType, class PredicateType>
struct StdSearchNFunctor {
  using red_value_type = typename ReducerType::value_type;

  IteratorType m_first;
  IteratorType m_last;
  SizeType m_count;
  ValueType m_value;
  ReducerType m_reducer;
  PredicateType m_p;

  FLARE_FUNCTION
  void operator()(const IndexType i, red_value_type& red_value) const {
    namespace KE = ::flare::experimental;
    auto myit    = m_first + i;
    bool found   = true;

    for (SizeType k = 0; k < m_count; ++k) {
      // note that we add this EXPECT to check if we are in a valid range
      // but I think we can remove this beceause the guarantee we don't go
      // out of bounds is taken care of at the calling site
      // where we launch the par-reduce.
      FLARE_EXPECTS((myit + k) < m_last);

      if (!m_p(myit[k], m_value)) {
        found = false;
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
  StdSearchNFunctor(IteratorType first, IteratorType last, SizeType count,
                    ValueType value, ReducerType reducer, PredicateType p)
      : m_first(std::move(first)),
        m_last(std::move(last)),
        m_count(std::move(count)),
        m_value(std::move(value)),
        m_reducer(std::move(reducer)),
        m_p(std::move(p)) {}
};

//
// exespace impl
//
template <class ExecutionSpace, class IteratorType, class SizeType,
          class ValueType, class BinaryPredicateType>
IteratorType search_n_exespace_impl(const std::string& label,
                                    const ExecutionSpace& ex,
                                    IteratorType first, IteratorType last,
                                    SizeType count, const ValueType& value,
                                    const BinaryPredicateType& pred) {
  // checks
  static_assert_random_access_and_accessible(ex, first);
  expect_valid_range(first, last);
  FLARE_EXPECTS((std::ptrdiff_t)count >= 0);

  // count should not be larger than the range [first, last)
  namespace KE            = ::flare::experimental;
  const auto num_elements = KE::distance(first, last);
  // cast things to avoid compiler warning
  FLARE_EXPECTS((std::size_t)num_elements >= (std::size_t)count);

  if (first == last) {
    return first;
  }

  // special case where num elements in [first, last) == count
  if ((std::size_t)num_elements == (std::size_t)count) {
    using equal_to_value = StdAlgoEqualsValUnaryPredicate<ValueType>;
    const auto satisfies =
        all_of_exespace_impl(label, ex, first, last, equal_to_value(value));
    return (satisfies) ? first : last;
  } else {
    // aliases
    using index_type           = typename IteratorType::difference_type;
    using reducer_type         = FirstLoc<index_type>;
    using reduction_value_type = typename reducer_type::value_type;
    using func_t =
        StdSearchNFunctor<index_type, IteratorType, SizeType, ValueType,
                          reducer_type, BinaryPredicateType>;

    // run
    reduction_value_type red_result;
    reducer_type reducer(red_result);

    // decide the size of the range policy of the par_red:
    // the last feasible index to start looking is the index
    // whose distance from the "last" is equal to count.
    // the +1 is because we need to include that location too.
    const auto range_size = num_elements - count + 1;

    // run par reduce
    ::flare::parallel_reduce(
        label, RangePolicy<ExecutionSpace>(ex, 0, range_size),
        func_t(first, last, count, value, reducer, pred), reducer);

    // fence not needed because reducing into scalar

    // decide and return
    if (red_result.min_loc_true ==
        ::flare::reduction_identity<index_type>::min()) {
      // location has not been found
      return last;
    } else {
      // location has been found
      return first + red_result.min_loc_true;
    }
  }
}

template <class ExecutionSpace, class IteratorType, class SizeType,
          class ValueType>
IteratorType search_n_exespace_impl(const std::string& label,
                                    const ExecutionSpace& ex,
                                    IteratorType first, IteratorType last,
                                    SizeType count, const ValueType& value) {
  using iter_value_type = typename IteratorType::value_type;
  using predicate_type =
      StdAlgoEqualBinaryPredicate<iter_value_type, ValueType>;

  /* above we use <iter_value_type, ValueType> for the predicate_type
     to be consistent with the standard, which says:

     "
     The signature of the predicate function should be equivalent to:

        bool pred(const Type1 &a, const Type2 &b);

     The type Type1 must be such that an object of type ForwardIt can be
     dereferenced and then implicitly converted to Type1. The type Type2 must be
     such that an object of type T can be implicitly converted to Type2.
     "

     In our case, IteratorType = ForwardIt, and ValueType = T.
   */

  return search_n_exespace_impl(label, ex, first, last, count, value,
                                predicate_type());
}

//
// team impl
//
template <class TeamHandleType, class IteratorType, class SizeType,
          class ValueType, class BinaryPredicateType>
FLARE_FUNCTION IteratorType search_n_team_impl(
    const TeamHandleType& teamHandle, IteratorType first, IteratorType last,
    SizeType count, const ValueType& value, const BinaryPredicateType& pred) {
  // checks
  static_assert_random_access_and_accessible(teamHandle, first);
  expect_valid_range(first, last);
  FLARE_EXPECTS((std::ptrdiff_t)count >= 0);

  // count should not be larger than the range [first, last)
  namespace KE            = ::flare::experimental;
  const auto num_elements = KE::distance(first, last);
  // cast things to avoid compiler warning
  FLARE_EXPECTS((std::size_t)num_elements >= (std::size_t)count);

  if (first == last) {
    return first;
  }

  // special case where num elements in [first, last) == count
  if ((std::size_t)num_elements == (std::size_t)count) {
    using equal_to_value = StdAlgoEqualsValUnaryPredicate<ValueType>;
    const auto satisfies =
        all_of_team_impl(teamHandle, first, last, equal_to_value(value));
    return (satisfies) ? first : last;
  } else {
    // aliases
    using index_type           = typename IteratorType::difference_type;
    using reducer_type         = FirstLoc<index_type>;
    using reduction_value_type = typename reducer_type::value_type;
    using func_t =
        StdSearchNFunctor<index_type, IteratorType, SizeType, ValueType,
                          reducer_type, BinaryPredicateType>;

    // run
    reduction_value_type red_result;
    reducer_type reducer(red_result);

    // decide the size of the range policy of the par_red:
    // the last feasible index to start looking is the index
    // whose distance from the "last" is equal to count.
    // the +1 is because we need to include that location too.
    const auto range_size = num_elements - count + 1;

    // run par reduce
    ::flare::parallel_reduce(TeamThreadRange(teamHandle, 0, range_size),
                              func_t(first, last, count, value, reducer, pred),
                              reducer);

    teamHandle.team_barrier();

    // decide and return
    if (red_result.min_loc_true ==
        ::flare::reduction_identity<index_type>::min()) {
      // location has not been found
      return last;
    } else {
      // location has been found
      return first + red_result.min_loc_true;
    }
  }
}

template <class TeamHandleType, class IteratorType, class SizeType,
          class ValueType>
FLARE_FUNCTION IteratorType
search_n_team_impl(const TeamHandleType& teamHandle, IteratorType first,
                   IteratorType last, SizeType count, const ValueType& value) {
  using iter_value_type = typename IteratorType::value_type;
  using predicate_type =
      StdAlgoEqualBinaryPredicate<iter_value_type, ValueType>;

  /* above we use <iter_value_type, ValueType> for the predicate_type
     to be consistent with the standard, which says:

     "
     The signature of the predicate function should be equivalent to:

        bool pred(const Type1 &a, const Type2 &b);

     The type Type1 must be such that an object of type ForwardIt can be
     dereferenced and then implicitly converted to Type1. The type Type2 must be
     such that an object of type T can be implicitly converted to Type2.
     "

     In our case, IteratorType = ForwardIt, and ValueType = T.
   */

  return search_n_team_impl(teamHandle, first, last, count, value,
                            predicate_type());
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_SEARCH_N_IMPL_H_
