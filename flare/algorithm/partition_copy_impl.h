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

#ifndef FLARE_ALGORITHM_PARTITION_COPY_IMPL_H_
#define FLARE_ALGORITHM_PARTITION_COPY_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class ValueType>
struct StdPartitionCopyScalar {
  ValueType true_count_;
  ValueType false_count_;
};

template <class IndexType, class FirstFrom, class FirstDestTrue,
          class FirstDestFalse, class PredType>
struct StdPartitionCopyFunctor {
  using value_type = StdPartitionCopyScalar<IndexType>;

  FirstFrom m_first_from;
  FirstDestTrue m_first_dest_true;
  FirstDestFalse m_first_dest_false;
  PredType m_pred;

  FLARE_FUNCTION
  StdPartitionCopyFunctor(FirstFrom first_from, FirstDestTrue first_dest_true,
                          FirstDestFalse first_dest_false, PredType pred)
      : m_first_from(std::move(first_from)),
        m_first_dest_true(std::move(first_dest_true)),
        m_first_dest_false(std::move(first_dest_false)),
        m_pred(std::move(pred)) {}

  FLARE_FUNCTION
  void operator()(const IndexType i, value_type& update,
                  const bool final_pass) const {
    const auto& myval = m_first_from[i];
    if (final_pass) {
      if (m_pred(myval)) {
        m_first_dest_true[update.true_count_] = myval;
      } else {
        m_first_dest_false[update.false_count_] = myval;
      }
    }

    if (m_pred(myval)) {
      update.true_count_ += 1;
    } else {
      update.false_count_ += 1;
    }
  }

  FLARE_FUNCTION
  void init(value_type& update) const {
    update.true_count_  = 0;
    update.false_count_ = 0;
  }

  FLARE_FUNCTION
  void join(value_type& update, const value_type& input) const {
    update.true_count_ += input.true_count_;
    update.false_count_ += input.false_count_;
  }
};

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorTrueType, class OutputIteratorFalseType,
          class PredicateType>
::flare::pair<OutputIteratorTrueType, OutputIteratorFalseType>
partition_copy_impl(const std::string& label, const ExecutionSpace& ex,
                    InputIteratorType from_first, InputIteratorType from_last,
                    OutputIteratorTrueType to_first_true,
                    OutputIteratorFalseType to_first_false,
                    PredicateType pred) {
  // impl uses a scan, this is similar how we implemented copy_if

  // checks
  detail::static_assert_random_access_and_accessible(
      ex, from_first, to_first_true, to_first_false);
  detail::static_assert_iterators_have_matching_difference_type(
      from_first, to_first_true, to_first_false);
  detail::expect_valid_range(from_first, from_last);

  if (from_first == from_last) {
    return {to_first_true, to_first_false};
  }

  // aliases
  using index_type = typename InputIteratorType::difference_type;
  using func_type =
      StdPartitionCopyFunctor<index_type, InputIteratorType,
                              OutputIteratorTrueType, OutputIteratorFalseType,
                              PredicateType>;

  // run
  const auto num_elements =
      flare::experimental::distance(from_first, from_last);
  typename func_type::value_type counts{0, 0};
  ::flare::parallel_scan(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements),
      func_type(from_first, to_first_true, to_first_false, pred), counts);

  // fence not needed here because of the scan into counts

  return {to_first_true + counts.true_count_,
          to_first_false + counts.false_count_};
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_PARTITION_COPY_IMPL_H_
