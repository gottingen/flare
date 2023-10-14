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

#ifndef FLARE_ALGORITHM_REMOVE_IMPL_H_
#define FLARE_ALGORITHM_REMOVE_IMPL_H_

#include <flare/core.h>
#include <flare/algorithm/constraints_impl.h>
#include <flare/algorithm/helper_predicates_impl.h>
#include <flare/algorithm/distance.h>
#include <flare/algorithm/count_if.h>
#include <flare/algorithm/copy_if.h>
#include <string>

namespace flare {
namespace experimental {
namespace detail {

template <class IndexType, class FirstFrom, class FirstDest, class PredType>
struct StdRemoveIfStage1Functor {
  FirstFrom m_first_from;
  FirstDest m_first_dest;
  PredType m_must_remove;

  FLARE_FUNCTION
  StdRemoveIfStage1Functor(FirstFrom first_from, FirstDest first_dest,
                           PredType pred)
      : m_first_from(std::move(first_from)),
        m_first_dest(std::move(first_dest)),
        m_must_remove(std::move(pred)) {}

  FLARE_FUNCTION
  void operator()(const IndexType i, IndexType& update,
                  const bool final_pass) const {
    auto& myval = m_first_from[i];
    if (final_pass) {
      if (!m_must_remove(myval)) {
        // calling move here is ok because we are inside final pass
        // we are calling move assign as specified by the std
        m_first_dest[update] = std::move(myval);
      }
    }

    if (!m_must_remove(myval)) {
      update += 1;
    }
  }
};

template <class IndexType, class InputIteratorType, class OutputIteratorType>
struct StdRemoveIfStage2Functor {
  InputIteratorType m_first_from;
  OutputIteratorType m_first_to;

  FLARE_FUNCTION
  StdRemoveIfStage2Functor(InputIteratorType first_from,
                           OutputIteratorType first_to)
      : m_first_from(std::move(first_from)), m_first_to(std::move(first_to)) {}

  FLARE_FUNCTION
  void operator()(const IndexType i) const {
    m_first_to[i] = std::move(m_first_from[i]);
  }
};

template <class ExecutionSpace, class IteratorType, class UnaryPredicateType>
IteratorType remove_if_impl(const std::string& label, const ExecutionSpace& ex,
                            IteratorType first, IteratorType last,
                            UnaryPredicateType pred) {
  detail::static_assert_random_access_and_accessible(ex, first);
  detail::expect_valid_range(first, last);

  if (first == last) {
    return last;
  } else {
    // create tmp buffer to use to *move* all elements that we need to keep.
    // note that the tmp buffer is just large enought to store
    // all elements to keep, because ideally we do not need/want one
    // as large as the original range.
    // To allocate the right tmp tensor, we need a call to count_if.
    // We could just do a "safe" allocation of a buffer as
    // large as (last-first), but I think a call to count_if is more afforable.

    // count how many elements we need to keep
    // note that the elements to remove are those that meet the predicate
    const auto remove_count =
        ::flare::experimental::count_if(ex, first, last, pred);
    const auto keep_count =
        flare::experimental::distance(first, last) - remove_count;

    // create helper tmp tensor
    using value_type    = typename IteratorType::value_type;
    using tmp_tensor_type = flare::Tensor<value_type*, ExecutionSpace>;
    tmp_tensor_type tmp_tensor("std_remove_if_tmp_tensor", keep_count);
    using tmp_readwrite_iterator_type = decltype(begin(tmp_tensor));

    // in stage 1, *move* all elements to keep from original range to tmp
    // we use similar impl as copy_if except that we *move* rather than copy
    using index_type = typename IteratorType::difference_type;
    using func1_type = StdRemoveIfStage1Functor<index_type, IteratorType,
                                                tmp_readwrite_iterator_type,
                                                UnaryPredicateType>;

    const auto scan_num_elements = flare::experimental::distance(first, last);
    index_type scan_count        = 0;
    ::flare::parallel_scan(
        label, RangePolicy<ExecutionSpace>(ex, 0, scan_num_elements),
        func1_type(first, begin(tmp_tensor), pred), scan_count);

    // scan_count should be equal to keep_count
    assert(scan_count == keep_count);
    (void)scan_count;  // to avoid unused complaints

    // stage 2, we do parfor to move from tmp to original range
    using func2_type =
        StdRemoveIfStage2Functor<index_type, tmp_readwrite_iterator_type,
                                 IteratorType>;
    ::flare::parallel_for(
        "remove_if_stage2_parfor",
        RangePolicy<ExecutionSpace>(ex, 0, tmp_tensor.extent(0)),
        func2_type(begin(tmp_tensor), first));
    ex.fence("flare::remove_if: fence after stage2");

    // return
    return first + keep_count;
  }
}

template <class ExecutionSpace, class IteratorType, class ValueType>
auto remove_impl(const std::string& label, const ExecutionSpace& ex,
                 IteratorType first, IteratorType last,
                 const ValueType& value) {
  using predicate_type = StdAlgoEqualsValUnaryPredicate<ValueType>;
  return remove_if_impl(label, ex, first, last, predicate_type(value));
}

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class ValueType>
auto remove_copy_impl(const std::string& label, const ExecutionSpace& ex,
                      InputIteratorType first_from, InputIteratorType last_from,
                      OutputIteratorType first_dest, const ValueType& value) {
  // this is like copy_if except that we need to *ignore* the elements
  // that match the value, so we can solve this as follows:

  using predicate_type = StdAlgoNotEqualsValUnaryPredicate<ValueType>;
  return ::flare::experimental::copy_if(label, ex, first_from, last_from,
                                         first_dest, predicate_type(value));
}

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class UnaryPredicate>
auto remove_copy_if_impl(const std::string& label, const ExecutionSpace& ex,
                         InputIteratorType first_from,
                         InputIteratorType last_from,
                         OutputIteratorType first_dest,
                         const UnaryPredicate& pred) {
  // this is like copy_if except that we need to *ignore* the elements
  // satisfying the pred, so we can solve this as follows:

  using value_type = typename InputIteratorType::value_type;
  using pred_wrapper_type =
      StdAlgoNegateUnaryPredicateWrapper<value_type, UnaryPredicate>;
  return ::flare::experimental::copy_if(label, ex, first_from, last_from,
                                         first_dest, pred_wrapper_type(pred));
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_REMOVE_IMPL_H_
