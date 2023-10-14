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

#ifndef FLARE_ALGORITHM_CONSTRAINTS_H_
#define FLARE_ALGORITHM_CONSTRAINTS_H_

#include <flare/core/common/detection_idiom.h>
#include <flare/core/tensor/tensor.h>

namespace flare {
namespace experimental {
namespace detail {

template <typename T, typename enable = void>
struct is_admissible_to_flare_std_algorithms : std::false_type {};

template <typename T>
struct is_admissible_to_flare_std_algorithms<
    T, std::enable_if_t< ::flare::is_tensor<T>::value && T::rank() == 1 &&
                         (std::is_same<typename T::traits::array_layout,
                                       flare::LayoutLeft>::value ||
                          std::is_same<typename T::traits::array_layout,
                                       flare::LayoutRight>::value ||
                          std::is_same<typename T::traits::array_layout,
                                       flare::LayoutStride>::value)> >
    : std::true_type {};

template <class TensorType>
FLARE_INLINE_FUNCTION constexpr void
static_assert_is_admissible_to_flare_std_algorithms(
    const TensorType& /* tensor */) {
  static_assert(is_admissible_to_flare_std_algorithms<TensorType>::value,
                "Currently, flare standard algorithms only accept 1D Tensors.");
}

//
// is_iterator
//
template <class T>
using iterator_category_t = typename T::iterator_category;

template <class T>
using is_iterator = flare::is_detected<iterator_category_t, T>;

template <class T>
inline constexpr bool is_iterator_v = is_iterator<T>::value;

//
// are_iterators
//
template <class... Args>
struct are_iterators;

template <class T>
struct are_iterators<T> {
  static constexpr bool value = is_iterator_v<T>;
};

template <class Head, class... Tail>
struct are_iterators<Head, Tail...> {
  static constexpr bool value =
      are_iterators<Head>::value && (are_iterators<Tail>::value && ... && true);
};

template <class... Ts>
inline constexpr bool are_iterators_v = are_iterators<Ts...>::value;

//
// are_random_access_iterators
//
template <class... Args>
struct are_random_access_iterators;

template <class T>
struct are_random_access_iterators<T> {
  static constexpr bool value =
      is_iterator_v<T> && std::is_base_of<std::random_access_iterator_tag,
                                          typename T::iterator_category>::value;
};

template <class Head, class... Tail>
struct are_random_access_iterators<Head, Tail...> {
  static constexpr bool value =
      are_random_access_iterators<Head>::value &&
      (are_random_access_iterators<Tail>::value && ... && true);
};

template <class... Ts>
inline constexpr bool are_random_access_iterators_v =
    are_random_access_iterators<Ts...>::value;

//
// iterators_are_accessible_from
//
template <class... Args>
struct iterators_are_accessible_from;

template <class ExeSpace, class IteratorType>
struct iterators_are_accessible_from<ExeSpace, IteratorType> {
  using tensor_type = typename IteratorType::tensor_type;
  static constexpr bool value =
      SpaceAccessibility<ExeSpace,
                         typename tensor_type::memory_space>::accessible;
};

template <class ExeSpace, class Head, class... Tail>
struct iterators_are_accessible_from<ExeSpace, Head, Tail...> {
  static constexpr bool value =
      iterators_are_accessible_from<ExeSpace, Head>::value &&
      iterators_are_accessible_from<ExeSpace, Tail...>::value;
};

template <class ExecutionSpaceOrTeamHandleType, class... IteratorTypes>
FLARE_INLINE_FUNCTION constexpr void
static_assert_random_access_and_accessible(
    const ExecutionSpaceOrTeamHandleType& /* ex_or_th*/,
    IteratorTypes... /* iterators */) {
  static_assert(
      are_random_access_iterators<IteratorTypes...>::value,
      "Currently, flare standard algorithms require random access iterators.");
  static_assert(iterators_are_accessible_from<
                    typename ExecutionSpaceOrTeamHandleType::execution_space,
                    IteratorTypes...>::value,
                "Incompatible tensor/iterator and execution space");
}

//
// have matching difference_type
//
template <class... Args>
struct iterators_have_matching_difference_type;

template <class T>
struct iterators_have_matching_difference_type<T> {
  static constexpr bool value = true;
};

template <class T1, class T2>
struct iterators_have_matching_difference_type<T1, T2> {
  static constexpr bool value =
      std::is_same<typename T1::difference_type,
                   typename T2::difference_type>::value;
};

template <class T1, class T2, class... Tail>
struct iterators_have_matching_difference_type<T1, T2, Tail...> {
  static constexpr bool value =
      iterators_have_matching_difference_type<T1, T2>::value &&
      iterators_have_matching_difference_type<T2, Tail...>::value;
};

template <class IteratorType1, class IteratorType2>
FLARE_INLINE_FUNCTION constexpr void
static_assert_iterators_have_matching_difference_type(IteratorType1 /* it1 */,
                                                      IteratorType2 /* it2 */) {
  static_assert(iterators_have_matching_difference_type<IteratorType1,
                                                        IteratorType2>::value,
                "Iterators do not have matching difference_type");
}

template <class IteratorType1, class IteratorType2, class IteratorType3>
FLARE_INLINE_FUNCTION constexpr void
static_assert_iterators_have_matching_difference_type(IteratorType1 it1,
                                                      IteratorType2 it2,
                                                      IteratorType3 it3) {
  static_assert_iterators_have_matching_difference_type(it1, it2);
  static_assert_iterators_have_matching_difference_type(it2, it3);
}

//
// valid range
//
template <class IteratorType>
FLARE_INLINE_FUNCTION void expect_valid_range(IteratorType first,
                                               IteratorType last) {
  // this is a no-op for release
  FLARE_EXPECTS(last >= first);
  // avoid compiler complaining when FLARE_EXPECTS is no-op
  (void)first;
  (void)last;
}

}  // namespace detail
}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_CONSTRAINTS_H_
