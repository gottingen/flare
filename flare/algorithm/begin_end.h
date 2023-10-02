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

#ifndef FLARE_ALGORITHM_BEGIN_END_H_
#define FLARE_ALGORITHM_BEGIN_END_H_

#include <flare/core/tensor/view.h>
#include <flare/algorithm/random_access_iterator_impl.h>
#include <flare/algorithm/constraints_impl.h>

/// \file begin_end.h
/// \brief flare begin, end, cbegin, cend

namespace flare {
namespace experimental {

template <class DataType, class... Properties>
FLARE_INLINE_FUNCTION auto begin(
    const flare::View<DataType, Properties...>& v) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  using it_t =
      detail::RandomAccessIterator<flare::View<DataType, Properties...>>;
  return it_t(v);
}

template <class DataType, class... Properties>
FLARE_INLINE_FUNCTION auto end(
    const flare::View<DataType, Properties...>& v) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  using it_t =
      detail::RandomAccessIterator<flare::View<DataType, Properties...>>;
  return it_t(v, v.extent(0));
}

template <class DataType, class... Properties>
FLARE_INLINE_FUNCTION auto cbegin(
    const flare::View<DataType, Properties...>& v) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  using ViewConstType =
      typename flare::View<DataType, Properties...>::const_type;
  const ViewConstType cv = v;
  using it_t             = detail::RandomAccessIterator<ViewConstType>;
  return it_t(cv);
}

template <class DataType, class... Properties>
FLARE_INLINE_FUNCTION auto cend(
    const flare::View<DataType, Properties...>& v) {
  detail::static_assert_is_admissible_to_flare_std_algorithms(v);

  using ViewConstType =
      typename flare::View<DataType, Properties...>::const_type;
  const ViewConstType cv = v;
  using it_t             = detail::RandomAccessIterator<ViewConstType>;
  return it_t(cv, cv.extent(0));
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ALGORITHM_BEGIN_END_H_
