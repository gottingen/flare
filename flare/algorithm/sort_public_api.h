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

#ifndef FLARE_ALGORITHM_SORT_PUBLIC_API_H_
#define FLARE_ALGORITHM_SORT_PUBLIC_API_H_

#include <flare/algorithm/sort_impl.h>
#include <flare/algorithm/begin_end.h>
#include <flare/core.h>
#include <algorithm>

namespace flare {

// ---------------------------------------------------------------
// basic overloads
// ---------------------------------------------------------------

template <class ExecutionSpace, class DataType, class... Properties>
void sort([[maybe_unused]] const ExecutionSpace& exec,
          const flare::View<DataType, Properties...>& view) {
  // constraints
  using ViewType = flare::View<DataType, Properties...>;
  using MemSpace = typename ViewType::memory_space;
  static_assert(
      ViewType::rank == 1 &&
          (std::is_same_v<typename ViewType::array_layout, LayoutRight> ||
           std::is_same_v<typename ViewType::array_layout, LayoutLeft> ||
           std::is_same_v<typename ViewType::array_layout, LayoutStride>),
      "flare::sort without comparator: supports 1D Views with LayoutRight, "
      "LayoutLeft or LayoutStride.");

  static_assert(SpaceAccessibility<ExecutionSpace, MemSpace>::accessible,
                "flare::sort: execution space instance is not able to access "
                "the memory space of the "
                "View argument!");

  if (view.extent(0) <= 1) {
    return;
  }

  if constexpr (detail::better_off_calling_std_sort_v<ExecutionSpace>) {
    auto first = ::flare::experimental::begin(view);
    auto last  = ::flare::experimental::end(view);
    std::sort(first, last);
  } else {
    detail::sort_device_view_without_comparator(exec, view);
  }
}

template <class DataType, class... Properties>
void sort(const flare::View<DataType, Properties...>& view) {
  using ViewType = flare::View<DataType, Properties...>;
  static_assert(ViewType::rank == 1,
                "flare::sort: currently only supports rank-1 Views.");

  flare::fence("flare::sort: before");

  if (view.extent(0) <= 1) {
    return;
  }

  typename ViewType::execution_space exec;
  sort(exec, view);
  exec.fence("flare::sort: fence after sorting");
}

// ---------------------------------------------------------------
// overloads supporting a custom comparator
// ---------------------------------------------------------------
template <class ExecutionSpace, class ComparatorType, class DataType,
          class... Properties>
void sort([[maybe_unused]] const ExecutionSpace& exec,
          const flare::View<DataType, Properties...>& view,
          const ComparatorType& comparator) {
  // constraints
  using ViewType = flare::View<DataType, Properties...>;
  using MemSpace = typename ViewType::memory_space;
  static_assert(
      ViewType::rank == 1 &&
          (std::is_same_v<typename ViewType::array_layout, LayoutRight> ||
           std::is_same_v<typename ViewType::array_layout, LayoutLeft> ||
           std::is_same_v<typename ViewType::array_layout, LayoutStride>),
      "flare::sort with comparator: supports 1D Views with LayoutRight, "
      "LayoutLeft or LayoutStride.");

  static_assert(SpaceAccessibility<ExecutionSpace, MemSpace>::accessible,
                "flare::sort: execution space instance is not able to access "
                "the memory space of the View argument!");

  if (view.extent(0) <= 1) {
    return;
  }

  if constexpr (detail::better_off_calling_std_sort_v<ExecutionSpace>) {
    auto first = ::flare::experimental::begin(view);
    auto last  = ::flare::experimental::end(view);
    std::sort(first, last, comparator);
  } else {
    detail::sort_device_view_with_comparator(exec, view, comparator);
  }
}

template <class ComparatorType, class DataType, class... Properties>
void sort(const flare::View<DataType, Properties...>& view,
          const ComparatorType& comparator) {
  using ViewType = flare::View<DataType, Properties...>;
  static_assert(
      ViewType::rank == 1 &&
          (std::is_same_v<typename ViewType::array_layout, LayoutRight> ||
           std::is_same_v<typename ViewType::array_layout, LayoutLeft> ||
           std::is_same_v<typename ViewType::array_layout, LayoutStride>),
      "flare::sort with comparator: supports 1D Views with LayoutRight, "
      "LayoutLeft or LayoutStride.");

  flare::fence("flare::sort with comparator: before");

  if (view.extent(0) <= 1) {
    return;
  }

  typename ViewType::execution_space exec;
  sort(exec, view, comparator);
  exec.fence("flare::sort with comparator: fence after sorting");
}

// ---------------------------------------------------------------
// overloads for sorting a view with a subrange
// specified via integers begin, end
// ---------------------------------------------------------------

template <class ExecutionSpace, class ViewType>
std::enable_if_t<flare::is_execution_space<ExecutionSpace>::value> sort(
    const ExecutionSpace& exec, ViewType view, size_t const begin,
    size_t const end) {
  // view must be rank-1 because the detail::min_max_functor
  // used below only works for rank-1 views for now
  static_assert(ViewType::rank == 1,
                "flare::sort: currently only supports rank-1 Views.");

  if (view.extent(0) <= 1) {
    return;
  }

  using range_policy = flare::RangePolicy<typename ViewType::execution_space>;
  using CompType     = BinOp1D<ViewType>;

  flare::MinMaxScalar<typename ViewType::non_const_value_type> result;
  flare::MinMax<typename ViewType::non_const_value_type> reducer(result);

  parallel_reduce("flare::Sort::FindExtent", range_policy(exec, begin, end),
                  detail::min_max_functor<ViewType>(view), reducer);

  if (result.min_val == result.max_val) return;

  BinSort<ViewType, CompType> bin_sort(
      exec, view, begin, end,
      CompType((end - begin) / 2, result.min_val, result.max_val), true);

  bin_sort.create_permute_vector(exec);
  bin_sort.sort(exec, view, begin, end);
}

template <class ViewType>
void sort(ViewType view, size_t const begin, size_t const end) {
  // same constraints as the overload above which this gets dispatched to
  static_assert(ViewType::rank == 1,
                "flare::sort: currently only supports rank-1 Views.");

  flare::fence("flare::sort: before");

  if (view.extent(0) <= 1) {
    return;
  }

  typename ViewType::execution_space exec;
  sort(exec, view, begin, end);
  exec.fence("flare::Sort: fence after sorting");
}

}  // namespace flare
#endif  // FLARE_ALGORITHM_SORT_PUBLIC_API_H_
