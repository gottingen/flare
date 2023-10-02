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

#ifndef FLARE_ALGORITHM_SORT_IMPL_H_
#define FLARE_ALGORITHM_SORT_IMPL_H_

#include <flare/algorithm/bin_ops.h>
#include <flare/algorithm/bin_sort.h>
#include <flare/algorithm/begin_end.h>
#include <flare/algorithm/copy.h>
#include <flare/core.h>

#if defined(FLARE_ON_CUDA_DEVICE)

// Workaround for `Instruction 'shfl' without '.sync' is not supported on
// .target sm_70 and higher from PTX ISA version 6.4`.
// Also see https://github.com/NVIDIA/cub/pull/170.
#if !defined(CUB_USE_COOPERATIVE_GROUPS)
#define CUB_USE_COOPERATIVE_GROUPS
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"

#if defined(FLARE_COMPILER_CLANG)
// Some versions of Clang fail to compile Thrust, failing with errors like
// this:
//    <snip>/thrust/system/cuda/detail/core/agent_launcher.h:557:11:
//    error: use of undeclared identifier 'va_printf'
// The exact combination of versions for Clang and Thrust (or CUDA) for this
// failure was not investigated, however even very recent version combination
// (Clang 10.0.0 and Cuda 10.0) demonstrated failure.
//
// Defining _CubLog here locally allows us to avoid that code path, however
// disabling some debugging diagnostics
#pragma push_macro("_CubLog")
#ifdef _CubLog
#undef _CubLog
#endif
#define _CubLog
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#pragma pop_macro("_CubLog")
#else
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

#pragma GCC diagnostic pop

#endif

namespace flare {
namespace detail {

template <class ExecutionSpace>
struct better_off_calling_std_sort : std::false_type {};

#if defined FLARE_ENABLE_SERIAL
template <>
struct better_off_calling_std_sort<flare::Serial> : std::true_type {};
#endif

#if defined FLARE_ENABLE_OPENMP
template <>
struct better_off_calling_std_sort<flare::OpenMP> : std::true_type {};
#endif

#if defined FLARE_ENABLE_THREADS
template <>
struct better_off_calling_std_sort<flare::Threads> : std::true_type {};
#endif


template <class T>
inline constexpr bool better_off_calling_std_sort_v =
    better_off_calling_std_sort<T>::value;

template <class ViewType>
struct min_max_functor {
  using minmax_scalar =
      flare::MinMaxScalar<typename ViewType::non_const_value_type>;

  ViewType view;
  min_max_functor(const ViewType& view_) : view(view_) {}

  FLARE_INLINE_FUNCTION
  void operator()(const size_t& i, minmax_scalar& minmax) const {
    if (view(i) < minmax.min_val) minmax.min_val = view(i);
    if (view(i) > minmax.max_val) minmax.max_val = view(i);
  }
};

template <class ExecutionSpace, class DataType, class... Properties>
void sort_via_binsort(const ExecutionSpace& exec,
                      const flare::View<DataType, Properties...>& view) {
  // Although we are using BinSort below, which could work on rank-2 views,
  // for now view must be rank-1 because the min_max_functor
  // used below only works for rank-1 views
  using ViewType = flare::View<DataType, Properties...>;
  static_assert(ViewType::rank == 1,
                "flare::sort: currently only supports rank-1 Views.");

  if (view.extent(0) <= 1) {
    return;
  }

  flare::MinMaxScalar<typename ViewType::non_const_value_type> result;
  flare::MinMax<typename ViewType::non_const_value_type> reducer(result);
  parallel_reduce("flare::Sort::FindExtent",
                  flare::RangePolicy<typename ViewType::execution_space>(
                      exec, 0, view.extent(0)),
                  min_max_functor<ViewType>(view), reducer);
  if (result.min_val == result.max_val) return;
  // For integral types the number of bins may be larger than the range
  // in which case we can exactly have one unique value per bin
  // and then don't need to sort bins.
  bool sort_in_bins = true;
  // TODO: figure out better max_bins then this ...
  int64_t max_bins = view.extent(0) / 2;
  if (std::is_integral<typename ViewType::non_const_value_type>::value) {
    // Cast to double to avoid possible overflow when using integer
    auto const max_val = static_cast<double>(result.max_val);
    auto const min_val = static_cast<double>(result.min_val);
    // using 10M as the cutoff for special behavior (roughly 40MB for the count
    // array)
    if ((max_val - min_val) < 10000000) {
      max_bins     = max_val - min_val + 1;
      sort_in_bins = false;
    }
  }
  if (std::is_floating_point<typename ViewType::non_const_value_type>::value) {
    FLARE_ASSERT(std::isfinite(static_cast<double>(result.max_val) -
                                static_cast<double>(result.min_val)));
  }

  using CompType = BinOp1D<ViewType>;
  BinSort<ViewType, CompType> bin_sort(
      view, CompType(max_bins, result.min_val, result.max_val), sort_in_bins);
  bin_sort.create_permute_vector(exec);
  bin_sort.sort(exec, view);
}

#if defined(FLARE_ON_CUDA_DEVICE)
template <class DataType, class... Properties, class... MaybeComparator>
void sort_cudathrust(const Cuda& space,
                     const flare::View<DataType, Properties...>& view,
                     MaybeComparator&&... maybeComparator) {
  using ViewType = flare::View<DataType, Properties...>;
  static_assert(ViewType::rank == 1,
                "flare::sort: currently only supports rank-1 Views.");

  if (view.extent(0) <= 1) {
    return;
  }
  const auto exec = thrust::cuda::par.on(space.cuda_stream());
  auto first      = ::flare::experimental::begin(view);
  auto last       = ::flare::experimental::end(view);
  thrust::sort(exec, first, last,
               std::forward<MaybeComparator>(maybeComparator)...);
}
#endif

template <class ExecutionSpace, class DataType, class... Properties,
          class... MaybeComparator>
void copy_to_host_run_stdsort_copy_back(
    const ExecutionSpace& exec,
    const flare::View<DataType, Properties...>& view,
    MaybeComparator&&... maybeComparator) {
  namespace KE = ::flare::experimental;

  using ViewType = flare::View<DataType, Properties...>;
  using layout   = typename ViewType::array_layout;
  if constexpr (std::is_same_v<LayoutStride, layout>) {
    // for strided views we cannot just deep_copy from device to host,
    // so we need to do a few more jumps
    using view_value_type      = typename ViewType::non_const_value_type;
    using view_exespace        = typename ViewType::execution_space;
    using view_deep_copyable_t = flare::View<view_value_type*, view_exespace>;
    view_deep_copyable_t view_dc("view_dc", view.extent(0));
    KE::copy(exec, view, view_dc);

    // run sort on the mirror of view_dc
    auto mv_h  = create_mirror_view_and_copy(flare::HostSpace(), view_dc);
    auto first = KE::begin(mv_h);
    auto last  = KE::end(mv_h);
    std::sort(first, last, std::forward<MaybeComparator>(maybeComparator)...);
    flare::deep_copy(exec, view_dc, mv_h);

    // copy back to argument view
    KE::copy(exec, KE::cbegin(view_dc), KE::cend(view_dc), KE::begin(view));
  } else {
    auto view_h = create_mirror_view_and_copy(flare::HostSpace(), view);
    auto first  = KE::begin(view_h);
    auto last   = KE::end(view_h);
    std::sort(first, last, std::forward<MaybeComparator>(maybeComparator)...);
    flare::deep_copy(exec, view, view_h);
  }
}

// --------------------------------------------------
//
// specialize cases for sorting without comparator
//
// --------------------------------------------------

#if defined(FLARE_ON_CUDA_DEVICE)
template <class DataType, class... Properties>
void sort_device_view_without_comparator(
    const Cuda& exec, const flare::View<DataType, Properties...>& view) {
  sort_cudathrust(exec, view);
}
#endif

// fallback case
template <class ExecutionSpace, class DataType, class... Properties>
std::enable_if_t<flare::is_execution_space<ExecutionSpace>::value>
sort_device_view_without_comparator(
    const ExecutionSpace& exec,
    const flare::View<DataType, Properties...>& view) {
  sort_via_binsort(exec, view);
}

// --------------------------------------------------
//
// specialize cases for sorting with comparator
//
// --------------------------------------------------

#if defined(FLARE_ON_CUDA_DEVICE)
template <class ComparatorType, class DataType, class... Properties>
void sort_device_view_with_comparator(
    const Cuda& exec, const flare::View<DataType, Properties...>& view,
    const ComparatorType& comparator) {
  sort_cudathrust(exec, view, comparator);
}
#endif


template <class ExecutionSpace, class ComparatorType, class DataType,
          class... Properties>
std::enable_if_t<flare::is_execution_space<ExecutionSpace>::value>
sort_device_view_with_comparator(
    const ExecutionSpace& exec,
    const flare::View<DataType, Properties...>& view,
    const ComparatorType& comparator) {
  // This is a fallback case if a more specialized overload does not exist:
  // for now, this fallback copies data to host, runs std::sort
  // and then copies data back. Potentially, this can later be changed
  // with a better solution like our own quicksort on device or similar.

  using ViewType = flare::View<DataType, Properties...>;
  using MemSpace = typename ViewType::memory_space;
  static_assert(!SpaceAccessibility<HostSpace, MemSpace>::accessible,
                "detail::sort_device_view_with_comparator: should not be called "
                "on a view that is already accessible on the host");

  copy_to_host_run_stdsort_copy_back(exec, view, comparator);
}

}  // namespace detail
}  // namespace flare
#endif  // FLARE_ALGORITHM_SORT_IMPL_H_
