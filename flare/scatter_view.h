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

/// \file scatter_view.h
/// \brief Declaration and definition of flare::ScatterView.
///
/// This header file declares and defines flare::ScatterView and its
/// related nonmember functions.

#ifndef FLARE_SCATTER_VIEW_H_
#define FLARE_SCATTER_VIEW_H_
#include <flare/core.h>
#include <utility>

namespace flare {
namespace experimental {

/*
 * Reduction Type list
 *  - These corresponds to subset of the reducers in parallel_reduce
 *  - See Implementations of ScatterValue for details.
 */
struct ScatterSum {};
struct ScatterProd {};
struct ScatterMax {};
struct ScatterMin {};

struct ScatterNonDuplicated {};
struct ScatterDuplicated {};

struct ScatterNonAtomic {};
struct ScatterAtomic {};

}  // namespace experimental
}  // namespace flare

namespace flare {
namespace detail {
namespace experimental {

template <typename ExecSpace>
struct DefaultDuplication;

template <typename ExecSpace, typename Duplication>
struct DefaultContribution;

#ifdef FLARE_ENABLE_SERIAL
template <>
struct DefaultDuplication<flare::Serial> {
  using type = flare::experimental::ScatterNonDuplicated;
};

template <>
struct DefaultContribution<flare::Serial,
                           flare::experimental::ScatterNonDuplicated> {
  using type = flare::experimental::ScatterNonAtomic;
};
template <>
struct DefaultContribution<flare::Serial,
                           flare::experimental::ScatterDuplicated> {
  using type = flare::experimental::ScatterNonAtomic;
};
#endif

#ifdef FLARE_ENABLE_OPENMP
template <>
struct DefaultDuplication<flare::OpenMP> {
  using type = flare::experimental::ScatterDuplicated;
};
template <>
struct DefaultContribution<flare::OpenMP,
                           flare::experimental::ScatterNonDuplicated> {
  using type = flare::experimental::ScatterAtomic;
};
template <>
struct DefaultContribution<flare::OpenMP,
                           flare::experimental::ScatterDuplicated> {
  using type = flare::experimental::ScatterNonAtomic;
};
#endif

#ifdef FLARE_ENABLE_THREADS
template <>
struct DefaultDuplication<flare::Threads> {
  using type = flare::experimental::ScatterDuplicated;
};
template <>
struct DefaultContribution<flare::Threads,
                           flare::experimental::ScatterNonDuplicated> {
  using type = flare::experimental::ScatterAtomic;
};
template <>
struct DefaultContribution<flare::Threads,
                           flare::experimental::ScatterDuplicated> {
  using type = flare::experimental::ScatterNonAtomic;
};
#endif

#ifdef FLARE_ON_CUDA_DEVICE
template <>
struct DefaultDuplication<flare::Cuda> {
  using type = flare::experimental::ScatterNonDuplicated;
};
template <>
struct DefaultContribution<flare::Cuda,
                           flare::experimental::ScatterNonDuplicated> {
  using type = flare::experimental::ScatterAtomic;
};
template <>
struct DefaultContribution<flare::Cuda,
                           flare::experimental::ScatterDuplicated> {
  using type = flare::experimental::ScatterAtomic;
};
#endif


// FIXME All these scatter values need overhaul:
//   - like should they be copyable at all?
//   - what is the internal handle type
//   - remove join
//   - consistently use the update function in operators
template <typename ValueType, typename Op, typename DeviceType,
          typename Contribution>
struct ScatterValue;

/* ScatterValue <Op=ScatterSum, Contribution=ScatterNonAtomic> is
   the object returned by the access operator() of ScatterAccess. This class
   inherits from the Sum<> reducer and it wraps join(dest, src) with convenient
   operator+=, etc. Note the addition of update(ValueType const& rhs) and
   reset()  so that all reducers can have common functions See ReduceDuplicates
   and ResetDuplicates ) */
template <typename ValueType, typename DeviceType>
struct ScatterValue<ValueType, flare::experimental::ScatterSum, DeviceType,
                    flare::experimental::ScatterNonAtomic> {
  ValueType& value;

 public:
  FLARE_FORCEINLINE_FUNCTION ScatterValue(ValueType& value_in)
      : value(value_in) {}
  FLARE_FORCEINLINE_FUNCTION ScatterValue(ScatterValue&& other)
      : value(other.value) {}
  FLARE_FORCEINLINE_FUNCTION void operator+=(ValueType const& rhs) {
    update(rhs);
  }
  FLARE_FORCEINLINE_FUNCTION void operator++() { update(1); }
  FLARE_FORCEINLINE_FUNCTION void operator++(int) { update(1); }
  FLARE_FORCEINLINE_FUNCTION void operator-=(ValueType const& rhs) {
    update(ValueType(-rhs));
  }
  FLARE_FORCEINLINE_FUNCTION void operator--() { update(ValueType(-1)); }
  FLARE_FORCEINLINE_FUNCTION void operator--(int) { update(ValueType(-1)); }
  FLARE_FORCEINLINE_FUNCTION void update(ValueType const& rhs) {
    value += rhs;
  }
  FLARE_FORCEINLINE_FUNCTION void reset() {
    value = reduction_identity<ValueType>::sum();
  }
};

/* ScatterValue <Op=ScatterSum, Contribution=ScatterAtomic> is the
 object returned by the access operator() of ScatterAccess. This class inherits
 from the Sum<> reducer, and similar to that returned by an Atomic View, it
 wraps flare::atomic_add with convenient operator+=, etc. This version also has
 the update(rhs) and reset() functions. */
template <typename ValueType, typename DeviceType>
struct ScatterValue<ValueType, flare::experimental::ScatterSum, DeviceType,
                    flare::experimental::ScatterAtomic> {
  ValueType& value;

 public:
  FLARE_FORCEINLINE_FUNCTION ScatterValue(ValueType& value_in)
      : value(value_in) {}

  FLARE_FORCEINLINE_FUNCTION void operator+=(ValueType const& rhs) {
    this->join(value, rhs);
  }
  FLARE_FORCEINLINE_FUNCTION void operator++() { this->join(value, 1); }
  FLARE_FORCEINLINE_FUNCTION void operator++(int) { this->join(value, 1); }
  FLARE_FORCEINLINE_FUNCTION void operator-=(ValueType const& rhs) {
    this->join(value, ValueType(-rhs));
  }
  FLARE_FORCEINLINE_FUNCTION void operator--() {
    this->join(value, ValueType(-1));
  }
  FLARE_FORCEINLINE_FUNCTION void operator--(int) {
    this->join(value, ValueType(-1));
  }

  FLARE_INLINE_FUNCTION
  void join(ValueType& dest, const ValueType& src) const {
    flare::atomic_add(&dest, src);
  }

  FLARE_FORCEINLINE_FUNCTION void update(ValueType const& rhs) {
    this->join(value, rhs);
  }

  FLARE_FORCEINLINE_FUNCTION void reset() {
    value = reduction_identity<ValueType>::sum();
  }
};

/* ScatterValue <Op=ScatterProd, Contribution=ScatterNonAtomic> is
   the object returned by the access operator() of ScatterAccess.  This class
   inherits from the Prod<> reducer, and it wraps join(dest, src) with
   convenient operator*=, etc. Note the addition of update(ValueType const& rhs)
   and reset()  so that all reducers can have common functions See
   ReduceDuplicates and ResetDuplicates ) */
template <typename ValueType, typename DeviceType>
struct ScatterValue<ValueType, flare::experimental::ScatterProd, DeviceType,
                    flare::experimental::ScatterNonAtomic> {
  ValueType& value;

 public:
  FLARE_FORCEINLINE_FUNCTION ScatterValue(ValueType& value_in)
      : value(value_in) {}
  FLARE_FORCEINLINE_FUNCTION ScatterValue(ScatterValue&& other)
      : value(other.value) {}
  FLARE_FORCEINLINE_FUNCTION void operator*=(ValueType const& rhs) {
    value *= rhs;
  }
  FLARE_FORCEINLINE_FUNCTION void operator/=(ValueType const& rhs) {
    value /= rhs;
  }

  FLARE_FORCEINLINE_FUNCTION void update(ValueType const& rhs) {
    value *= rhs;
  }
  FLARE_FORCEINLINE_FUNCTION void reset() {
    value = reduction_identity<ValueType>::prod();
  }
};

/* ScatterValue <Op=ScatterProd, Contribution=ScatterAtomic> is the
 object returned by the access operator() of ScatterAccess.  This class
 inherits from the Prod<> reducer, and similar to that returned by an Atomic
 View, it wraps and atomic_prod with convenient operator*=, etc. atomic_prod
 uses the atomic_compare_exchange. This version also has the update(rhs)
 and reset() functions. */
template <typename ValueType, typename DeviceType>
struct ScatterValue<ValueType, flare::experimental::ScatterProd, DeviceType,
                    flare::experimental::ScatterAtomic> {
  ValueType& value;

 public:
  FLARE_FORCEINLINE_FUNCTION ScatterValue(ValueType& value_in)
      : value(value_in) {}
  FLARE_FORCEINLINE_FUNCTION ScatterValue(ScatterValue&& other)
      : value(other.value) {}

  FLARE_FORCEINLINE_FUNCTION void operator*=(ValueType const& rhs) {
    flare::atomic_mul(&value, rhs);
  }
  FLARE_FORCEINLINE_FUNCTION void operator/=(ValueType const& rhs) {
    flare::atomic_div(&value, rhs);
  }

  FLARE_INLINE_FUNCTION
  void join(ValueType& dest, const ValueType& src) const {
    atomic_prod(&dest, src);
  }

  FLARE_FORCEINLINE_FUNCTION void update(ValueType const& rhs) {
    atomic_prod(&value, rhs);
  }
  FLARE_FORCEINLINE_FUNCTION void reset() {
    value = reduction_identity<ValueType>::prod();
  }
};

/* ScatterValue <Op=ScatterMin, Contribution=ScatterNonAtomic> is
   the object returned by the access operator() of ScatterAccess. This class
   inherits from the Min<> reducer and it wraps join(dest, src) with convenient
   update(rhs). Note the addition of update(ValueType const& rhs) and reset()
   are so that all reducers can have a common update function See
   ReduceDuplicates and ResetDuplicates ) */
template <typename ValueType, typename DeviceType>
struct ScatterValue<ValueType, flare::experimental::ScatterMin, DeviceType,
                    flare::experimental::ScatterNonAtomic> {
  ValueType& value;
  FLARE_FORCEINLINE_FUNCTION ScatterValue(ValueType& value_in)
      : value(value_in) {}
  FLARE_FORCEINLINE_FUNCTION ScatterValue(ScatterValue&& other)
      : value(other.value) {}

 public:
  FLARE_FORCEINLINE_FUNCTION void update(ValueType const& rhs) {
    value = rhs < value ? rhs : value;
  }
  FLARE_FORCEINLINE_FUNCTION void reset() {
    value = reduction_identity<ValueType>::min();
  }
};

/* ScatterValue <Op=ScatterMin, Contribution=ScatterAtomic> is the
   object returned by the access operator() of ScatterAccess. This class
   inherits from the Min<> reducer, and similar to that returned by an Atomic
   View, it wraps atomic_min with join(), etc. atomic_min uses the
   atomic_compare_exchange. This version also has the update(rhs) and reset()
   functions. */
template <typename ValueType, typename DeviceType>
struct ScatterValue<ValueType, flare::experimental::ScatterMin, DeviceType,
                    flare::experimental::ScatterAtomic> {
  ValueType& value;

 public:
  FLARE_FORCEINLINE_FUNCTION ScatterValue(ValueType& value_in)
      : value(value_in) {}
  FLARE_FORCEINLINE_FUNCTION ScatterValue(ScatterValue&& other)
      : value(other.value) {}

  FLARE_INLINE_FUNCTION
  void join(ValueType& dest, const ValueType& src) const {
    atomic_min(&dest, src);
  }

  FLARE_FORCEINLINE_FUNCTION void update(ValueType const& rhs) {
    this->join(value, rhs);
  }
  FLARE_FORCEINLINE_FUNCTION void reset() {
    value = reduction_identity<ValueType>::min();
  }
};

/* ScatterValue <Op=ScatterMax, Contribution=ScatterNonAtomic> is
   the object returned by the access operator() of ScatterAccess. This class
   inherits from the Max<> reducer and it wraps join(dest, src) with convenient
   update(rhs). Note the addition of update(ValueType const& rhs) and reset()
   are so that all reducers can have a common update function See
   ReduceDuplicates and ResetDuplicates ) */
template <typename ValueType, typename DeviceType>
struct ScatterValue<ValueType, flare::experimental::ScatterMax, DeviceType,
                    flare::experimental::ScatterNonAtomic> {
  ValueType& value;

 public:
  FLARE_FORCEINLINE_FUNCTION ScatterValue(ValueType& value_in)
      : value(value_in) {}
  FLARE_FORCEINLINE_FUNCTION ScatterValue(ScatterValue&& other)
      : value(other.value) {}
  FLARE_FORCEINLINE_FUNCTION void update(ValueType const& rhs) {
    value = rhs > value ? rhs : value;
  }
  FLARE_FORCEINLINE_FUNCTION void reset() {
    value = reduction_identity<ValueType>::max();
  }
};

/* ScatterValue <Op=ScatterMax, Contribution=ScatterAtomic> is the
   object returned by the access operator() of ScatterAccess. This class
   inherits from the Max<> reducer, and similar to that returned by an Atomic
   View, it wraps atomic_max with join(), etc. atomic_max uses the
   atomic_compare_exchange. This version also has the update(rhs) and reset()
   functions. */
template <typename ValueType, typename DeviceType>
struct ScatterValue<ValueType, flare::experimental::ScatterMax, DeviceType,
                    flare::experimental::ScatterAtomic> {
  ValueType& value;

 public:
  FLARE_FORCEINLINE_FUNCTION ScatterValue(ValueType& value_in)
      : value(value_in) {}
  FLARE_FORCEINLINE_FUNCTION ScatterValue(ScatterValue&& other)
      : value(other.value) {}

  FLARE_INLINE_FUNCTION
  void join(ValueType& dest, const ValueType& src) const {
    atomic_max(&dest, src);
  }

  FLARE_FORCEINLINE_FUNCTION void update(ValueType const& rhs) {
    this->join(value, rhs);
  }
  FLARE_FORCEINLINE_FUNCTION void reset() {
    value = reduction_identity<ValueType>::max();
  }
};

/* DuplicatedDataType, given a View DataType, will create a new DataType
   that has a new runtime dimension which becomes the largest-stride dimension.
   In the case of LayoutLeft, due to the limitation induced by the design of
   DataType itself, it must convert any existing compile-time dimensions into
   runtime dimensions. */
template <typename T, typename Layout>
struct DuplicatedDataType;

template <typename T>
struct DuplicatedDataType<T, flare::LayoutRight> {
  using value_type = T*;  // For LayoutRight, add a star all the way on the left
};

template <typename T, size_t N>
struct DuplicatedDataType<T[N], flare::LayoutRight> {
  using value_type =
      typename DuplicatedDataType<T, flare::LayoutRight>::value_type[N];
};

template <typename T>
struct DuplicatedDataType<T[], flare::LayoutRight> {
  using value_type =
      typename DuplicatedDataType<T, flare::LayoutRight>::value_type[];
};

template <typename T>
struct DuplicatedDataType<T*, flare::LayoutRight> {
  using value_type =
      typename DuplicatedDataType<T, flare::LayoutRight>::value_type*;
};

template <typename T>
struct DuplicatedDataType<T, flare::LayoutLeft> {
  using value_type = T*;
};

template <typename T, size_t N>
struct DuplicatedDataType<T[N], flare::LayoutLeft> {
  using value_type =
      typename DuplicatedDataType<T, flare::LayoutLeft>::value_type*;
};

template <typename T>
struct DuplicatedDataType<T[], flare::LayoutLeft> {
  using value_type =
      typename DuplicatedDataType<T, flare::LayoutLeft>::value_type*;
};

template <typename T>
struct DuplicatedDataType<T*, flare::LayoutLeft> {
  using value_type =
      typename DuplicatedDataType<T, flare::LayoutLeft>::value_type*;
};

/* Insert integer argument pack into array */

template <class T>
void args_to_array(size_t* array, int pos, T dim0) {
  array[pos] = dim0;
}
template <class T, class... Dims>
void args_to_array(size_t* array, int pos, T dim0, Dims... dims) {
  array[pos] = dim0;
  args_to_array(array, pos + 1, dims...);
}

/* Slice is just responsible for stuffing the correct number of flare::ALL
   arguments on the correct side of the index in a call to subview() to get a
   subview where the index specified is the largest-stride one. */
template <typename Layout, int rank, typename V, typename... Args>
struct Slice {
  using next       = Slice<Layout, rank - 1, V, flare::ALL_t, Args...>;
  using value_type = typename next::value_type;

  static value_type get(V const& src, const size_t i, Args... args) {
    return next::get(src, i, flare::ALL, args...);
  }
};

template <typename V, typename... Args>
struct Slice<flare::LayoutRight, 1, V, Args...> {
  using value_type =
      typename flare::detail::ViewMapping<void, V, const size_t, Args...>::type;
  static value_type get(V const& src, const size_t i, Args... args) {
    return flare::subview(src, i, args...);
  }
};

template <typename V, typename... Args>
struct Slice<flare::LayoutLeft, 1, V, Args...> {
  using value_type =
      typename flare::detail::ViewMapping<void, V, Args..., const size_t>::type;
  static value_type get(V const& src, const size_t i, Args... args) {
    return flare::subview(src, args..., i);
  }
};

template <typename ExecSpace, typename ValueType, typename Op>
struct ReduceDuplicates;

template <typename ExecSpace, typename ValueType, typename Op>
struct ReduceDuplicatesBase {
  using Derived = ReduceDuplicates<ExecSpace, ValueType, Op>;
  ValueType const* src;
  ValueType* dst;
  size_t stride;
  size_t start;
  size_t n;
  ReduceDuplicatesBase(ExecSpace const& exec_space, ValueType const* src_in,
                       ValueType* dest_in, size_t stride_in, size_t start_in,
                       size_t n_in, std::string const& name)
      : src(src_in), dst(dest_in), stride(stride_in), start(start_in), n(n_in) {
    parallel_for(
        std::string("flare::ScatterView::ReduceDuplicates [") + name + "]",
        RangePolicy<ExecSpace, size_t>(exec_space, 0, stride),
        static_cast<Derived const&>(*this));
  }
};

/* ReduceDuplicates -- Perform reduction on destination array using strided
 * source Use ScatterValue<> specific to operation to wrap destination array so
 * that the reduction operation can be accessed via the update(rhs) function */
template <typename ExecSpace, typename ValueType, typename Op>
struct ReduceDuplicates
    : public ReduceDuplicatesBase<ExecSpace, ValueType, Op> {
  using Base = ReduceDuplicatesBase<ExecSpace, ValueType, Op>;
  ReduceDuplicates(ExecSpace const& exec_space, ValueType const* src_in,
                   ValueType* dst_in, size_t stride_in, size_t start_in,
                   size_t n_in, std::string const& name)
      : Base(exec_space, src_in, dst_in, stride_in, start_in, n_in, name) {}
  FLARE_FORCEINLINE_FUNCTION void operator()(size_t i) const {
    for (size_t j = Base::start; j < Base::n; ++j) {
      ScatterValue<ValueType, Op, ExecSpace,
                   flare::experimental::ScatterNonAtomic>
          sv(Base::dst[i]);
      sv.update(Base::src[i + Base::stride * j]);
    }
  }
};

template <typename ExecSpace, typename ValueType, typename Op>
struct ResetDuplicates;

template <typename ExecSpace, typename ValueType, typename Op>
struct ResetDuplicatesBase {
  using Derived = ResetDuplicates<ExecSpace, ValueType, Op>;
  ValueType* data;
  ResetDuplicatesBase(ExecSpace const& exec_space, ValueType* data_in,
                      size_t size_in, std::string const& name)
      : data(data_in) {
    parallel_for(
        std::string("flare::ScatterView::ResetDuplicates [") + name + "]",
        RangePolicy<ExecSpace, size_t>(exec_space, 0, size_in),
        static_cast<Derived const&>(*this));
  }
};

/* ResetDuplicates -- Perform reset on destination array
 *    Use ScatterValue<> specific to operation to wrap destination array so that
 *    the reset operation can be accessed via the reset() function */
template <typename ExecSpace, typename ValueType, typename Op>
struct ResetDuplicates : public ResetDuplicatesBase<ExecSpace, ValueType, Op> {
  using Base = ResetDuplicatesBase<ExecSpace, ValueType, Op>;
  ResetDuplicates(ExecSpace const& exec_space, ValueType* data_in,
                  size_t size_in, std::string const& name)
      : Base(exec_space, data_in, size_in, name) {}
  FLARE_FORCEINLINE_FUNCTION void operator()(size_t i) const {
    ScatterValue<ValueType, Op, ExecSpace,
                 flare::experimental::ScatterNonAtomic>
        sv(Base::data[i]);
    sv.reset();
  }
};

template <typename... P>
void check_scatter_view_allocation_properties_argument(
    ViewCtorProp<P...> const&) {
  static_assert(ViewCtorProp<P...>::has_execution_space &&
                    ViewCtorProp<P...>::has_label &&
                    ViewCtorProp<P...>::initialize,
                "Allocation property must have an execution name as well as a "
                "label, and must perform the view initialization");
}

}  // namespace experimental
}  // namespace detail
}  // namespace flare

namespace flare {
namespace experimental {

template <typename DataType,
          typename Layout      = flare::DefaultExecutionSpace::array_layout,
          typename DeviceType  = flare::DefaultExecutionSpace,
          typename Op          = flare::experimental::ScatterSum,
          typename Duplication = typename flare::detail::experimental::
              DefaultDuplication<typename DeviceType::execution_space>::type,
          typename Contribution =
              typename flare::detail::experimental::DefaultContribution<
                  typename DeviceType::execution_space, Duplication>::type>
class ScatterView;

template <class>
struct is_scatter_view : public std::false_type {};

template <class D, class... P>
struct is_scatter_view<ScatterView<D, P...>> : public std::true_type {};

template <class D, class... P>
struct is_scatter_view<const ScatterView<D, P...>> : public std::true_type {};

template <class T>
inline constexpr bool is_scatter_view_v = is_scatter_view<T>::value;

template <typename DataType, typename Op, typename DeviceType, typename Layout,
          typename Duplication, typename Contribution,
          typename OverrideContribution>
class ScatterAccess;

// non-duplicated implementation
template <typename DataType, typename Op, typename DeviceType, typename Layout,
          typename Contribution>
class ScatterView<DataType, Layout, DeviceType, Op, ScatterNonDuplicated,
                  Contribution> {
 public:
  using execution_space         = typename DeviceType::execution_space;
  using memory_space            = typename DeviceType::memory_space;
  using device_type             = flare::Device<execution_space, memory_space>;
  using original_view_type      = flare::View<DataType, Layout, device_type>;
  using original_value_type     = typename original_view_type::value_type;
  using original_reference_type = typename original_view_type::reference_type;
  friend class ScatterAccess<DataType, Op, DeviceType, Layout,
                             ScatterNonDuplicated, Contribution,
                             ScatterNonAtomic>;
  friend class ScatterAccess<DataType, Op, DeviceType, Layout,
                             ScatterNonDuplicated, Contribution, ScatterAtomic>;
  template <class, class, class, class, class, class>
  friend class ScatterView;

  ScatterView() = default;

  template <typename RT, typename... RP>
  ScatterView(View<RT, RP...> const& original_view)
      : internal_view(original_view) {}

  template <typename RT, typename... P, typename... RP>
  ScatterView(execution_space const& /* exec_space */,
              View<RT, RP...> const& original_view)
      : internal_view(original_view) {}

  template <typename... Dims>
  ScatterView(std::string const& name, Dims... dims)
      : internal_view(name, dims...) {}

  // This overload allows specifying an execution space instance to be
  // used by passing, e.g., flare::view_alloc(exec_space, "label") as
  // first argument.
  template <typename... P, typename... Dims>
  ScatterView(::flare::detail::ViewCtorProp<P...> const& arg_prop, Dims... dims)
      : internal_view(arg_prop, dims...) {
    using ::flare::detail::experimental::
        check_scatter_view_allocation_properties_argument;
    check_scatter_view_allocation_properties_argument(arg_prop);
  }

  template <typename OtherDataType, typename OtherDeviceType>
  FLARE_FUNCTION ScatterView(
      const ScatterView<OtherDataType, Layout, OtherDeviceType, Op,
                        ScatterNonDuplicated, Contribution>& other_view)
      : internal_view(other_view.internal_view) {}

  template <typename OtherDataType, typename OtherDeviceType>
  FLARE_FUNCTION ScatterView& operator=(
      const ScatterView<OtherDataType, Layout, OtherDeviceType, Op,
                        ScatterNonDuplicated, Contribution>& other_view) {
    internal_view = other_view.internal_view;
    return *this;
  }

  template <typename OverrideContribution = Contribution>
  FLARE_FORCEINLINE_FUNCTION
      ScatterAccess<DataType, Op, DeviceType, Layout, ScatterNonDuplicated,
                    Contribution, OverrideContribution>
      access() const {
    return ScatterAccess<DataType, Op, DeviceType, Layout, ScatterNonDuplicated,
                         Contribution, OverrideContribution>(*this);
  }

  original_view_type subview() const { return internal_view; }

  FLARE_INLINE_FUNCTION constexpr bool is_allocated() const {
    return internal_view.is_allocated();
  }

  template <typename DT, typename... RP>
  void contribute_into(View<DT, RP...> const& dest) const {
    contribute_into(execution_space(), dest);
  }

  template <typename DT, typename... RP>
  void contribute_into(execution_space const& exec_space,
                       View<DT, RP...> const& dest) const {
    using dest_type = View<DT, RP...>;
    static_assert(std::is_same<typename dest_type::array_layout, Layout>::value,
                  "ScatterView contribute destination has different layout");
    static_assert(
        flare::SpaceAccessibility<
            execution_space, typename dest_type::memory_space>::accessible,
        "ScatterView contribute destination memory space not accessible");
    if (dest.data() == internal_view.data()) return;
    flare::detail::experimental::ReduceDuplicates<execution_space,
                                                 original_value_type, Op>(
        exec_space, internal_view.data(), dest.data(), 0, 0, 1,
        internal_view.label());
  }

  void reset(execution_space const& exec_space = execution_space()) {
    flare::detail::experimental::ResetDuplicates<execution_space,
                                                original_value_type, Op>(
        exec_space, internal_view.data(), internal_view.size(),
        internal_view.label());
  }
  template <typename DT, typename... RP>
  void reset_except(View<DT, RP...> const& view) {
    reset_except(execution_space(), view);
  }

  template <typename DT, typename... RP>
  void reset_except(const execution_space& exec_space,
                    View<DT, RP...> const& view) {
    if (view.data() != internal_view.data()) reset(exec_space);
  }

  void resize(const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    ::flare::resize(internal_view, n0, n1, n2, n3, n4, n5, n6, n7);
  }

  template <class... ViewCtorArgs>
  void resize(const ::flare::detail::ViewCtorProp<ViewCtorArgs...>& arg_prop,
              const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    ::flare::resize(arg_prop, internal_view, n0, n1, n2, n3, n4, n5, n6, n7);
  }

  template <class I>
  std::enable_if_t<flare::detail::is_view_ctor_property<I>::value> resize(
      const I& arg_prop, const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    ::flare::resize(arg_prop, internal_view, n0, n1, n2, n3, n4, n5, n6, n7);
  }

  template <class... ViewCtorArgs>
  void realloc(const flare::detail::ViewCtorProp<ViewCtorArgs...>& arg_prop,
               const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    ::flare::realloc(arg_prop, internal_view, n0, n1, n2, n3, n4, n5, n6, n7);
  }

  void realloc(const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    ::flare::realloc(internal_view, n0, n1, n2, n3, n4, n5, n6, n7);
  }

  template <class I>
  std::enable_if_t<flare::detail::is_view_ctor_property<I>::value> realloc(
      const I& arg_prop, const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n7 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    ::flare::realloc(arg_prop, internal_view, n0, n1, n2, n3, n4, n5, n6, n7);
  }

 protected:
  template <typename... Args>
  FLARE_FORCEINLINE_FUNCTION original_reference_type at(Args... args) const {
    return internal_view(args...);
  }

 private:
  using internal_view_type = original_view_type;
  internal_view_type internal_view;
};

template <typename DataType, typename Op, typename DeviceType, typename Layout,
          typename Contribution, typename OverrideContribution>
class ScatterAccess<DataType, Op, DeviceType, Layout, ScatterNonDuplicated,
                    Contribution, OverrideContribution> {
 public:
  using view_type           = ScatterView<DataType, Layout, DeviceType, Op,
                                ScatterNonDuplicated, Contribution>;
  using original_value_type = typename view_type::original_value_type;
  using value_type          = flare::detail::experimental::ScatterValue<
      original_value_type, Op, DeviceType, OverrideContribution>;

  FLARE_INLINE_FUNCTION
  ScatterAccess() : view(view_type()) {}

  FLARE_INLINE_FUNCTION
  ScatterAccess(view_type const& view_in) : view(view_in) {}
  FLARE_DEFAULTED_FUNCTION
  ~ScatterAccess() = default;

  template <typename... Args>
  FLARE_FORCEINLINE_FUNCTION value_type operator()(Args... args) const {
    return view.at(args...);
  }

  template <typename Arg>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      view_type::original_view_type::rank == 1 && std::is_integral<Arg>::value,
      value_type>
  operator[](Arg arg) const {
    return view.at(arg);
  }

 private:
  view_type const& view;
};

// duplicated implementation
// LayoutLeft and LayoutRight are different enough that we'll just specialize
// each

template <typename DataType, typename Op, typename DeviceType,
          typename Contribution>
class ScatterView<DataType, flare::LayoutRight, DeviceType, Op,
                  ScatterDuplicated, Contribution> {
 public:
  using execution_space = typename DeviceType::execution_space;
  using memory_space    = typename DeviceType::memory_space;
  using device_type     = flare::Device<execution_space, memory_space>;
  using original_view_type =
      flare::View<DataType, flare::LayoutRight, device_type>;
  using original_value_type     = typename original_view_type::value_type;
  using original_reference_type = typename original_view_type::reference_type;
  friend class ScatterAccess<DataType, Op, DeviceType, flare::LayoutRight,
                             ScatterDuplicated, Contribution, ScatterNonAtomic>;
  friend class ScatterAccess<DataType, Op, DeviceType, flare::LayoutRight,
                             ScatterDuplicated, Contribution, ScatterAtomic>;
  template <class, class, class, class, class, class>
  friend class ScatterView;

  using data_type_info =
      typename flare::detail::experimental::DuplicatedDataType<
          DataType, flare::LayoutRight>;
  using internal_data_type = typename data_type_info::value_type;
  using internal_view_type =
      flare::View<internal_data_type, flare::LayoutRight, device_type>;

  ScatterView() = default;

  template <typename OtherDataType, typename OtherDeviceType>
  FLARE_FUNCTION ScatterView(
      const ScatterView<OtherDataType, flare::LayoutRight, OtherDeviceType, Op,
                        ScatterDuplicated, Contribution>& other_view)
      : unique_token(other_view.unique_token),
        internal_view(other_view.internal_view) {}

  template <typename OtherDataType, typename OtherDeviceType>
  FLARE_FUNCTION ScatterView& operator=(
      const ScatterView<OtherDataType, flare::LayoutRight, OtherDeviceType, Op,
                        ScatterDuplicated, Contribution>& other_view) {
    unique_token  = other_view.unique_token;
    internal_view = other_view.internal_view;
    return *this;
  }

  template <typename RT, typename... RP>
  ScatterView(View<RT, RP...> const& original_view)
      : ScatterView(execution_space(), original_view) {}

  template <typename RT, typename... P, typename... RP>
  ScatterView(execution_space const& exec_space,
              View<RT, RP...> const& original_view)
      : unique_token(),
        internal_view(
            view_alloc(WithoutInitializing,
                       std::string("duplicated_") + original_view.label(),
                       exec_space),
            unique_token.size(),
            original_view.rank_dynamic > 0 ? original_view.extent(0)
                                           : FLARE_IMPL_CTOR_DEFAULT_ARG,
            original_view.rank_dynamic > 1 ? original_view.extent(1)
                                           : FLARE_IMPL_CTOR_DEFAULT_ARG,
            original_view.rank_dynamic > 2 ? original_view.extent(2)
                                           : FLARE_IMPL_CTOR_DEFAULT_ARG,
            original_view.rank_dynamic > 3 ? original_view.extent(3)
                                           : FLARE_IMPL_CTOR_DEFAULT_ARG,
            original_view.rank_dynamic > 4 ? original_view.extent(4)
                                           : FLARE_IMPL_CTOR_DEFAULT_ARG,
            original_view.rank_dynamic > 5 ? original_view.extent(5)
                                           : FLARE_IMPL_CTOR_DEFAULT_ARG,
            original_view.rank_dynamic > 6 ? original_view.extent(6)
                                           : FLARE_IMPL_CTOR_DEFAULT_ARG)

  {
    reset(exec_space);
  }

  template <typename... Dims>
  ScatterView(std::string const& name, Dims... dims)
      : ScatterView(view_alloc(execution_space(), name), dims...) {}

  // This overload allows specifying an execution space instance to be
  // used by passing, e.g., flare::view_alloc(exec_space, "label") as
  // first argument.
  template <typename... P, typename... Dims>
  ScatterView(::flare::detail::ViewCtorProp<P...> const& arg_prop, Dims... dims)
      : internal_view(view_alloc(WithoutInitializing,
                                 static_cast<::flare::detail::ViewCtorProp<
                                     void, std::string> const&>(arg_prop)
                                     .value),
                      unique_token.size(), dims...) {
    using ::flare::detail::experimental::
        check_scatter_view_allocation_properties_argument;
    check_scatter_view_allocation_properties_argument(arg_prop);

    auto const& exec_space =
        flare::detail::get_property<flare::detail::ExecutionSpaceTag>(arg_prop);
    reset(exec_space);
  }

  template <typename OverrideContribution = Contribution>
  FLARE_FORCEINLINE_FUNCTION
      ScatterAccess<DataType, Op, DeviceType, flare::LayoutRight,
                    ScatterDuplicated, Contribution, OverrideContribution>
      access() const {
    return ScatterAccess<DataType, Op, DeviceType, flare::LayoutRight,
                         ScatterDuplicated, Contribution, OverrideContribution>(
        *this);
  }

  typename flare::detail::experimental::Slice<flare::LayoutRight,
                                             internal_view_type::rank,
                                             internal_view_type>::value_type
  subview() const {
    return flare::detail::experimental::Slice<
        flare::LayoutRight, internal_view_type::rank,
        internal_view_type>::get(internal_view, 0);
  }

  FLARE_INLINE_FUNCTION constexpr bool is_allocated() const {
    return internal_view.is_allocated();
  }

  template <typename DT, typename... RP>
  void contribute_into(View<DT, RP...> const& dest) const {
    contribute_into(execution_space(), dest);
  }

  template <typename DT, typename... RP>
  void contribute_into(execution_space const& exec_space,
                       View<DT, RP...> const& dest) const {
    using dest_type = View<DT, RP...>;
    static_assert(std::is_same<typename dest_type::array_layout,
                               flare::LayoutRight>::value,
                  "ScatterView deep_copy destination has different layout");
    static_assert(
        flare::SpaceAccessibility<
            execution_space, typename dest_type::memory_space>::accessible,
        "ScatterView deep_copy destination memory space not accessible");
    bool is_equal = (dest.data() == internal_view.data());
    size_t start  = is_equal ? 1 : 0;
    flare::detail::experimental::ReduceDuplicates<execution_space,
                                                 original_value_type, Op>(
        exec_space, internal_view.data(), dest.data(), internal_view.stride(0),
        start, internal_view.extent(0), internal_view.label());
  }

  void reset(execution_space const& exec_space = execution_space()) {
    flare::detail::experimental::ResetDuplicates<execution_space,
                                                original_value_type, Op>(
        exec_space, internal_view.data(), internal_view.size(),
        internal_view.label());
  }

  template <typename DT, typename... RP>
  void reset_except(View<DT, RP...> const& view) {
    reset_except(execution_space(), view);
  }

  template <typename DT, typename... RP>
  void reset_except(execution_space const& exec_space,
                    View<DT, RP...> const& view) {
    if (view.data() != internal_view.data()) {
      reset(exec_space);
      return;
    }
    flare::detail::experimental::ResetDuplicates<execution_space,
                                                original_value_type, Op>(
        exec_space, internal_view.data() + view.size(),
        internal_view.size() - view.size(), internal_view.label());
  }

  void resize(const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    ::flare::resize(internal_view, unique_token.size(), n0, n1, n2, n3, n4, n5,
                     n6);
  }

  template <class... ViewCtorArgs>
  void resize(const ::flare::detail::ViewCtorProp<ViewCtorArgs...>& arg_prop,
              const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    ::flare::resize(arg_prop, internal_view, unique_token.size(), n0, n1, n2,
                     n3, n4, n5, n6);
  }

  template <class I>
  std::enable_if_t<flare::detail::is_view_ctor_property<I>::value> resize(
      const I& arg_prop, const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    ::flare::resize(arg_prop, internal_view, unique_token.size(), n0, n1, n2,
                     n3, n4, n5, n6);
  }

  template <class... ViewCtorArgs>
  void realloc(const ::flare::detail::ViewCtorProp<ViewCtorArgs...>& arg_prop,
               const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    ::flare::realloc(arg_prop, internal_view, unique_token.size(), n0, n1, n2,
                      n3, n4, n5, n6);
  }

  void realloc(const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    ::flare::realloc(internal_view, unique_token.size(), n0, n1, n2, n3, n4,
                      n5, n6);
  }

  template <class I>
  std::enable_if_t<flare::detail::is_view_ctor_property<I>::value> realloc(
      const I& arg_prop, const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    ::flare::realloc(arg_prop, internal_view, unique_token.size(), n0, n1, n2,
                      n3, n4, n5, n6);
  }

 protected:
  template <typename... Args>
  FLARE_FORCEINLINE_FUNCTION original_reference_type at(int rank,
                                                         Args... args) const {
    return internal_view(rank, args...);
  }

 protected:
  using unique_token_type = flare::experimental::UniqueToken<
      execution_space, flare::experimental::UniqueTokenScope::Global>;

  unique_token_type unique_token;
  internal_view_type internal_view;
};

template <typename DataType, typename Op, typename DeviceType,
          typename Contribution>
class ScatterView<DataType, flare::LayoutLeft, DeviceType, Op,
                  ScatterDuplicated, Contribution> {
 public:
  using execution_space = typename DeviceType::execution_space;
  using memory_space    = typename DeviceType::memory_space;
  using device_type     = flare::Device<execution_space, memory_space>;
  using original_view_type =
      flare::View<DataType, flare::LayoutLeft, device_type>;
  using original_value_type     = typename original_view_type::value_type;
  using original_reference_type = typename original_view_type::reference_type;
  friend class ScatterAccess<DataType, Op, DeviceType, flare::LayoutLeft,
                             ScatterDuplicated, Contribution, ScatterNonAtomic>;
  friend class ScatterAccess<DataType, Op, DeviceType, flare::LayoutLeft,
                             ScatterDuplicated, Contribution, ScatterAtomic>;
  template <class, class, class, class, class, class>
  friend class ScatterView;

  using data_type_info =
      typename flare::detail::experimental::DuplicatedDataType<
          DataType, flare::LayoutLeft>;
  using internal_data_type = typename data_type_info::value_type;
  using internal_view_type =
      flare::View<internal_data_type, flare::LayoutLeft, device_type>;

  ScatterView() = default;

  template <typename RT, typename... RP>
  ScatterView(View<RT, RP...> const& original_view)
      : ScatterView(execution_space(), original_view) {}

  template <typename RT, typename... P, typename... RP>
  ScatterView(execution_space const& exec_space,
              View<RT, RP...> const& original_view)
      : unique_token() {
    size_t arg_N[8] = {original_view.rank > 0 ? original_view.extent(0)
                                              : FLARE_IMPL_CTOR_DEFAULT_ARG,
                       original_view.rank > 1 ? original_view.extent(1)
                                              : FLARE_IMPL_CTOR_DEFAULT_ARG,
                       original_view.rank > 2 ? original_view.extent(2)
                                              : FLARE_IMPL_CTOR_DEFAULT_ARG,
                       original_view.rank > 3 ? original_view.extent(3)
                                              : FLARE_IMPL_CTOR_DEFAULT_ARG,
                       original_view.rank > 4 ? original_view.extent(4)
                                              : FLARE_IMPL_CTOR_DEFAULT_ARG,
                       original_view.rank > 5 ? original_view.extent(5)
                                              : FLARE_IMPL_CTOR_DEFAULT_ARG,
                       original_view.rank > 6 ? original_view.extent(6)
                                              : FLARE_IMPL_CTOR_DEFAULT_ARG,
                       FLARE_IMPL_CTOR_DEFAULT_ARG};
    arg_N[internal_view_type::rank - 1] = unique_token.size();
    internal_view                       = internal_view_type(
        view_alloc(WithoutInitializing,
                   std::string("duplicated_") + original_view.label(),
                   exec_space),
        arg_N[0], arg_N[1], arg_N[2], arg_N[3], arg_N[4], arg_N[5], arg_N[6],
        arg_N[7]);
    reset(exec_space);
  }

  template <typename... Dims>
  ScatterView(std::string const& name, Dims... dims)
      : ScatterView(view_alloc(execution_space(), name), dims...) {}

  // This overload allows specifying an execution space instance to be
  // used by passing, e.g., flare::view_alloc(exec_space, "label") as
  // first argument.
  template <typename... P, typename... Dims>
  ScatterView(::flare::detail::ViewCtorProp<P...> const& arg_prop,
              Dims... dims) {
    using ::flare::detail::experimental::
        check_scatter_view_allocation_properties_argument;
    check_scatter_view_allocation_properties_argument(arg_prop);

    original_view_type original_view;
    size_t arg_N[8] = {original_view.rank > 0 ? original_view.static_extent(0)
                                              : FLARE_IMPL_CTOR_DEFAULT_ARG,
                       original_view.rank > 1 ? original_view.static_extent(1)
                                              : FLARE_IMPL_CTOR_DEFAULT_ARG,
                       original_view.rank > 2 ? original_view.static_extent(2)
                                              : FLARE_IMPL_CTOR_DEFAULT_ARG,
                       original_view.rank > 3 ? original_view.static_extent(3)
                                              : FLARE_IMPL_CTOR_DEFAULT_ARG,
                       original_view.rank > 4 ? original_view.static_extent(4)
                                              : FLARE_IMPL_CTOR_DEFAULT_ARG,
                       original_view.rank > 5 ? original_view.static_extent(5)
                                              : FLARE_IMPL_CTOR_DEFAULT_ARG,
                       original_view.rank > 6 ? original_view.static_extent(6)
                                              : FLARE_IMPL_CTOR_DEFAULT_ARG,
                       FLARE_IMPL_CTOR_DEFAULT_ARG};
    flare::detail::experimental::args_to_array(arg_N, 0, dims...);
    arg_N[internal_view_type::rank - 1] = unique_token.size();

    auto const& name =
        flare::detail::get_property<flare::detail::LabelTag>(arg_prop);
    internal_view = internal_view_type(view_alloc(WithoutInitializing, name),
                                       arg_N[0], arg_N[1], arg_N[2], arg_N[3],
                                       arg_N[4], arg_N[5], arg_N[6], arg_N[7]);

    auto const& exec_space =
        flare::detail::get_property<flare::detail::ExecutionSpaceTag>(arg_prop);
    reset(exec_space);
  }

  template <typename OtherDataType, typename OtherDeviceType>
  FLARE_FUNCTION ScatterView(
      const ScatterView<OtherDataType, flare::LayoutLeft, OtherDeviceType, Op,
                        ScatterDuplicated, Contribution>& other_view)
      : unique_token(other_view.unique_token),
        internal_view(other_view.internal_view) {}

  template <typename OtherDataType, typename OtherDeviceType>
  FLARE_FUNCTION ScatterView& operator=(
      const ScatterView<OtherDataType, flare::LayoutLeft, OtherDeviceType, Op,
                        ScatterDuplicated, Contribution>& other_view) {
    unique_token  = other_view.unique_token;
    internal_view = other_view.internal_view;
    return *this;
  }

  template <typename OverrideContribution = Contribution>
  FLARE_FORCEINLINE_FUNCTION
      ScatterAccess<DataType, Op, DeviceType, flare::LayoutLeft,
                    ScatterDuplicated, Contribution, OverrideContribution>
      access() const {
    return ScatterAccess<DataType, Op, DeviceType, flare::LayoutLeft,
                         ScatterDuplicated, Contribution, OverrideContribution>(
        *this);
  }

  typename flare::detail::experimental::Slice<flare::LayoutLeft,
                                             internal_view_type::rank,
                                             internal_view_type>::value_type
  subview() const {
    return flare::detail::experimental::Slice<
        flare::LayoutLeft, internal_view_type::rank,
        internal_view_type>::get(internal_view, 0);
  }

  FLARE_INLINE_FUNCTION constexpr bool is_allocated() const {
    return internal_view.is_allocated();
  }

  template <typename... RP>
  void contribute_into(View<RP...> const& dest) const {
    contribute_into(execution_space(), dest);
  }

  template <typename... RP>
  void contribute_into(execution_space const& exec_space,
                       View<RP...> const& dest) const {
    using dest_type = View<RP...>;
    static_assert(
        std::is_same<typename dest_type::value_type,
                     typename original_view_type::non_const_value_type>::value,
        "ScatterView deep_copy destination has wrong value_type");
    static_assert(std::is_same<typename dest_type::array_layout,
                               flare::LayoutLeft>::value,
                  "ScatterView deep_copy destination has different layout");
    static_assert(
        flare::SpaceAccessibility<
            execution_space, typename dest_type::memory_space>::accessible,
        "ScatterView deep_copy destination memory space not accessible");
    auto extent   = internal_view.extent(internal_view_type::rank - 1);
    bool is_equal = (dest.data() == internal_view.data());
    size_t start  = is_equal ? 1 : 0;
    flare::detail::experimental::ReduceDuplicates<execution_space,
                                                 original_value_type, Op>(
        exec_space, internal_view.data(), dest.data(),
        internal_view.stride(internal_view_type::rank - 1), start, extent,
        internal_view.label());
  }

  void reset(execution_space const& exec_space = execution_space()) {
    flare::detail::experimental::ResetDuplicates<execution_space,
                                                original_value_type, Op>(
        exec_space, internal_view.data(), internal_view.size(),
        internal_view.label());
  }

  template <typename DT, typename... RP>
  void reset_except(View<DT, RP...> const& view) {
    reset_except(execution_space(), view);
  }

  template <typename DT, typename... RP>
  void reset_except(execution_space const& exec_space,
                    View<DT, RP...> const& view) {
    if (view.data() != internal_view.data()) {
      reset(exec_space);
      return;
    }
    flare::detail::experimental::ResetDuplicates<execution_space,
                                                original_value_type, Op>(
        exec_space, internal_view.data() + view.size(),
        internal_view.size() - view.size(), internal_view.label());
  }

  void resize(const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
              const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    size_t arg_N[8] = {n0, n1, n2, n3, n4, n5, n6, 0};
    const int i     = internal_view.rank - 1;
    arg_N[i]        = unique_token.size();

    ::flare::resize(internal_view, arg_N[0], arg_N[1], arg_N[2], arg_N[3],
                     arg_N[4], arg_N[5], arg_N[6], arg_N[7]);
  }

  void realloc(const size_t n0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
               const size_t n6 = FLARE_IMPL_CTOR_DEFAULT_ARG) {
    size_t arg_N[8] = {n0, n1, n2, n3, n4, n5, n6, 0};
    const int i     = internal_view.rank - 1;
    arg_N[i]        = unique_token.size();

    ::flare::realloc(internal_view, arg_N[0], arg_N[1], arg_N[2], arg_N[3],
                      arg_N[4], arg_N[5], arg_N[6], arg_N[7]);
  }

 protected:
  template <typename... Args>
  FLARE_FORCEINLINE_FUNCTION original_reference_type at(int thread_id,
                                                         Args... args) const {
    return internal_view(args..., thread_id);
  }

 protected:
  using unique_token_type = flare::experimental::UniqueToken<
      execution_space, flare::experimental::UniqueTokenScope::Global>;

  unique_token_type unique_token;
  internal_view_type internal_view;
};

/* This object has to be separate in order to store the thread ID, which cannot
   be obtained until one is inside a parallel construct, and may be relatively
   expensive to obtain at every contribution
   (calls a non-inlined function, looks up a thread-local variable).
   Due to the expense, it is sensible to query it at most once per parallel
   iterate (ideally once per thread, but parallel_for doesn't expose that) and
   then store it in a stack variable.
   ScatterAccess serves as a non-const object on the stack which can store the
   thread ID */

template <typename DataType, typename Op, typename DeviceType, typename Layout,
          typename Contribution, typename OverrideContribution>
class ScatterAccess<DataType, Op, DeviceType, Layout, ScatterDuplicated,
                    Contribution, OverrideContribution> {
 public:
  using view_type           = ScatterView<DataType, Layout, DeviceType, Op,
                                ScatterDuplicated, Contribution>;
  using original_value_type = typename view_type::original_value_type;
  using value_type          = flare::detail::experimental::ScatterValue<
      original_value_type, Op, DeviceType, OverrideContribution>;

  FLARE_FORCEINLINE_FUNCTION
  ScatterAccess(view_type const& view_in)
      : view(view_in), thread_id(view_in.unique_token.acquire()) {}

  FLARE_FORCEINLINE_FUNCTION
  ~ScatterAccess() {
    if (thread_id != ~thread_id_type(0)) view.unique_token.release(thread_id);
  }

  template <typename... Args>
  FLARE_FORCEINLINE_FUNCTION value_type operator()(Args... args) const {
    return view.at(thread_id, args...);
  }

  template <typename Arg>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      view_type::original_view_type::rank == 1 && std::is_integral<Arg>::value,
      value_type>
  operator[](Arg arg) const {
    return view.at(thread_id, arg);
  }

 private:
  view_type const& view;

  // simplify RAII by disallowing copies
  ScatterAccess(ScatterAccess const& other) = delete;
  ScatterAccess& operator=(ScatterAccess const& other) = delete;
  ScatterAccess& operator=(ScatterAccess&& other) = delete;

 public:
  // do need to allow moves though, for the common
  // auto b = a.access();
  // that assignments turns into a move constructor call
  FLARE_FORCEINLINE_FUNCTION
  ScatterAccess(ScatterAccess&& other)
      : view(other.view), thread_id(other.thread_id) {
    other.thread_id = ~thread_id_type(0);
  }

 private:
  using unique_token_type = typename view_type::unique_token_type;
  using thread_id_type    = typename unique_token_type::size_type;
  thread_id_type thread_id;
};

template <typename Op          = flare::experimental::ScatterSum,
          typename Duplication = void, typename Contribution = void,
          typename RT, typename... RP>
ScatterView<
    RT, typename ViewTraits<RT, RP...>::array_layout,
    typename ViewTraits<RT, RP...>::device_type, Op,
    std::conditional_t<
        std::is_void<Duplication>::value,
        typename flare::detail::experimental::DefaultDuplication<
            typename ViewTraits<RT, RP...>::execution_space>::type,
        Duplication>,
    std::conditional_t<
        std::is_void<Contribution>::value,
        typename flare::detail::experimental::DefaultContribution<
            typename ViewTraits<RT, RP...>::execution_space,
            typename std::conditional_t<
                std::is_void<Duplication>::value,
                typename flare::detail::experimental::DefaultDuplication<
                    typename ViewTraits<RT, RP...>::execution_space>::type,
                Duplication>>::type,
        Contribution>>
create_scatter_view(View<RT, RP...> const& original_view) {
  return original_view;  // implicit ScatterView constructor call
}

template <typename Op, typename RT, typename... RP>
ScatterView<
    RT, typename ViewTraits<RT, RP...>::array_layout,
    typename ViewTraits<RT, RP...>::device_type, Op,
    typename flare::detail::experimental::DefaultDuplication<
        typename ViewTraits<RT, RP...>::execution_space>::type,
    typename flare::detail::experimental::DefaultContribution<
        typename ViewTraits<RT, RP...>::execution_space,
        typename flare::detail::experimental::DefaultDuplication<
            typename ViewTraits<RT, RP...>::execution_space>::type>::type>
create_scatter_view(Op, View<RT, RP...> const& original_view) {
  return original_view;  // implicit ScatterView constructor call
}

template <typename Op, typename Duplication, typename Contribution, typename RT,
          typename... RP>
ScatterView<RT, typename ViewTraits<RT, RP...>::array_layout,
            typename ViewTraits<RT, RP...>::device_type, Op, Duplication,
            Contribution>
create_scatter_view(Op, Duplication, Contribution,
                    View<RT, RP...> const& original_view) {
  return original_view;  // implicit ScatterView constructor call
}

}  // namespace experimental
}  // namespace flare

namespace flare {
namespace experimental {

template <typename DT1, typename DT2, typename LY, typename ES, typename OP,
          typename CT, typename DP, typename... VP>
void contribute(
    typename ES::execution_space const& exec_space, View<DT1, VP...>& dest,
    flare::experimental::ScatterView<DT2, LY, ES, OP, CT, DP> const& src) {
  src.contribute_into(exec_space, dest);
}

template <typename DT1, typename DT2, typename LY, typename ES, typename OP,
          typename CT, typename DP, typename... VP>
void contribute(
    View<DT1, VP...>& dest,
    flare::experimental::ScatterView<DT2, LY, ES, OP, CT, DP> const& src) {
  using execution_space = typename ES::execution_space;
  contribute(execution_space{}, dest, src);
}

}  // namespace experimental
}  // namespace flare

namespace flare {

template <typename DT, typename LY, typename ES, typename OP, typename CT,
          typename DP, typename... IS, class... ViewCtorArgs>
void realloc(
    const detail::ViewCtorProp<ViewCtorArgs...>& arg_prop,
    flare::experimental::ScatterView<DT, LY, ES, OP, CT, DP>& scatter_view,
    IS... is) {
  scatter_view.realloc(arg_prop, is...);
}

template <typename DT, typename LY, typename ES, typename OP, typename CT,
          typename DP, typename... IS>
void realloc(
    flare::experimental::ScatterView<DT, LY, ES, OP, CT, DP>& scatter_view,
    IS... is) {
  scatter_view.realloc(is...);
}

template <typename I, typename DT, typename LY, typename ES, typename OP,
          typename CT, typename DP, typename... IS>
std::enable_if_t<flare::detail::is_view_ctor_property<I>::value> realloc(
    const I& arg_prop,
    flare::experimental::ScatterView<DT, LY, ES, OP, CT, DP>& scatter_view,
    IS... is) {
  scatter_view.realloc(arg_prop, is...);
}

template <typename DT, typename LY, typename ES, typename OP, typename CT,
          typename DP, typename... IS>
void resize(
    flare::experimental::ScatterView<DT, LY, ES, OP, CT, DP>& scatter_view,
    IS... is) {
  scatter_view.resize(is...);
}

template <class... ViewCtorArgs, typename DT, typename LY, typename ES,
          typename OP, typename CT, typename DP, typename... IS>
void resize(
    const detail::ViewCtorProp<ViewCtorArgs...>& arg_prop,
    flare::experimental::ScatterView<DT, LY, ES, OP, CT, DP>& scatter_view,
    IS... is) {
  scatter_view.resize(arg_prop, is...);
}

template <typename I, typename DT, typename LY, typename ES, typename OP,
          typename CT, typename DP, typename... IS>
std::enable_if_t<flare::detail::is_view_ctor_property<I>::value> resize(
    const I& arg_prop,
    flare::experimental::ScatterView<DT, LY, ES, OP, CT, DP>& scatter_view,
    IS... is) {
  scatter_view.resize(arg_prop, is...);
}

}  // namespace flare

#endif  // FLARE_SCATTER_VIEW_H_
