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

#ifndef FLARE_CORE_TENSOR_VIEW_H_
#define FLARE_CORE_TENSOR_VIEW_H_

#include <type_traits>
#include <string>
#include <algorithm>
#include <initializer_list>

#include <flare/core_fwd.h>
#include <flare/core/memory/host_space.h>
#include <flare/core/memory/memory_traits.h>
#include <flare/core/policy/exec_policy.h>
#include <flare/core/tensor/view_hooks.h>

#include <flare/core/profile/tools.h>
#include <flare/core/common/utilities.h>
#include <flare/core/tensor/mdspan_extents.h>
#include <flare/core/common/min_max_clamp.h>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {
namespace detail {

template <class DataType>
struct ViewArrayAnalysis;

template <class DataType, class ArrayLayout,
          typename ValueType =
              typename ViewArrayAnalysis<DataType>::non_const_value_type>
struct ViewDataAnalysis;

template <class, class...>
class ViewMapping {
 public:
  enum : bool { is_assignable_data_type = false };
  enum : bool { is_assignable = false };
};

template <typename IntType>
constexpr FLARE_INLINE_FUNCTION std::size_t count_valid_integers(
    const IntType i0, const IntType i1, const IntType i2, const IntType i3,
    const IntType i4, const IntType i5, const IntType i6, const IntType i7) {
  static_assert(std::is_integral<IntType>::value,
                "count_valid_integers() must have integer arguments.");

  return (i0 != FLARE_INVALID_INDEX) + (i1 != FLARE_INVALID_INDEX) +
         (i2 != FLARE_INVALID_INDEX) + (i3 != FLARE_INVALID_INDEX) +
         (i4 != FLARE_INVALID_INDEX) + (i5 != FLARE_INVALID_INDEX) +
         (i6 != FLARE_INVALID_INDEX) + (i7 != FLARE_INVALID_INDEX);
}

FLARE_INLINE_FUNCTION
void runtime_check_rank(const size_t rank, const size_t dyn_rank,
                        const bool is_void_spec, const size_t i0,
                        const size_t i1, const size_t i2, const size_t i3,
                        const size_t i4, const size_t i5, const size_t i6,
                        const size_t i7, const std::string& label) {
  (void)(label);

  if (is_void_spec) {
    const size_t num_passed_args =
        count_valid_integers(i0, i1, i2, i3, i4, i5, i6, i7);

    if (num_passed_args != dyn_rank && num_passed_args != rank) {
      FLARE_IF_ON_HOST(
          const std::string message =
              "Constructor for flare View '" + label +
              "' has mismatched number of arguments. Number of arguments = " +
              std::to_string(num_passed_args) +
              " but dynamic rank = " + std::to_string(dyn_rank) + " \n";
          flare::abort(message.c_str());)
      FLARE_IF_ON_DEVICE(flare::abort("Constructor for flare View has "
                                        "mismatched number of arguments.");)
    }
  }
}

} /* namespace detail */
} /* namespace flare */

// Class to provide a uniform type
namespace flare {
namespace detail {
template <class ViewType, int Traits = 0>
struct ViewUniformType;
}
}  // namespace flare

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {

/** \class ViewTraits
 *  \brief Traits class for accessing attributes of a View.
 *
 * This is an implementation detail of View.  It is only of interest
 * to developers implementing a new specialization of View.
 *
 * Template argument options:
 *   - View< DataType >
 *   - View< DataType , Space >
 *   - View< DataType , Space , MemoryTraits >
 *   - View< DataType , ArrayLayout >
 *   - View< DataType , ArrayLayout , Space >
 *   - View< DataType , ArrayLayout , MemoryTraits >
 *   - View< DataType , ArrayLayout , Space , MemoryTraits >
 *   - View< DataType , MemoryTraits >
 */

template <class DataType, class... Properties>
struct ViewTraits;

template <>
struct ViewTraits<void> {
  using execution_space = void;
  using memory_space    = void;
  using HostMirrorSpace = void;
  using array_layout    = void;
  using memory_traits   = void;
  using specialize      = void;
  using hooks_policy    = void;
};

template <class... Prop>
struct ViewTraits<void, void, Prop...> {
  // Ignore an extraneous 'void'
  using execution_space = typename ViewTraits<void, Prop...>::execution_space;
  using memory_space    = typename ViewTraits<void, Prop...>::memory_space;
  using HostMirrorSpace = typename ViewTraits<void, Prop...>::HostMirrorSpace;
  using array_layout    = typename ViewTraits<void, Prop...>::array_layout;
  using memory_traits   = typename ViewTraits<void, Prop...>::memory_traits;
  using specialize      = typename ViewTraits<void, Prop...>::specialize;
  using hooks_policy    = typename ViewTraits<void, Prop...>::hooks_policy;
};

template <class HooksPolicy, class... Prop>
struct ViewTraits<
    std::enable_if_t<flare::experimental::is_hooks_policy<HooksPolicy>::value>,
    HooksPolicy, Prop...> {
  using execution_space = typename ViewTraits<void, Prop...>::execution_space;
  using memory_space    = typename ViewTraits<void, Prop...>::memory_space;
  using HostMirrorSpace = typename ViewTraits<void, Prop...>::HostMirrorSpace;
  using array_layout    = typename ViewTraits<void, Prop...>::array_layout;
  using memory_traits   = typename ViewTraits<void, Prop...>::memory_traits;
  using specialize      = typename ViewTraits<void, Prop...>::specialize;
  using hooks_policy    = HooksPolicy;
};

template <class ArrayLayout, class... Prop>
struct ViewTraits<std::enable_if_t<flare::is_array_layout<ArrayLayout>::value>,
                  ArrayLayout, Prop...> {
  // Specify layout, keep subsequent space and memory traits arguments

  using execution_space = typename ViewTraits<void, Prop...>::execution_space;
  using memory_space    = typename ViewTraits<void, Prop...>::memory_space;
  using HostMirrorSpace = typename ViewTraits<void, Prop...>::HostMirrorSpace;
  using array_layout    = ArrayLayout;
  using memory_traits   = typename ViewTraits<void, Prop...>::memory_traits;
  using specialize      = typename ViewTraits<void, Prop...>::specialize;
  using hooks_policy    = typename ViewTraits<void, Prop...>::hooks_policy;
};

template <class Space, class... Prop>
struct ViewTraits<std::enable_if_t<flare::is_space<Space>::value>, Space,
                  Prop...> {
  // Specify Space, memory traits should be the only subsequent argument.

  static_assert(
      std::is_same<typename ViewTraits<void, Prop...>::execution_space,
                   void>::value &&
          std::is_same<typename ViewTraits<void, Prop...>::memory_space,
                       void>::value &&
          std::is_same<typename ViewTraits<void, Prop...>::HostMirrorSpace,
                       void>::value &&
          std::is_same<typename ViewTraits<void, Prop...>::array_layout,
                       void>::value,
      "Only one View Execution or Memory Space template argument");

  using execution_space = typename Space::execution_space;
  using memory_space    = typename Space::memory_space;
  using HostMirrorSpace =
      typename flare::detail::HostMirror<Space>::Space::memory_space;
  using array_layout  = typename execution_space::array_layout;
  using memory_traits = typename ViewTraits<void, Prop...>::memory_traits;
  using specialize    = typename ViewTraits<void, Prop...>::specialize;
  using hooks_policy  = typename ViewTraits<void, Prop...>::hooks_policy;
};

template <class MemoryTraits, class... Prop>
struct ViewTraits<
    std::enable_if_t<flare::is_memory_traits<MemoryTraits>::value>,
    MemoryTraits, Prop...> {
  // Specify memory trait, should not be any subsequent arguments

  static_assert(
      std::is_same<typename ViewTraits<void, Prop...>::execution_space,
                   void>::value &&
          std::is_same<typename ViewTraits<void, Prop...>::memory_space,
                       void>::value &&
          std::is_same<typename ViewTraits<void, Prop...>::array_layout,
                       void>::value &&
          std::is_same<typename ViewTraits<void, Prop...>::memory_traits,
                       void>::value &&
          std::is_same<typename ViewTraits<void, Prop...>::hooks_policy,
                       void>::value,
      "MemoryTrait is the final optional template argument for a View");

  using execution_space = void;
  using memory_space    = void;
  using HostMirrorSpace = void;
  using array_layout    = void;
  using memory_traits   = MemoryTraits;
  using specialize      = void;
  using hooks_policy    = void;
};

template <class DataType, class... Properties>
struct ViewTraits {
 private:
  // Unpack the properties arguments
  using prop = ViewTraits<void, Properties...>;

  using ExecutionSpace =
      std::conditional_t<!std::is_void<typename prop::execution_space>::value,
                         typename prop::execution_space,
                         flare::DefaultExecutionSpace>;

  using MemorySpace =
      std::conditional_t<!std::is_void<typename prop::memory_space>::value,
                         typename prop::memory_space,
                         typename ExecutionSpace::memory_space>;

  using ArrayLayout =
      std::conditional_t<!std::is_void<typename prop::array_layout>::value,
                         typename prop::array_layout,
                         typename ExecutionSpace::array_layout>;

  using HostMirrorSpace = std::conditional_t<
      !std::is_void<typename prop::HostMirrorSpace>::value,
      typename prop::HostMirrorSpace,
      typename flare::detail::HostMirror<ExecutionSpace>::Space>;

  using MemoryTraits =
      std::conditional_t<!std::is_void<typename prop::memory_traits>::value,
                         typename prop::memory_traits,
                         typename flare::MemoryManaged>;

  using HooksPolicy =
      std::conditional_t<!std::is_void<typename prop::hooks_policy>::value,
                         typename prop::hooks_policy,
                         flare::experimental::DefaultViewHooks>;

  // Analyze data type's properties,
  // May be specialized based upon the layout and value type
  using data_analysis = flare::detail::ViewDataAnalysis<DataType, ArrayLayout>;

 public:
  //------------------------------------
  // Data type traits:

  using data_type           = typename data_analysis::type;
  using const_data_type     = typename data_analysis::const_type;
  using non_const_data_type = typename data_analysis::non_const_type;

  //------------------------------------
  // Compatible array of trivial type traits:

  using scalar_array_type = typename data_analysis::scalar_array_type;
  using const_scalar_array_type =
      typename data_analysis::const_scalar_array_type;
  using non_const_scalar_array_type =
      typename data_analysis::non_const_scalar_array_type;

  //------------------------------------
  // Value type traits:

  using value_type           = typename data_analysis::value_type;
  using const_value_type     = typename data_analysis::const_value_type;
  using non_const_value_type = typename data_analysis::non_const_value_type;

  //------------------------------------
  // Mapping traits:

  using array_layout = ArrayLayout;
  using dimension    = typename data_analysis::dimension;

  using specialize = std::conditional_t<
      std::is_void<typename data_analysis::specialize>::value,
      typename prop::specialize,
      typename data_analysis::specialize>; /* mapping specialization tag */

  static constexpr unsigned rank         = dimension::rank;
  static constexpr unsigned rank_dynamic = dimension::rank_dynamic;

  //------------------------------------
  // Execution space, memory space, memory access traits, and host mirror space.

  using execution_space   = ExecutionSpace;
  using memory_space      = MemorySpace;
  using device_type       = flare::Device<ExecutionSpace, MemorySpace>;
  using memory_traits     = MemoryTraits;
  using host_mirror_space = HostMirrorSpace;
  using hooks_policy      = HooksPolicy;

  using size_type = typename MemorySpace::size_type;

  enum { is_hostspace = std::is_same<MemorySpace, HostSpace>::value };
  enum { is_managed = MemoryTraits::is_unmanaged == 0 };
  enum { is_random_access = MemoryTraits::is_random_access == 1 };

  //------------------------------------
};

/** \class View
 *  \brief View to an array of data.
 *
 * A View represents an array of one or more dimensions.
 * For details, please refer to flare' tutorial materials.
 *
 * \section View_TemplateParameters Template parameters
 *
 * This class has both required and optional template parameters.  The
 * \c DataType parameter must always be provided, and must always be
 * first. The parameters \c Arg1Type, \c Arg2Type, and \c Arg3Type are
 * placeholders for different template parameters.  The default value
 * of the fifth template parameter \c Specialize suffices for most use
 * cases.  When explaining the template parameters, we won't refer to
 * \c Arg1Type, \c Arg2Type, and \c Arg3Type; instead, we will refer
 * to the valid categories of template parameters, in whatever order
 * they may occur.
 *
 * Valid ways in which template arguments may be specified:
 *   - View< DataType >
 *   - View< DataType , Layout >
 *   - View< DataType , Layout , Space >
 *   - View< DataType , Layout , Space , MemoryTraits >
 *   - View< DataType , Space >
 *   - View< DataType , Space , MemoryTraits >
 *   - View< DataType , MemoryTraits >
 *
 * \tparam DataType (required) This indicates both the type of each
 *   entry of the array, and the combination of compile-time and
 *   run-time array dimension(s).  For example, <tt>double*</tt>
 *   indicates a one-dimensional array of \c double with run-time
 *   dimension, and <tt>int*[3]</tt> a two-dimensional array of \c int
 *   with run-time first dimension and compile-time second dimension
 *   (of 3).  In general, the run-time dimensions (if any) must go
 *   first, followed by zero or more compile-time dimensions.  For
 *   more examples, please refer to the tutorial materials.
 *
 * \tparam Space (required) The memory space.
 *
 * \tparam Layout (optional) The array's layout in memory.  For
 *   example, LayoutLeft indicates a column-major (Fortran style)
 *   layout, and LayoutRight a row-major (C style) layout.  If not
 *   specified, this defaults to the preferred layout for the
 *   <tt>Space</tt>.
 *
 * \tparam MemoryTraits (optional) Assertion of the user's intended
 *   access behavior.  For example, RandomAccess indicates read-only
 *   access with limited spatial locality, and Unmanaged lets users
 *   wrap externally allocated memory in a View without automatic
 *   deallocation.
 *
 * \section View_MT MemoryTraits discussion
 *
 * \subsection View_MT_Interp MemoryTraits interpretation depends on
 * Space
 *
 * Some \c MemoryTraits options may have different interpretations for
 * different \c Space types.  For example, with the Cuda device,
 * \c RandomAccess tells flare to fetch the data through the texture
 * cache, whereas the non-GPU devices have no such hardware construct.
 *
 * \subsection View_MT_PrefUse Preferred use of MemoryTraits
 *
 * Users should defer applying the optional \c MemoryTraits parameter
 * until the point at which they actually plan to rely on it in a
 * computational kernel.  This minimizes the number of template
 * parameters exposed in their code, which reduces the cost of
 * compilation.  Users may always assign a View without specified
 * \c MemoryTraits to a compatible View with that specification.
 * For example:
 * \code
 * // Pass in the simplest types of View possible.
 * void
 * doSomething (View<double*, Cuda> out,
 *              View<const double*, Cuda> in)
 * {
 *   // Assign the "generic" View in to a RandomAccess View in_rr.
 *   // Note that RandomAccess View objects must have const data.
 *   View<const double*, Cuda, RandomAccess> in_rr = in;
 *   // ... do something with in_rr and out ...
 * }
 * \endcode
 */

}  // namespace flare

namespace flare {

template <class T1, class T2>
struct is_always_assignable_impl;

template <class... ViewTDst, class... ViewTSrc>
struct is_always_assignable_impl<flare::View<ViewTDst...>,
                                 flare::View<ViewTSrc...>> {
  using mapping_type = flare::detail::ViewMapping<
      typename flare::View<ViewTDst...>::traits,
      typename flare::View<ViewTSrc...>::traits,
      typename flare::View<ViewTDst...>::traits::specialize>;

  constexpr static bool value =
      mapping_type::is_assignable &&
      static_cast<int>(flare::View<ViewTDst...>::rank_dynamic) >=
          static_cast<int>(flare::View<ViewTSrc...>::rank_dynamic);
};

template <class View1, class View2>
using is_always_assignable = is_always_assignable_impl<
    std::remove_reference_t<View1>,
    std::remove_const_t<std::remove_reference_t<View2>>>;

template <class T1, class T2>
inline constexpr bool is_always_assignable_v =
    is_always_assignable<T1, T2>::value;

template <class... ViewTDst, class... ViewTSrc>
constexpr bool is_assignable(const flare::View<ViewTDst...>& dst,
                             const flare::View<ViewTSrc...>& src) {
  using DstTraits = typename flare::View<ViewTDst...>::traits;
  using SrcTraits = typename flare::View<ViewTSrc...>::traits;
  using mapping_type =
      flare::detail::ViewMapping<DstTraits, SrcTraits,
                                typename DstTraits::specialize>;

  return is_always_assignable_v<flare::View<ViewTDst...>,
                                flare::View<ViewTSrc...>> ||
         (mapping_type::is_assignable &&
          ((DstTraits::dimension::rank_dynamic >= 1) ||
           (dst.static_extent(0) == src.extent(0))) &&
          ((DstTraits::dimension::rank_dynamic >= 2) ||
           (dst.static_extent(1) == src.extent(1))) &&
          ((DstTraits::dimension::rank_dynamic >= 3) ||
           (dst.static_extent(2) == src.extent(2))) &&
          ((DstTraits::dimension::rank_dynamic >= 4) ||
           (dst.static_extent(3) == src.extent(3))) &&
          ((DstTraits::dimension::rank_dynamic >= 5) ||
           (dst.static_extent(4) == src.extent(4))) &&
          ((DstTraits::dimension::rank_dynamic >= 6) ||
           (dst.static_extent(5) == src.extent(5))) &&
          ((DstTraits::dimension::rank_dynamic >= 7) ||
           (dst.static_extent(6) == src.extent(6))) &&
          ((DstTraits::dimension::rank_dynamic >= 8) ||
           (dst.static_extent(7) == src.extent(7))));
}

} /* namespace flare */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#include <flare/core/tensor/view_mapping.h>
#include <flare/core/tensor/view_array.h>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {


inline constexpr flare::ALL_t ALL{};


inline constexpr flare::detail::WithoutInitializing_t WithoutInitializing{};

inline constexpr flare::detail::AllowPadding_t AllowPadding{};

/** \brief  Create View allocation parameter bundle from argument list.
 *
 *  Valid argument list members are:
 *    1) label as a "string" or std::string
 *    2) memory space instance of the View::memory_space type
 *    3) execution space instance compatible with the View::memory_space
 *    4) flare::WithoutInitializing to bypass initialization
 *    4) flare::AllowPadding to allow allocation to pad dimensions for memory
 * alignment
 */
template <class... Args>
inline detail::ViewCtorProp<typename detail::ViewCtorProp<void, Args>::type...>
view_alloc(Args const&... args) {
  using return_type =
      detail::ViewCtorProp<typename detail::ViewCtorProp<void, Args>::type...>;

  static_assert(!return_type::has_pointer,
                "Cannot give pointer-to-memory for view allocation");

  return return_type(args...);
}

template <class... Args>
FLARE_INLINE_FUNCTION
    detail::ViewCtorProp<typename detail::ViewCtorProp<void, Args>::type...>
    view_wrap(Args const&... args) {
  using return_type =
      detail::ViewCtorProp<typename detail::ViewCtorProp<void, Args>::type...>;

  static_assert(!return_type::has_memory_space &&
                    !return_type::has_execution_space &&
                    !return_type::has_label && return_type::has_pointer,
                "Must only give pointer-to-memory for view wrapping");

  return return_type(args...);
}

} /* namespace flare */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {

template <class DataType, class... Properties>
class View;

template <class>
struct is_view : public std::false_type {};

template <class D, class... P>
struct is_view<View<D, P...>> : public std::true_type {};

template <class D, class... P>
struct is_view<const View<D, P...>> : public std::true_type {};

template <class T>
inline constexpr bool is_view_v = is_view<T>::value;

template <class DataType, class... Properties>
class View : public ViewTraits<DataType, Properties...> {
 private:
  template <class, class...>
  friend class View;
  template <class, class...>
  friend class flare::detail::ViewMapping;

  using view_tracker_type = flare::detail::ViewTracker<View>;

 public:
  using traits = ViewTraits<DataType, Properties...>;

 private:
  using map_type =
      flare::detail::ViewMapping<traits, typename traits::specialize>;
  template <typename V>
  friend struct flare::detail::ViewTracker;
  using hooks_policy = typename traits::hooks_policy;

  view_tracker_type m_track;
  map_type m_map;

 public:
  //----------------------------------------
  /** \brief  Compatible view of array of scalar types */
  using array_type =
      View<typename traits::scalar_array_type, typename traits::array_layout,
           typename traits::device_type, typename traits::hooks_policy,
           typename traits::memory_traits>;

  /** \brief  Compatible view of const data type */
  using const_type =
      View<typename traits::const_data_type, typename traits::array_layout,
           typename traits::device_type, typename traits::hooks_policy,
           typename traits::memory_traits>;

  /** \brief  Compatible view of non-const data type */
  using non_const_type =
      View<typename traits::non_const_data_type, typename traits::array_layout,
           typename traits::device_type, typename traits::hooks_policy,
           typename traits::memory_traits>;

  /** \brief  Compatible HostMirror view */
  using HostMirror =
      View<typename traits::non_const_data_type, typename traits::array_layout,
           Device<DefaultHostExecutionSpace,
                  typename traits::host_mirror_space::memory_space>,
           typename traits::hooks_policy>;

  /** \brief  Compatible HostMirror view */
  using host_mirror_type =
      View<typename traits::non_const_data_type, typename traits::array_layout,
           typename traits::host_mirror_space, typename traits::hooks_policy>;

  /** \brief Unified types */
  using uniform_type = typename detail::ViewUniformType<View, 0>::type;
  using uniform_const_type =
      typename detail::ViewUniformType<View, 0>::const_type;
  using uniform_runtime_type =
      typename detail::ViewUniformType<View, 0>::runtime_type;
  using uniform_runtime_const_type =
      typename detail::ViewUniformType<View, 0>::runtime_const_type;
  using uniform_nomemspace_type =
      typename detail::ViewUniformType<View, 0>::nomemspace_type;
  using uniform_const_nomemspace_type =
      typename detail::ViewUniformType<View, 0>::const_nomemspace_type;
  using uniform_runtime_nomemspace_type =
      typename detail::ViewUniformType<View, 0>::runtime_nomemspace_type;
  using uniform_runtime_const_nomemspace_type =
      typename detail::ViewUniformType<View, 0>::runtime_const_nomemspace_type;

  //----------------------------------------
  // Domain rank and extents

  static constexpr detail::integral_constant<size_t, traits::dimension::rank>
      rank = {};
  static constexpr detail::integral_constant<size_t,
                                           traits::dimension::rank_dynamic>
      rank_dynamic = {};

  template <typename iType>
  FLARE_INLINE_FUNCTION constexpr std::enable_if_t<
      std::is_integral<iType>::value, size_t>
  extent(const iType& r) const noexcept {
    return m_map.extent(r);
  }

  static FLARE_INLINE_FUNCTION constexpr size_t static_extent(
      const unsigned r) noexcept {
    return map_type::static_extent(r);
  }

  template <typename iType>
  FLARE_INLINE_FUNCTION constexpr std::enable_if_t<
      std::is_integral<iType>::value, int>
  extent_int(const iType& r) const noexcept {
    return static_cast<int>(m_map.extent(r));
  }

  FLARE_INLINE_FUNCTION constexpr typename traits::array_layout layout()
      const {
    return m_map.layout();
  }

  //----------------------------------------
  /*  Deprecate all 'dimension' functions in favor of
   *  ISO/C++ vocabulary 'extent'.
   */

  FLARE_INLINE_FUNCTION constexpr size_t size() const {
    return m_map.dimension_0() * m_map.dimension_1() * m_map.dimension_2() *
           m_map.dimension_3() * m_map.dimension_4() * m_map.dimension_5() *
           m_map.dimension_6() * m_map.dimension_7();
  }

  FLARE_INLINE_FUNCTION constexpr size_t stride_0() const {
    return m_map.stride_0();
  }
  FLARE_INLINE_FUNCTION constexpr size_t stride_1() const {
    return m_map.stride_1();
  }
  FLARE_INLINE_FUNCTION constexpr size_t stride_2() const {
    return m_map.stride_2();
  }
  FLARE_INLINE_FUNCTION constexpr size_t stride_3() const {
    return m_map.stride_3();
  }
  FLARE_INLINE_FUNCTION constexpr size_t stride_4() const {
    return m_map.stride_4();
  }
  FLARE_INLINE_FUNCTION constexpr size_t stride_5() const {
    return m_map.stride_5();
  }
  FLARE_INLINE_FUNCTION constexpr size_t stride_6() const {
    return m_map.stride_6();
  }
  FLARE_INLINE_FUNCTION constexpr size_t stride_7() const {
    return m_map.stride_7();
  }

  template <typename iType>
  FLARE_INLINE_FUNCTION constexpr std::enable_if_t<
      std::is_integral<iType>::value, size_t>
  stride(iType r) const {
    return (
        r == 0
            ? m_map.stride_0()
            : (r == 1
                   ? m_map.stride_1()
                   : (r == 2
                          ? m_map.stride_2()
                          : (r == 3
                                 ? m_map.stride_3()
                                 : (r == 4
                                        ? m_map.stride_4()
                                        : (r == 5
                                               ? m_map.stride_5()
                                               : (r == 6
                                                      ? m_map.stride_6()
                                                      : m_map.stride_7())))))));
  }

  template <typename iType>
  FLARE_INLINE_FUNCTION void stride(iType* const s) const {
    m_map.stride(s);
  }

  //----------------------------------------
  // Range span is the span which contains all members.

  using reference_type = typename map_type::reference_type;
  using pointer_type   = typename map_type::pointer_type;

  enum {
    reference_type_is_lvalue_reference =
        std::is_lvalue_reference<reference_type>::value
  };

  FLARE_INLINE_FUNCTION constexpr size_t span() const { return m_map.span(); }
  FLARE_INLINE_FUNCTION bool span_is_contiguous() const {
    return m_map.span_is_contiguous();
  }
  FLARE_INLINE_FUNCTION constexpr bool is_allocated() const {
    return m_map.data() != nullptr;
  }
  FLARE_INLINE_FUNCTION constexpr pointer_type data() const {
    return m_map.data();
  }

  //----------------------------------------
  // Allow specializations to query their specialized map

  FLARE_INLINE_FUNCTION
  const flare::detail::ViewMapping<traits, typename traits::specialize>&
  impl_map() const {
    return m_map;
  }
  FLARE_INLINE_FUNCTION
  const flare::detail::SharedAllocationTracker& impl_track() const {
    return m_track.m_tracker;
  }
  //----------------------------------------

 private:
  static constexpr bool is_layout_left =
      std::is_same<typename traits::array_layout, flare::LayoutLeft>::value;

  static constexpr bool is_layout_right =
      std::is_same<typename traits::array_layout, flare::LayoutRight>::value;

  static constexpr bool is_layout_stride =
      std::is_same<typename traits::array_layout, flare::LayoutStride>::value;

  static constexpr bool is_default_map =
      std::is_void<typename traits::specialize>::value &&
      (is_layout_left || is_layout_right || is_layout_stride);

#if defined(FLARE_ENABLE_DEBUG_BOUNDS_CHECK)

#define FLARE_IMPL_VIEW_OPERATOR_VERIFY(...)                               \
  flare::detail::runtime_check_memory_access_violation<                      \
      typename traits::memory_space>(                                       \
      "flare::View ERROR: attempt to access inaccessible memory space",    \
      __VA_ARGS__);                                                         \
  flare::detail::view_verify_operator_bounds<typename traits::memory_space>( \
      __VA_ARGS__);

#else

#define FLARE_IMPL_VIEW_OPERATOR_VERIFY(...)                            \
  flare::detail::runtime_check_memory_access_violation<                   \
      typename traits::memory_space>(                                    \
      "flare::View ERROR: attempt to access inaccessible memory space", \
      __VA_ARGS__);

#endif

  template <typename... Is>
  static FLARE_FUNCTION void check_access_member_function_valid_args(Is...) {
    static_assert(rank <= sizeof...(Is), "");
    static_assert(sizeof...(Is) <= 8, "");
    static_assert(flare::detail::are_integral<Is...>::value, "");
  }

  template <typename... Is>
  static FLARE_FUNCTION void check_operator_parens_valid_args(Is...) {
    static_assert(rank == sizeof...(Is), "");
    static_assert(flare::detail::are_integral<Is...>::value, "");
  }

 public:
  //------------------------------
  // Rank 1 default map operator()

  template <typename I0>
  FLARE_FORCEINLINE_FUNCTION
      std::enable_if_t<(flare::detail::always_true<I0>::value &&  //
                        (1 == rank) && is_default_map && !is_layout_stride),
                       reference_type>
      operator()(I0 i0) const {
    check_operator_parens_valid_args(i0);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0)
    return m_map.m_impl_handle[i0];
  }

  template <typename I0>
  FLARE_FORCEINLINE_FUNCTION
      std::enable_if_t<(flare::detail::always_true<I0>::value &&  //
                        (1 == rank) && is_default_map && is_layout_stride),
                       reference_type>
      operator()(I0 i0) const {
    check_operator_parens_valid_args(i0);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0)
    return m_map.m_impl_handle[m_map.m_impl_offset.m_stride.S0 * i0];
  }

  //------------------------------
  // Rank 1 operator[]

  template <typename I0>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      ((1 == rank) && flare::detail::are_integral<I0>::value && !is_default_map),
      reference_type>
  operator[](I0 i0) const {
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0)
    return m_map.reference(i0);
  }

  template <typename I0>
  FLARE_FORCEINLINE_FUNCTION
      std::enable_if_t<((1 == rank) && flare::detail::are_integral<I0>::value &&
                        is_default_map && !is_layout_stride),
                       reference_type>
      operator[](I0 i0) const {
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0)
    return m_map.m_impl_handle[i0];
  }

  template <typename I0>
  FLARE_FORCEINLINE_FUNCTION
      std::enable_if_t<((1 == rank) && flare::detail::are_integral<I0>::value &&
                        is_default_map && is_layout_stride),
                       reference_type>
      operator[](I0 i0) const {
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0)
    return m_map.m_impl_handle[m_map.m_impl_offset.m_stride.S0 * i0];
  }

  //------------------------------
  // Rank 2 default map operator()

  template <typename I0, typename I1>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<I0, I1>::value &&  //
       (2 == rank) && is_default_map && is_layout_left && (rank_dynamic == 0)),
      reference_type>
  operator()(I0 i0, I1 i1) const {
    check_operator_parens_valid_args(i0, i1);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1)
    return m_map.m_impl_handle[i0 + m_map.m_impl_offset.m_dim.N0 * i1];
  }

  template <typename I0, typename I1>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<I0, I1>::value &&  //
       (2 == rank) && is_default_map && is_layout_left && (rank_dynamic != 0)),
      reference_type>
  operator()(I0 i0, I1 i1) const {
    check_operator_parens_valid_args(i0, i1);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1)
    return m_map.m_impl_handle[i0 + m_map.m_impl_offset.m_stride * i1];
  }

  template <typename I0, typename I1>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<I0, I1>::value &&  //
       (2 == rank) && is_default_map && is_layout_right && (rank_dynamic == 0)),
      reference_type>
  operator()(I0 i0, I1 i1) const {
    check_operator_parens_valid_args(i0, i1);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1)
    return m_map.m_impl_handle[i1 + m_map.m_impl_offset.m_dim.N1 * i0];
  }

  template <typename I0, typename I1>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<I0, I1>::value &&  //
       (2 == rank) && is_default_map && is_layout_right && (rank_dynamic != 0)),
      reference_type>
  operator()(I0 i0, I1 i1) const {
    check_operator_parens_valid_args(i0, i1);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1)
    return m_map.m_impl_handle[i1 + m_map.m_impl_offset.m_stride * i0];
  }

  template <typename I0, typename I1>
  FLARE_FORCEINLINE_FUNCTION
      std::enable_if_t<(flare::detail::always_true<I0, I1>::value &&  //
                        (2 == rank) && is_default_map && is_layout_stride),
                       reference_type>
      operator()(I0 i0, I1 i1) const {
    check_operator_parens_valid_args(i0, i1);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1)
    return m_map.m_impl_handle[i0 * m_map.m_impl_offset.m_stride.S0 +
                               i1 * m_map.m_impl_offset.m_stride.S1];
  }

  // Rank 0 -> 8 operator() except for rank-1 and rank-2 with default map which
  // have "inlined" versions above

  template <typename... Is>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<Is...>::value &&  //
       (2 != rank) && (1 != rank) && (0 != rank) && is_default_map),
      reference_type>
  operator()(Is... indices) const {
    check_operator_parens_valid_args(indices...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, indices...)
    return m_map.m_impl_handle[m_map.m_impl_offset(indices...)];
  }

  template <typename... Is>
  FLARE_FORCEINLINE_FUNCTION
      std::enable_if_t<(flare::detail::always_true<Is...>::value &&  //
                        ((0 == rank) || !is_default_map)),
                       reference_type>
      operator()(Is... indices) const {
    check_operator_parens_valid_args(indices...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, indices...)
    return m_map.reference(indices...);
  }

  //------------------------------
  // Rank 0

  template <typename... Is>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<Is...>::value && (0 == rank)), reference_type>
  access(Is... extra) const {
    check_access_member_function_valid_args(extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, extra...)
    return m_map.reference();
  }

  //------------------------------
  // Rank 1

  template <typename I0, typename... Is>
  FLARE_FORCEINLINE_FUNCTION
      std::enable_if_t<(flare::detail::always_true<I0, Is...>::value &&
                        (1 == rank) && !is_default_map),
                       reference_type>
      access(I0 i0, Is... extra) const {
    check_access_member_function_valid_args(i0, extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, extra...)
    return m_map.reference(i0);
  }

  template <typename I0, typename... Is>
  FLARE_FORCEINLINE_FUNCTION
      std::enable_if_t<(flare::detail::always_true<I0, Is...>::value &&
                        (1 == rank) && is_default_map && !is_layout_stride),
                       reference_type>
      access(I0 i0, Is... extra) const {
    check_access_member_function_valid_args(i0, extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, extra...)
    return m_map.m_impl_handle[i0];
  }

  template <typename I0, typename... Is>
  FLARE_FORCEINLINE_FUNCTION
      std::enable_if_t<(flare::detail::always_true<I0, Is...>::value &&
                        (1 == rank) && is_default_map && is_layout_stride),
                       reference_type>
      access(I0 i0, Is... extra) const {
    check_access_member_function_valid_args(i0, extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, extra...)
    return m_map.m_impl_handle[m_map.m_impl_offset.m_stride.S0 * i0];
  }

  //------------------------------
  // Rank 2

  template <typename I0, typename I1, typename... Is>
  FLARE_FORCEINLINE_FUNCTION
      std::enable_if_t<(flare::detail::always_true<I0, I1, Is...>::value &&
                        (2 == rank) && !is_default_map),
                       reference_type>
      access(I0 i0, I1 i1, Is... extra) const {
    check_access_member_function_valid_args(i0, i1, extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, extra...)
    return m_map.reference(i0, i1);
  }

  template <typename I0, typename I1, typename... Is>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<I0, I1, Is...>::value && (2 == rank) &&
       is_default_map && is_layout_left && (rank_dynamic == 0)),
      reference_type>
  access(I0 i0, I1 i1, Is... extra) const {
    check_access_member_function_valid_args(i0, i1, extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, extra...)
    return m_map.m_impl_handle[i0 + m_map.m_impl_offset.m_dim.N0 * i1];
  }

  template <typename I0, typename I1, typename... Is>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<I0, I1, Is...>::value && (2 == rank) &&
       is_default_map && is_layout_left && (rank_dynamic != 0)),
      reference_type>
  access(I0 i0, I1 i1, Is... extra) const {
    check_access_member_function_valid_args(i0, i1, extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, extra...)
    return m_map.m_impl_handle[i0 + m_map.m_impl_offset.m_stride * i1];
  }

  template <typename I0, typename I1, typename... Is>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<I0, I1, Is...>::value && (2 == rank) &&
       is_default_map && is_layout_right && (rank_dynamic == 0)),
      reference_type>
  access(I0 i0, I1 i1, Is... extra) const {
    check_access_member_function_valid_args(i0, i1, extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, extra...)
    return m_map.m_impl_handle[i1 + m_map.m_impl_offset.m_dim.N1 * i0];
  }

  template <typename I0, typename I1, typename... Is>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<I0, I1, Is...>::value && (2 == rank) &&
       is_default_map && is_layout_right && (rank_dynamic != 0)),
      reference_type>
  access(I0 i0, I1 i1, Is... extra) const {
    check_access_member_function_valid_args(i0, i1, extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, extra...)
    return m_map.m_impl_handle[i1 + m_map.m_impl_offset.m_stride * i0];
  }

  template <typename I0, typename I1, typename... Is>
  FLARE_FORCEINLINE_FUNCTION
      std::enable_if_t<(flare::detail::always_true<I0, I1, Is...>::value &&
                        (2 == rank) && is_default_map && is_layout_stride),
                       reference_type>
      access(I0 i0, I1 i1, Is... extra) const {
    check_access_member_function_valid_args(i0, i1, extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, extra...)
    return m_map.m_impl_handle[i0 * m_map.m_impl_offset.m_stride.S0 +
                               i1 * m_map.m_impl_offset.m_stride.S1];
  }

  //------------------------------
  // Rank 3

  template <typename I0, typename I1, typename I2, typename... Is>
  FLARE_FORCEINLINE_FUNCTION
      std::enable_if_t<(flare::detail::always_true<I0, I1, I2, Is...>::value &&
                        (3 == rank) && is_default_map),
                       reference_type>
      access(I0 i0, I1 i1, I2 i2, Is... extra) const {
    check_access_member_function_valid_args(i0, i1, i2, extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, i2, extra...)
    return m_map.m_impl_handle[m_map.m_impl_offset(i0, i1, i2)];
  }

  template <typename I0, typename I1, typename I2, typename... Is>
  FLARE_FORCEINLINE_FUNCTION
      std::enable_if_t<(flare::detail::always_true<I0, I1, I2, Is...>::value &&
                        (3 == rank) && !is_default_map),
                       reference_type>
      access(I0 i0, I1 i1, I2 i2, Is... extra) const {
    check_access_member_function_valid_args(i0, i1, i2, extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, i2, extra...)
    return m_map.reference(i0, i1, i2);
  }

  //------------------------------
  // Rank 4

  template <typename I0, typename I1, typename I2, typename I3, typename... Is>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<I0, I1, I2, I3, Is...>::value && (4 == rank) &&
       is_default_map),
      reference_type>
  access(I0 i0, I1 i1, I2 i2, I3 i3, Is... extra) const {
    check_access_member_function_valid_args(i0, i1, i2, i3, extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, i2, i3, extra...)
    return m_map.m_impl_handle[m_map.m_impl_offset(i0, i1, i2, i3)];
  }

  template <typename I0, typename I1, typename I2, typename I3, typename... Is>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<I0, I1, I2, I3, Is...>::value && (4 == rank) &&
       !is_default_map),
      reference_type>
  access(I0 i0, I1 i1, I2 i2, I3 i3, Is... extra) const {
    check_access_member_function_valid_args(i0, i1, i2, i3, extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, i2, i3, extra...)
    return m_map.reference(i0, i1, i2, i3);
  }

  //------------------------------
  // Rank 5

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename... Is>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<I0, I1, I2, I3, I4, Is...>::value &&
       (5 == rank) && is_default_map),
      reference_type>
  access(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, Is... extra) const {
    check_access_member_function_valid_args(i0, i1, i2, i3, i4, extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, i2, i3, i4,
                                     extra...)
    return m_map.m_impl_handle[m_map.m_impl_offset(i0, i1, i2, i3, i4)];
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename... Is>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<I0, I1, I2, I3, I4, Is...>::value &&
       (5 == rank) && !is_default_map),
      reference_type>
  access(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, Is... extra) const {
    check_access_member_function_valid_args(i0, i1, i2, i3, i4, extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, i2, i3, i4,
                                     extra...)
    return m_map.reference(i0, i1, i2, i3, i4);
  }

  //------------------------------
  // Rank 6

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename... Is>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<I0, I1, I2, I3, I4, I5, Is...>::value &&
       (6 == rank) && is_default_map),
      reference_type>
  access(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5, Is... extra) const {
    check_access_member_function_valid_args(i0, i1, i2, i3, i4, i5, extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, i2, i3, i4, i5,
                                     extra...)
    return m_map.m_impl_handle[m_map.m_impl_offset(i0, i1, i2, i3, i4, i5)];
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename... Is>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<I0, I1, I2, I3, I4, I5, Is...>::value &&
       (6 == rank) && !is_default_map),
      reference_type>
  access(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5, Is... extra) const {
    check_access_member_function_valid_args(i0, i1, i2, i3, i4, i5, extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, i2, i3, i4, i5,
                                     extra...)
    return m_map.reference(i0, i1, i2, i3, i4, i5);
  }

  //------------------------------
  // Rank 7

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename... Is>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<I0, I1, I2, I3, I4, I5, I6, Is...>::value &&
       (7 == rank) && is_default_map),
      reference_type>
  access(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5, I6 i6, Is... extra) const {
    check_access_member_function_valid_args(i0, i1, i2, i3, i4, i5, i6,
                                            extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, i2, i3, i4, i5, i6,
                                     extra...)
    return m_map.m_impl_handle[m_map.m_impl_offset(i0, i1, i2, i3, i4, i5, i6)];
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename... Is>
  FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
      (flare::detail::always_true<I0, I1, I2, I3, I4, I5, I6, Is...>::value &&
       (7 == rank) && !is_default_map),
      reference_type>
  access(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5, I6 i6, Is... extra) const {
    check_access_member_function_valid_args(i0, i1, i2, i3, i4, i5, i6,
                                            extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, i2, i3, i4, i5, i6,
                                     extra...)
    return m_map.reference(i0, i1, i2, i3, i4, i5, i6);
  }

  //------------------------------
  // Rank 8

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename I7, typename... Is>
  FLARE_FORCEINLINE_FUNCTION
      std::enable_if_t<(flare::detail::always_true<I0, I1, I2, I3, I4, I5, I6,
                                                  I7, Is...>::value &&
                        (8 == rank) && is_default_map),
                       reference_type>
      access(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5, I6 i6, I7 i7,
             Is... extra) const {
    check_access_member_function_valid_args(i0, i1, i2, i3, i4, i5, i6, i7,
                                            extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, i2, i3, i4, i5, i6,
                                     i7, extra...)
    return m_map
        .m_impl_handle[m_map.m_impl_offset(i0, i1, i2, i3, i4, i5, i6, i7)];
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename I7, typename... Is>
  FLARE_FORCEINLINE_FUNCTION
      std::enable_if_t<(flare::detail::always_true<I0, I1, I2, I3, I4, I5, I6,
                                                  I7, Is...>::value &&
                        (8 == rank) && !is_default_map),
                       reference_type>
      access(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5, I6 i6, I7 i7,
             Is... extra) const {
    check_access_member_function_valid_args(i0, i1, i2, i3, i4, i5, i6, i7,
                                            extra...);
    FLARE_IMPL_VIEW_OPERATOR_VERIFY(m_track, m_map, i0, i1, i2, i3, i4, i5, i6,
                                     i7, extra...)
    return m_map.reference(i0, i1, i2, i3, i4, i5, i6, i7);
  }

#undef FLARE_IMPL_VIEW_OPERATOR_VERIFY

  //----------------------------------------
  // Standard destructor, constructors, and assignment operators

  FLARE_DEFAULTED_FUNCTION
  ~View() = default;

  FLARE_DEFAULTED_FUNCTION
  View() = default;

  FLARE_FUNCTION
  View(const View& other) : m_track(other.m_track), m_map(other.m_map) {
    FLARE_IF_ON_HOST((hooks_policy::copy_construct(*this, other);))
  }

  FLARE_FUNCTION
  View(View&& other)
      : m_track{std::move(other.m_track)}, m_map{std::move(other.m_map)} {
    FLARE_IF_ON_HOST((hooks_policy::move_construct(*this, other);))
  }

  FLARE_FUNCTION
  View& operator=(const View& other) {
    m_map   = other.m_map;
    m_track = other.m_track;

    FLARE_IF_ON_HOST((hooks_policy::copy_assign(*this, other);))

    return *this;
  }

  FLARE_FUNCTION
  View& operator=(View&& other) {
    m_map   = std::move(other.m_map);
    m_track = std::move(other.m_track);

    FLARE_IF_ON_HOST((hooks_policy::move_assign(*this, other);))

    return *this;
  }

  //----------------------------------------
  // Compatible view copy constructor and assignment
  // may assign unmanaged from managed.

  template <class RT, class... RP>
  FLARE_INLINE_FUNCTION View(
      const View<RT, RP...>& rhs,
      std::enable_if_t<flare::detail::ViewMapping<
          traits, typename View<RT, RP...>::traits,
          typename traits::specialize>::is_assignable_data_type>* = nullptr)
      : m_track(rhs), m_map() {
    using SrcTraits = typename View<RT, RP...>::traits;
    using Mapping   = flare::detail::ViewMapping<traits, SrcTraits,
                                              typename traits::specialize>;
    static_assert(Mapping::is_assignable,
                  "Incompatible View copy construction");
    Mapping::assign(m_map, rhs.m_map, rhs.m_track.m_tracker);
  }

  template <class RT, class... RP>
  FLARE_INLINE_FUNCTION std::enable_if_t<
      flare::detail::ViewMapping<
          traits, typename View<RT, RP...>::traits,
          typename traits::specialize>::is_assignable_data_type,
      View>&
  operator=(const View<RT, RP...>& rhs) {
    using SrcTraits = typename View<RT, RP...>::traits;
    using Mapping   = flare::detail::ViewMapping<traits, SrcTraits,
                                              typename traits::specialize>;
    static_assert(Mapping::is_assignable, "Incompatible View copy assignment");
    Mapping::assign(m_map, rhs.m_map, rhs.m_track.m_tracker);
    m_track.assign(rhs);
    return *this;
  }

  //----------------------------------------
  // Compatible subview constructor
  // may assign unmanaged from managed.

  template <class RT, class... RP, class Arg0, class... Args>
  FLARE_INLINE_FUNCTION View(const View<RT, RP...>& src_view, const Arg0 arg0,
                              Args... args)
      : m_track(src_view), m_map() {
    using SrcType = View<RT, RP...>;

    using Mapping = flare::detail::ViewMapping<void, typename SrcType::traits,
                                              Arg0, Args...>;

    using DstType = typename Mapping::type;

    static_assert(
        flare::detail::ViewMapping<traits, typename DstType::traits,
                                  typename traits::specialize>::is_assignable,
        "Subview construction requires compatible view and subview arguments");

    Mapping::assign(m_map, src_view.m_map, arg0, args...);
  }

  //----------------------------------------
  // Allocation tracking properties

  FLARE_INLINE_FUNCTION
  int use_count() const { return m_track.m_tracker.use_count(); }

  inline const std::string label() const {
    return m_track.m_tracker
        .template get_label<typename traits::memory_space>();
  }

 public:
  //----------------------------------------
  // Allocation according to allocation properties and array layout

  template <class... P>
  explicit inline View(
      const detail::ViewCtorProp<P...>& arg_prop,
      std::enable_if_t<!detail::ViewCtorProp<P...>::has_pointer,
                       typename traits::array_layout> const& arg_layout)
      : m_track(), m_map() {
    // Copy the input allocation properties with possibly defaulted properties
    // We need to split it in two to avoid MSVC compiler errors
    auto prop_copy_tmp =
        detail::with_properties_if_unset(arg_prop, std::string{});
    auto prop_copy = detail::with_properties_if_unset(
        prop_copy_tmp, typename traits::device_type::memory_space{},
        typename traits::device_type::execution_space{});
    using alloc_prop = decltype(prop_copy);

    static_assert(traits::is_managed,
                  "View allocation constructor requires managed memory");

    if (alloc_prop::initialize &&
        !alloc_prop::execution_space::impl_is_initialized()) {
      // If initializing view data then
      // the execution space must be initialized.
      flare::detail::throw_runtime_exception(
          "Constructing View and initializing data with uninitialized "
          "execution space");
    }

    size_t i0 = arg_layout.dimension[0];
    size_t i1 = arg_layout.dimension[1];
    size_t i2 = arg_layout.dimension[2];
    size_t i3 = arg_layout.dimension[3];
    size_t i4 = arg_layout.dimension[4];
    size_t i5 = arg_layout.dimension[5];
    size_t i6 = arg_layout.dimension[6];
    size_t i7 = arg_layout.dimension[7];

    const std::string& alloc_name =
        detail::get_property<detail::LabelTag>(prop_copy);
    detail::runtime_check_rank(
        rank, rank_dynamic,
        std::is_same<typename traits::specialize, void>::value, i0, i1, i2, i3,
        i4, i5, i6, i7, alloc_name);

    flare::detail::SharedAllocationRecord<>* record = m_map.allocate_shared(
        prop_copy, arg_layout, detail::ViewCtorProp<P...>::has_execution_space);

    // Setup and initialization complete, start tracking
    m_track.m_tracker.assign_allocated_record_to_uninitialized(record);
  }

  FLARE_INLINE_FUNCTION
  void assign_data(pointer_type arg_data) {
    m_track.m_tracker.clear();
    m_map.assign_data(arg_data);
  }

  // Wrap memory according to properties and array layout
  template <class... P>
  explicit FLARE_INLINE_FUNCTION View(
      const detail::ViewCtorProp<P...>& arg_prop,
      std::enable_if_t<detail::ViewCtorProp<P...>::has_pointer,
                       typename traits::array_layout> const& arg_layout)
      : m_track()  // No memory tracking
        ,
        m_map(arg_prop, arg_layout) {
    static_assert(
        std::is_same<pointer_type,
                     typename detail::ViewCtorProp<P...>::pointer_type>::value,
        "Constructing View to wrap user memory must supply matching pointer "
        "type");
  }

  // Simple dimension-only layout
  template <class... P>
  explicit inline View(
      const detail::ViewCtorProp<P...>& arg_prop,
      std::enable_if_t<!detail::ViewCtorProp<P...>::has_pointer, size_t> const
          arg_N0          = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N7 = FLARE_IMPL_CTOR_DEFAULT_ARG)
      : View(arg_prop,
             typename traits::array_layout(arg_N0, arg_N1, arg_N2, arg_N3,
                                           arg_N4, arg_N5, arg_N6, arg_N7)) {
    static_assert(traits::array_layout::is_extent_constructible,
                  "Layout is not constructible from extent arguments. Use "
                  "overload taking a layout object instead.");
  }

  template <class... P>
  explicit FLARE_INLINE_FUNCTION View(
      const detail::ViewCtorProp<P...>& arg_prop,
      std::enable_if_t<detail::ViewCtorProp<P...>::has_pointer, size_t> const
          arg_N0          = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N7 = FLARE_IMPL_CTOR_DEFAULT_ARG)
      : View(arg_prop,
             typename traits::array_layout(arg_N0, arg_N1, arg_N2, arg_N3,
                                           arg_N4, arg_N5, arg_N6, arg_N7)) {
    static_assert(traits::array_layout::is_extent_constructible,
                  "Layout is not constructible from extent arguments. Use "
                  "overload taking a layout object instead.");
  }

  // Allocate with label and layout
  template <typename Label>
  explicit inline View(
      const Label& arg_label,
      std::enable_if_t<flare::detail::is_view_label<Label>::value,
                       typename traits::array_layout> const& arg_layout)
      : View(detail::ViewCtorProp<std::string>(arg_label), arg_layout) {}

  // Allocate label and layout, must disambiguate from subview constructor.
  template <typename Label>
  explicit inline View(
      const Label& arg_label,
      std::enable_if_t<flare::detail::is_view_label<Label>::value, const size_t>
          arg_N0          = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N7 = FLARE_IMPL_CTOR_DEFAULT_ARG)
      : View(detail::ViewCtorProp<std::string>(arg_label),
             typename traits::array_layout(arg_N0, arg_N1, arg_N2, arg_N3,
                                           arg_N4, arg_N5, arg_N6, arg_N7)) {
    static_assert(traits::array_layout::is_extent_constructible,
                  "Layout is not constructible from extent arguments. Use "
                  "overload taking a layout object instead.");
  }

  // Construct view from ViewTracker and map
  // This should be the preferred method because future extensions may need to
  // use the ViewTracker class.
  template <class Traits>
  FLARE_INLINE_FUNCTION View(
      const view_tracker_type& track,
      const flare::detail::ViewMapping<Traits, typename Traits::specialize>& map)
      : m_track(track), m_map() {
    using Mapping =
        flare::detail::ViewMapping<traits, Traits, typename traits::specialize>;
    static_assert(Mapping::is_assignable,
                  "Incompatible View copy construction");
    Mapping::assign(m_map, map, track.m_tracker);
  }

  // Construct View from internal shared allocation tracker object and map
  // This is here for backwards compatibility for classes that derive from
  // flare::View
  template <class Traits>
  FLARE_INLINE_FUNCTION View(
      const typename view_tracker_type::track_type& track,
      const flare::detail::ViewMapping<Traits, typename Traits::specialize>& map)
      : m_track(track), m_map() {
    using Mapping =
        flare::detail::ViewMapping<traits, Traits, typename traits::specialize>;
    static_assert(Mapping::is_assignable,
                  "Incompatible View copy construction");
    Mapping::assign(m_map, map, track);
  }

  //----------------------------------------
  // Memory span required to wrap these dimensions.
  static constexpr size_t required_allocation_size(
      typename traits::array_layout const& layout) {
    return map_type::memory_span(layout);
  }

  static constexpr size_t required_allocation_size(
      const size_t arg_N0 = 0, const size_t arg_N1 = 0, const size_t arg_N2 = 0,
      const size_t arg_N3 = 0, const size_t arg_N4 = 0, const size_t arg_N5 = 0,
      const size_t arg_N6 = 0, const size_t arg_N7 = 0) {
    static_assert(traits::array_layout::is_extent_constructible,
                  "Layout is not constructible from extent arguments. Use "
                  "overload taking a layout object instead.");
    return map_type::memory_span(typename traits::array_layout(
        arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7));
  }

  explicit FLARE_INLINE_FUNCTION View(
      pointer_type arg_ptr, const size_t arg_N0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N7 = FLARE_IMPL_CTOR_DEFAULT_ARG)
      : View(detail::ViewCtorProp<pointer_type>(arg_ptr),
             typename traits::array_layout(arg_N0, arg_N1, arg_N2, arg_N3,
                                           arg_N4, arg_N5, arg_N6, arg_N7)) {
    static_assert(traits::array_layout::is_extent_constructible,
                  "Layout is not constructible from extent arguments. Use "
                  "overload taking a layout object instead.");
  }

  explicit FLARE_INLINE_FUNCTION View(
      pointer_type arg_ptr, const typename traits::array_layout& arg_layout)
      : View(detail::ViewCtorProp<pointer_type>(arg_ptr), arg_layout) {}

  //----------------------------------------
  // Shared scratch memory constructor

  static FLARE_INLINE_FUNCTION size_t
  shmem_size(const size_t arg_N0 = FLARE_INVALID_INDEX,
             const size_t arg_N1 = FLARE_INVALID_INDEX,
             const size_t arg_N2 = FLARE_INVALID_INDEX,
             const size_t arg_N3 = FLARE_INVALID_INDEX,
             const size_t arg_N4 = FLARE_INVALID_INDEX,
             const size_t arg_N5 = FLARE_INVALID_INDEX,
             const size_t arg_N6 = FLARE_INVALID_INDEX,
             const size_t arg_N7 = FLARE_INVALID_INDEX) {
    static_assert(traits::array_layout::is_extent_constructible,
                  "Layout is not constructible from extent arguments. Use "
                  "overload taking a layout object instead.");
    const size_t num_passed_args = detail::count_valid_integers(
        arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7);

    if (std::is_void<typename traits::specialize>::value &&
        num_passed_args != rank_dynamic) {
      flare::abort(
          "flare::View::shmem_size() rank_dynamic != number of arguments.\n");
    }

    return View::shmem_size(typename traits::array_layout(
        arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7));
  }

 private:
  // Want to be able to align to minimum scratch alignment or sizeof or alignof
  // elements
  static constexpr size_t scratch_value_alignment =
      max({sizeof(typename traits::value_type),
           alignof(typename traits::value_type),
           static_cast<size_t>(
               traits::execution_space::scratch_memory_space::ALIGN)});

 public:
  static FLARE_INLINE_FUNCTION size_t
  shmem_size(typename traits::array_layout const& arg_layout) {
    return map_type::memory_span(arg_layout) + scratch_value_alignment;
  }

  explicit FLARE_INLINE_FUNCTION View(
      const typename traits::execution_space::scratch_memory_space& arg_space,
      const typename traits::array_layout& arg_layout)
      : View(detail::ViewCtorProp<pointer_type>(reinterpret_cast<pointer_type>(
                 arg_space.get_shmem_aligned(map_type::memory_span(arg_layout),
                                             scratch_value_alignment))),
             arg_layout) {}

  explicit FLARE_INLINE_FUNCTION View(
      const typename traits::execution_space::scratch_memory_space& arg_space,
      const size_t arg_N0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
      const size_t arg_N7 = FLARE_IMPL_CTOR_DEFAULT_ARG)
      : View(detail::ViewCtorProp<pointer_type>(
                 reinterpret_cast<pointer_type>(arg_space.get_shmem_aligned(
                     map_type::memory_span(typename traits::array_layout(
                         arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6,
                         arg_N7)),
                     scratch_value_alignment))),
             typename traits::array_layout(arg_N0, arg_N1, arg_N2, arg_N3,
                                           arg_N4, arg_N5, arg_N6, arg_N7)) {
    static_assert(traits::array_layout::is_extent_constructible,
                  "Layout is not constructible from extent arguments. Use "
                  "overload taking a layout object instead.");
  }
};

template <typename D, class... P>
FLARE_INLINE_FUNCTION constexpr unsigned rank(const View<D, P...>&) {
  return View<D, P...>::rank();
}

namespace detail {

template <typename ValueType, unsigned int Rank>
struct RankDataType {
  using type = typename RankDataType<ValueType, Rank - 1>::type*;
};

template <typename ValueType>
struct RankDataType<ValueType, 0> {
  using type = ValueType;
};

template <unsigned N, typename... Args>
FLARE_FUNCTION std::enable_if_t<
    N == View<Args...>::rank() &&
        std::is_same<typename ViewTraits<Args...>::specialize, void>::value,
    View<Args...>>
as_view_of_rank_n(View<Args...> v) {
  return v;
}

// Placeholder implementation to compile generic code for DynRankView; should
// never be called
template <unsigned N, typename T, typename... Args>
FLARE_FUNCTION std::enable_if_t<
    N != View<T, Args...>::rank() &&
        std::is_same<typename ViewTraits<T, Args...>::specialize, void>::value,
    View<typename RankDataType<typename View<T, Args...>::value_type, N>::type,
         Args...>>
as_view_of_rank_n(View<T, Args...>) {
  flare::abort("Trying to get at a View of the wrong rank");
  return {};
}

template <typename Function, typename... Args>
void apply_to_view_of_static_rank(Function&& f, View<Args...> a) {
  f(a);
}

}  // namespace detail
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace detail {
template <class ValueType, class TypeList>
struct TypeListToViewTraits;

template <class ValueType, class... Properties>
struct TypeListToViewTraits<ValueType, flare::detail::type_list<Properties...>> {
  using type = ViewTraits<ValueType, Properties...>;
};

// It is not safe to assume that subviews of views with the Aligned memory trait
// are also aligned. Hence, just remove that attribute for subviews.
template <class D, class... P>
struct RemoveAlignedMemoryTrait {
 private:
  using type_list_in  = flare::detail::type_list<P...>;
  using memory_traits = typename ViewTraits<D, P...>::memory_traits;
  using type_list_in_wo_memory_traits =
      typename flare::detail::type_list_remove_first<memory_traits,
                                                    type_list_in>::type;
  using new_memory_traits =
      flare::MemoryTraits<memory_traits::impl_value & ~flare::Aligned>;
  using new_type_list = typename flare::detail::concat_type_list<
      type_list_in_wo_memory_traits,
      flare::detail::type_list<new_memory_traits>>::type;

 public:
  using type = typename TypeListToViewTraits<D, new_type_list>::type;
};
}  // namespace detail

template <class D, class... P, class... Args>
FLARE_INLINE_FUNCTION auto subview(const View<D, P...>& src, Args... args) {
  static_assert(View<D, P...>::rank == sizeof...(Args),
                "subview requires one argument for each source View rank");

  return typename flare::detail::ViewMapping<
      void /* deduce subview type from source view traits */
      ,
      typename detail::RemoveAlignedMemoryTrait<D, P...>::type,
      Args...>::type(src, args...);
}


template <class V, class... Args>
using Subview = decltype(subview(std::declval<V>(), std::declval<Args>()...));

} /* namespace flare */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {

template <class LT, class... LP, class RT, class... RP>
FLARE_INLINE_FUNCTION bool operator==(const View<LT, LP...>& lhs,
                                       const View<RT, RP...>& rhs) {
  // Same data, layout, dimensions
  using lhs_traits = ViewTraits<LT, LP...>;
  using rhs_traits = ViewTraits<RT, RP...>;

  return std::is_same<typename lhs_traits::const_value_type,
                      typename rhs_traits::const_value_type>::value &&
         std::is_same<typename lhs_traits::array_layout,
                      typename rhs_traits::array_layout>::value &&
         std::is_same<typename lhs_traits::memory_space,
                      typename rhs_traits::memory_space>::value &&
         View<LT, LP...>::rank() == View<RT, RP...>::rank() &&
         lhs.data() == rhs.data() && lhs.span() == rhs.span() &&
         lhs.extent(0) == rhs.extent(0) && lhs.extent(1) == rhs.extent(1) &&
         lhs.extent(2) == rhs.extent(2) && lhs.extent(3) == rhs.extent(3) &&
         lhs.extent(4) == rhs.extent(4) && lhs.extent(5) == rhs.extent(5) &&
         lhs.extent(6) == rhs.extent(6) && lhs.extent(7) == rhs.extent(7);
}

template <class LT, class... LP, class RT, class... RP>
FLARE_INLINE_FUNCTION bool operator!=(const View<LT, LP...>& lhs,
                                       const View<RT, RP...>& rhs) {
  return !(operator==(lhs, rhs));
}

} /* namespace flare */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {
namespace detail {

inline void shared_allocation_tracking_disable() {
  flare::detail::SharedAllocationRecord<void, void>::tracking_disable();
}

inline void shared_allocation_tracking_enable() {
  flare::detail::SharedAllocationRecord<void, void>::tracking_enable();
}

} /* namespace detail */
} /* namespace flare */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {
namespace detail {

template <class Specialize, typename A, typename B>
struct CommonViewValueType;

template <typename A, typename B>
struct CommonViewValueType<void, A, B> {
  using value_type = std::common_type_t<A, B>;
};

template <class Specialize, class ValueType>
struct CommonViewAllocProp;

template <class ValueType>
struct CommonViewAllocProp<void, ValueType> {
  using value_type        = ValueType;
  using scalar_array_type = ValueType;

  template <class... Views>
  FLARE_INLINE_FUNCTION CommonViewAllocProp(const Views&...) {}
};

template <class... Views>
struct DeduceCommonViewAllocProp;

// Base case must provide types for:
// 1. specialize  2. value_type  3. is_view  4. prop_type
template <class FirstView>
struct DeduceCommonViewAllocProp<FirstView> {
  using specialize = typename FirstView::traits::specialize;

  using value_type = typename FirstView::traits::value_type;

  enum : bool { is_view = is_view<FirstView>::value };

  using prop_type = CommonViewAllocProp<specialize, value_type>;
};

template <class FirstView, class... NextViews>
struct DeduceCommonViewAllocProp<FirstView, NextViews...> {
  using NextTraits = DeduceCommonViewAllocProp<NextViews...>;

  using first_specialize = typename FirstView::traits::specialize;
  using first_value_type = typename FirstView::traits::value_type;

  enum : bool { first_is_view = is_view<FirstView>::value };

  using next_specialize = typename NextTraits::specialize;
  using next_value_type = typename NextTraits::value_type;

  enum : bool { next_is_view = NextTraits::is_view };

  // common types

  // determine specialize type
  // if first and next specialize differ, but are not the same specialize, error
  // out
  static_assert(!(!std::is_same<first_specialize, next_specialize>::value &&
                  !std::is_void<first_specialize>::value &&
                  !std::is_void<next_specialize>::value),
                "flare DeduceCommonViewAllocProp ERROR: Only one non-void "
                "specialize trait allowed");

  // otherwise choose non-void specialize if either/both are non-void
  using specialize = std::conditional_t<
      std::is_same<first_specialize, next_specialize>::value, first_specialize,
      std::conditional_t<(std::is_void<first_specialize>::value &&
                          !std::is_void<next_specialize>::value),
                         next_specialize, first_specialize>>;

  using value_type = typename CommonViewValueType<specialize, first_value_type,
                                                  next_value_type>::value_type;

  enum : bool { is_view = (first_is_view && next_is_view) };

  using prop_type = CommonViewAllocProp<specialize, value_type>;
};

}  // end namespace detail

template <class... Views>
using DeducedCommonPropsType =
    typename detail::DeduceCommonViewAllocProp<Views...>::prop_type;

// This function is required in certain scenarios where users customize
// flare View internals. One example are dynamic length embedded ensemble
// types. The function is used to propagate necessary information
// (like the ensemble size) when creating new views.
// However, most of the time it is called with a single view.
// Furthermore, the propagated information is not just for view allocations.
// From what I can tell, the type of functionality provided by
// common_view_alloc_prop is the equivalent of propagating accessors in mdspan,
// a mechanism we will eventually use to replace this clunky approach here, when
// we are finally mdspan based.
// TODO: get rid of this when we have mdspan
template <class... Views>
FLARE_INLINE_FUNCTION DeducedCommonPropsType<Views...> common_view_alloc_prop(
    Views const&... views) {
  return DeducedCommonPropsType<Views...>(views...);
}

}  // namespace flare

#include <flare/core/tensor/view_uniform_type.h>
#include <flare/core/common/atomic_view.h>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif  // FLARE_CORE_TENSOR_VIEW_H_

