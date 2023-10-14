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

#ifndef FLARE_OFFSET_TENSOR_H_
#define FLARE_OFFSET_TENSOR_H_

#include <flare/core.h>

#include <flare/core/tensor/tensor.h>

namespace flare {

    namespace experimental {
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

        template<class DataType, class... Properties>
        class OffsetTensor;

        template<class>
        struct is_offset_tensor : public std::false_type {
        };

        template<class D, class... P>
        struct is_offset_tensor<OffsetTensor<D, P...>> : public std::true_type {
        };

        template<class D, class... P>
        struct is_offset_tensor<const OffsetTensor<D, P...>> : public std::true_type {
        };

        template<class T>
        inline constexpr bool is_offset_tensor_v = is_offset_tensor<T>::value;

#define FLARE_INVALID_OFFSET int64_t(0x7FFFFFFFFFFFFFFFLL)
#define FLARE_INVALID_INDEX_RANGE \
  { FLARE_INVALID_OFFSET, FLARE_INVALID_OFFSET }

        template<typename iType, std::enable_if_t<std::is_integral<iType>::value &&
                                                  std::is_signed<iType>::value,
                iType> = 0>
        using IndexRange = flare::Array<iType, 2>;

        using index_list_type = std::initializer_list<int64_t>;

        //  template <typename iType,
        //    std::enable_if_t< std::is_integral<iType>::value &&
        //      std::is_signed<iType>::value, iType > = 0> using min_index_type =
        //      std::initializer_list<iType>;

        namespace detail {

            template<class TensorType>
            struct GetOffsetTensorTypeFromTensorType {
                using type =
                        OffsetTensor<typename TensorType::data_type, typename TensorType::array_layout,
                                typename TensorType::device_type,
                                typename TensorType::memory_traits>;
            };

            template<unsigned, class MapType, class BeginsType>
            FLARE_INLINE_FUNCTION bool offsettensor_verify_operator_bounds(
                    const MapType &, const BeginsType &) {
                return true;
            }

            template<unsigned R, class MapType, class BeginsType, class iType,
                    class... Args>
            FLARE_INLINE_FUNCTION bool offsettensor_verify_operator_bounds(
                    const MapType &map, const BeginsType &begins, const iType &i,
                    Args... args) {
                const bool legalIndex =
                        (int64_t(i) >= begins[R]) &&
                        (int64_t(i) <= int64_t(begins[R] + map.extent(R) - 1));
                return legalIndex &&
                       offsettensor_verify_operator_bounds<R + 1>(map, begins, args...);
            }

            template<unsigned, class MapType, class BeginsType>
            inline void offsettensor_error_operator_bounds(char *, int, const MapType &,
                                                           const BeginsType &) {}

            template<unsigned R, class MapType, class BeginsType, class iType,
                    class... Args>
            inline void offsettensor_error_operator_bounds(char *buf, int len,
                                                           const MapType &map,
                                                           const BeginsType begins,
                                                           const iType &i, Args... args) {
                const int64_t b = begins[R];
                const int64_t e = b + map.extent(R) - 1;
                const int n =
                        snprintf(buf, len, " %ld <= %ld <= %ld %c", static_cast<unsigned long>(b),
                                 static_cast<unsigned long>(i), static_cast<unsigned long>(e),
                                 (sizeof...(Args) ? ',' : ')'));
                offsettensor_error_operator_bounds < R + 1 >(buf + n, len - n, map, begins,
                        args...);
            }

            template<class MemorySpace, class MapType, class BeginsType, class... Args>
            FLARE_INLINE_FUNCTION void offsettensor_verify_operator_bounds(
                    flare::detail::SharedAllocationTracker const &tracker, const MapType &map,
                    const BeginsType &begins, Args... args) {
                if (!offsettensor_verify_operator_bounds<0>(map, begins, args...)) {
                    FLARE_IF_ON_HOST(
                    (enum { LEN = 1024 }; char buffer[LEN];
                            const std::string label = tracker.template get_label<MemorySpace>();
                            int n                   = snprintf(buffer, LEN,
                            "OffsetTensor bounds error of tensor labeled %s (",
                            label.c_str());
                            offsettensor_error_operator_bounds<0>(buffer + n, LEN - n, map, begins,
                            args...);
                            flare::detail::throw_runtime_exception(std::string(buffer));))

                    FLARE_IF_ON_DEVICE((
                                               /* Check #1: is there a SharedAllocationRecord?
                                                 (we won't use it, but if it is not there then there isn't
                                                  a corresponding SharedAllocationHeader containing a label).
                                                 This check should cover the case of Tensors that don't
                                                 have the Unmanaged trait but were initialized by pointer. */
                                               if (tracker.has_record()) {
                                               flare::detail::operator_bounds_error_on_device(map);
                                       } else { flare::abort("OffsetTensor bounds error"); }))
                }
            }

            inline void runtime_check_rank_host(const size_t rank_dynamic,
                                                const size_t rank,
                                                const index_list_type minIndices,
                                                const std::string &label) {
                bool isBad = false;
                std::string message =
                        "flare::experimental::OffsetTensor ERROR: for OffsetTensor labeled '" +
                        label + "':";
                if (rank_dynamic != rank) {
                    message +=
                            "The full rank must be the same as the dynamic rank. full rank = ";
                    message += std::to_string(rank) +
                               " dynamic rank = " + std::to_string(rank_dynamic) + "\n";
                    isBad = true;
                }

                size_t numOffsets = 0;
                for (size_t i = 0; i < minIndices.size(); ++i) {
                    if (minIndices.begin()[i] != FLARE_INVALID_OFFSET) numOffsets++;
                }
                if (numOffsets != rank_dynamic) {
                    message += "The number of offsets provided ( " +
                               std::to_string(numOffsets) +
                               " ) must equal the dynamic rank ( " +
                               std::to_string(rank_dynamic) + " ).";
                    isBad = true;
                }

                if (isBad) flare::abort(message.c_str());
            }

            FLARE_INLINE_FUNCTION
            void runtime_check_rank_device(const size_t rank_dynamic, const size_t rank,
                                           const index_list_type minIndices) {
                if (rank_dynamic != rank) {
                    flare::abort(
                            "The full rank of an OffsetTensor must be the same as the dynamic rank.");
                }
                size_t numOffsets = 0;
                for (size_t i = 0; i < minIndices.size(); ++i) {
                    if (minIndices.begin()[i] != FLARE_INVALID_OFFSET) numOffsets++;
                }
                if (numOffsets != rank) {
                    flare::abort(
                            "The number of offsets provided to an OffsetTensor constructor must "
                            "equal the dynamic rank.");
                }
            }
        }  // namespace detail

        template<class DataType, class... Properties>
        class OffsetTensor : public TensorTraits<DataType, Properties...> {
        public:
            using traits = TensorTraits<DataType, Properties...>;

        private:
            template<class, class...>
            friend
            class OffsetTensor;

            template<class, class...>
            friend
            class Tensor;  // FIXME delete this line
            template<class, class...>
            friend
            class flare::detail::TensorMapping;

            using map_type = flare::detail::TensorMapping<traits, void>;
            using track_type = flare::detail::SharedAllocationTracker;

        public:
            enum {
                Rank = map_type::Rank
            };
            using begins_type = flare::Array<int64_t, Rank>;

            template<typename iType,
                    std::enable_if_t<std::is_integral<iType>::value, iType> = 0>
            FLARE_FUNCTION int64_t begin(const iType local_dimension) const {
                return local_dimension < Rank ? m_begins[local_dimension]
                                              : FLARE_INVALID_OFFSET;
            }

            FLARE_FUNCTION
            begins_type begins() const { return m_begins; }

            template<typename iType,
                    std::enable_if_t<std::is_integral<iType>::value, iType> = 0>
            FLARE_FUNCTION int64_t end(const iType local_dimension) const {
                return begin(local_dimension) + m_map.extent(local_dimension);
            }

        private:
            track_type m_track;
            map_type m_map;
            begins_type m_begins;

        public:
            //----------------------------------------
            /** \brief  Compatible tensor of array of scalar types */
            using array_type =
                    OffsetTensor<typename traits::scalar_array_type,
                            typename traits::array_layout, typename traits::device_type,
                            typename traits::memory_traits>;

            /** \brief  Compatible tensor of const data type */
            using const_type =
                    OffsetTensor<typename traits::const_data_type,
                            typename traits::array_layout, typename traits::device_type,
                            typename traits::memory_traits>;

            /** \brief  Compatible tensor of non-const data type */
            using non_const_type =
                    OffsetTensor<typename traits::non_const_data_type,
                            typename traits::array_layout, typename traits::device_type,
                            typename traits::memory_traits>;

            /** \brief  Compatible HostMirror tensor */
            using HostMirror = OffsetTensor<typename traits::non_const_data_type,
                    typename traits::array_layout,
                    typename traits::host_mirror_space>;

            //----------------------------------------
            // Domain rank and extents

            /** \brief rank() to be implemented
             */
            // FLARE_FUNCTION
            // static
            // constexpr unsigned rank() { return map_type::Rank; }

            template<typename iType>
            FLARE_FUNCTION constexpr std::enable_if_t<std::is_integral<iType>::value,
                    size_t>
            extent(const iType &r) const {
                return m_map.extent(r);
            }

            template<typename iType>
            FLARE_FUNCTION constexpr std::enable_if_t<std::is_integral<iType>::value,
                    int>
            extent_int(const iType &r) const {
                return static_cast<int>(m_map.extent(r));
            }

            FLARE_FUNCTION constexpr typename traits::array_layout layout() const {
                return m_map.layout();
            }

            FLARE_FUNCTION constexpr size_t size() const {
                return m_map.dimension_0() * m_map.dimension_1() * m_map.dimension_2() *
                       m_map.dimension_3() * m_map.dimension_4() * m_map.dimension_5() *
                       m_map.dimension_6() * m_map.dimension_7();
            }

            FLARE_FUNCTION constexpr size_t stride_0() const { return m_map.stride_0(); }

            FLARE_FUNCTION constexpr size_t stride_1() const { return m_map.stride_1(); }

            FLARE_FUNCTION constexpr size_t stride_2() const { return m_map.stride_2(); }

            FLARE_FUNCTION constexpr size_t stride_3() const { return m_map.stride_3(); }

            FLARE_FUNCTION constexpr size_t stride_4() const { return m_map.stride_4(); }

            FLARE_FUNCTION constexpr size_t stride_5() const { return m_map.stride_5(); }

            FLARE_FUNCTION constexpr size_t stride_6() const { return m_map.stride_6(); }

            FLARE_FUNCTION constexpr size_t stride_7() const { return m_map.stride_7(); }

            template<typename iType>
            FLARE_FUNCTION constexpr std::enable_if_t<std::is_integral<iType>::value,
                    size_t>
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

            template<typename iType>
            FLARE_FUNCTION void stride(iType *const s) const {
                m_map.stride(s);
            }

            //----------------------------------------
            // Range span is the span which contains all members.

            using reference_type = typename map_type::reference_type;
            using pointer_type = typename map_type::pointer_type;

            enum {
                reference_type_is_lvalue_reference =
                std::is_lvalue_reference<reference_type>::value
            };

            FLARE_FUNCTION constexpr size_t span() const { return m_map.span(); }

            FLARE_FUNCTION bool span_is_contiguous() const {
                return m_map.span_is_contiguous();
            }

            FLARE_FUNCTION constexpr bool is_allocated() const {
                return m_map.data() != nullptr;
            }

            FLARE_FUNCTION constexpr pointer_type data() const { return m_map.data(); }

            //----------------------------------------
            // Allow specializations to query their specialized map

            FLARE_FUNCTION
            const flare::detail::TensorMapping<traits, void> &implementation_map() const {
                return m_map;
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

#define FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY(ARG)                      \
  flare::detail::runtime_check_memory_access_violation<                   \
      typename traits::memory_space>(                                    \
      "flare::OffsetTensor ERROR: attempt to access inaccessible memory " \
      "space");                                                          \
  flare::experimental::detail::offsettensor_verify_operator_bounds<         \
      typename traits::memory_space>                                     \
      ARG;

#else

#define FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY(ARG)                      \
  flare::detail::runtime_check_memory_access_violation<                   \
      typename traits::memory_space>(                                    \
      "flare::OffsetTensor ERROR: attempt to access inaccessible memory " \
      "space");

#endif
        public:
            //------------------------------
            // Rank 0 operator()

            FLARE_FORCEINLINE_FUNCTION
            reference_type operator()() const { return m_map.reference(); }
            //------------------------------
            // Rank 1 operator()

            template<typename I0>
            FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
                    (flare::detail::are_integral<I0>::value && (1 == Rank) && !is_default_map),
                    reference_type>
            operator()(const I0 &i0) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY((m_track, m_map, m_begins, i0))
                const size_t j0 = i0 - m_begins[0];
                return m_map.reference(j0);
            }

            template<typename I0>
            FLARE_FORCEINLINE_FUNCTION
            std::enable_if_t<(flare::detail::are_integral<I0>::value && (1 == Rank) &&
                              is_default_map && !is_layout_stride),
                    reference_type>
            operator()(const I0 &i0) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY((m_track, m_map, m_begins, i0))
                const size_t j0 = i0 - m_begins[0];
                return m_map.m_impl_handle[j0];
            }

            template<typename I0>
            FLARE_FORCEINLINE_FUNCTION
            std::enable_if_t<(flare::detail::are_integral<I0>::value && (1 == Rank) &&
                              is_default_map && is_layout_stride),
                    reference_type>
            operator()(const I0 &i0) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY((m_track, m_map, m_begins, i0))
                const size_t j0 = i0 - m_begins[0];
                return m_map.m_impl_handle[m_map.m_impl_offset.m_stride.S0 * j0];
            }
            //------------------------------
            // Rank 1 operator[]

            template<typename I0>
            FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
                    (flare::detail::are_integral<I0>::value && (1 == Rank) && !is_default_map),
                    reference_type>
            operator[](const I0 &i0) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY((m_track, m_map, m_begins, i0))
                const size_t j0 = i0 - m_begins[0];
                return m_map.reference(j0);
            }

            template<typename I0>
            FLARE_FORCEINLINE_FUNCTION
            std::enable_if_t<(flare::detail::are_integral<I0>::value && (1 == Rank) &&
                              is_default_map && !is_layout_stride),
                    reference_type>
            operator[](const I0 &i0) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY((m_track, m_map, m_begins, i0))
                const size_t j0 = i0 - m_begins[0];
                return m_map.m_impl_handle[j0];
            }

            template<typename I0>
            FLARE_FORCEINLINE_FUNCTION
            std::enable_if_t<(flare::detail::are_integral<I0>::value && (1 == Rank) &&
                              is_default_map && is_layout_stride),
                    reference_type>
            operator[](const I0 &i0) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY((m_track, m_map, m_begins, i0))
                const size_t j0 = i0 - m_begins[0];
                return m_map.m_impl_handle[m_map.m_impl_offset.m_stride.S0 * j0];
            }

            //------------------------------
            // Rank 2

            template<typename I0, typename I1>
            FLARE_FORCEINLINE_FUNCTION
            std::enable_if_t<(flare::detail::are_integral<I0, I1>::value &&
                              (2 == Rank) && !is_default_map),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY((m_track, m_map, m_begins, i0, i1))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                return m_map.reference(j0, j1);
            }

            template<typename I0, typename I1>
            FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
                    (flare::detail::are_integral<I0, I1>::value && (2 == Rank) &&
                     is_default_map && is_layout_left && (traits::rank_dynamic == 0)),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY((m_track, m_map, m_begins, i0, i1))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                return m_map.m_impl_handle[j0 + m_map.m_impl_offset.m_dim.N0 * j1];
            }

            template<typename I0, typename I1>
            FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
                    (flare::detail::are_integral<I0, I1>::value && (2 == Rank) &&
                     is_default_map && is_layout_left && (traits::rank_dynamic != 0)),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY((m_track, m_map, m_begins, i0, i1))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                return m_map.m_impl_handle[j0 + m_map.m_impl_offset.m_stride * j1];
            }

            template<typename I0, typename I1>
            FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
                    (flare::detail::are_integral<I0, I1>::value && (2 == Rank) &&
                     is_default_map && is_layout_right && (traits::rank_dynamic == 0)),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY((m_track, m_map, m_begins, i0, i1))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                return m_map.m_impl_handle[j1 + m_map.m_impl_offset.m_dim.N1 * j0];
            }

            template<typename I0, typename I1>
            FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
                    (flare::detail::are_integral<I0, I1>::value && (2 == Rank) &&
                     is_default_map && is_layout_right && (traits::rank_dynamic != 0)),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY((m_track, m_map, m_begins, i0, i1))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                return m_map.m_impl_handle[j1 + m_map.m_impl_offset.m_stride * j0];
            }

            template<typename I0, typename I1>
            FLARE_FORCEINLINE_FUNCTION
            std::enable_if_t<(flare::detail::are_integral<I0, I1>::value &&
                              (2 == Rank) && is_default_map && is_layout_stride),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY((m_track, m_map, m_begins, i0, i1))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                return m_map.m_impl_handle[j0 * m_map.m_impl_offset.m_stride.S0 +
                                           j1 * m_map.m_impl_offset.m_stride.S1];
            }

            //------------------------------
            // Rank 3

            template<typename I0, typename I1, typename I2>
            FLARE_FORCEINLINE_FUNCTION
            std::enable_if_t<(flare::detail::are_integral<I0, I1, I2>::value &&
                              (3 == Rank) && is_default_map),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1, const I2 &i2) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY(
                        (m_track, m_map, m_begins, i0, i1, i2))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                const size_t j2 = i2 - m_begins[2];
                return m_map.m_impl_handle[m_map.m_impl_offset(j0, j1, j2)];
            }

            template<typename I0, typename I1, typename I2>
            FLARE_FORCEINLINE_FUNCTION
            std::enable_if_t<(flare::detail::are_integral<I0, I1, I2>::value &&
                              (3 == Rank) && !is_default_map),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1, const I2 &i2) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY(
                        (m_track, m_map, m_begins, i0, i1, i2))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                const size_t j2 = i2 - m_begins[2];
                return m_map.reference(j0, j1, j2);
            }

            //------------------------------
            // Rank 4

            template<typename I0, typename I1, typename I2, typename I3>
            FLARE_FORCEINLINE_FUNCTION
            std::enable_if_t<(flare::detail::are_integral<I0, I1, I2, I3>::value &&
                              (4 == Rank) && is_default_map),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY(
                        (m_track, m_map, m_begins, i0, i1, i2, i3))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                const size_t j2 = i2 - m_begins[2];
                const size_t j3 = i3 - m_begins[3];
                return m_map.m_impl_handle[m_map.m_impl_offset(j0, j1, j2, j3)];
            }

            template<typename I0, typename I1, typename I2, typename I3>
            FLARE_FORCEINLINE_FUNCTION
            std::enable_if_t<(flare::detail::are_integral<I0, I1, I2, I3>::value &&
                              (4 == Rank) && !is_default_map),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY(
                        (m_track, m_map, m_begins, i0, i1, i2, i3))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                const size_t j2 = i2 - m_begins[2];
                const size_t j3 = i3 - m_begins[3];
                return m_map.reference(j0, j1, j2, j3);
            }

            //------------------------------
            // Rank 5

            template<typename I0, typename I1, typename I2, typename I3, typename I4>
            FLARE_FORCEINLINE_FUNCTION
            std::enable_if_t<(flare::detail::are_integral<I0, I1, I2, I3, I4>::value &&
                              (5 == Rank) && is_default_map),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
                       const I4 &i4) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY(
                        (m_track, m_map, m_begins, i0, i1, i2, i3, i4))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                const size_t j2 = i2 - m_begins[2];
                const size_t j3 = i3 - m_begins[3];
                const size_t j4 = i4 - m_begins[4];
                return m_map.m_impl_handle[m_map.m_impl_offset(j0, j1, j2, j3, j4)];
            }

            template<typename I0, typename I1, typename I2, typename I3, typename I4>
            FLARE_FORCEINLINE_FUNCTION
            std::enable_if_t<(flare::detail::are_integral<I0, I1, I2, I3, I4>::value &&
                              (5 == Rank) && !is_default_map),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
                       const I4 &i4) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY(
                        (m_track, m_map, m_begins, i0, i1, i2, i3, i4))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                const size_t j2 = i2 - m_begins[2];
                const size_t j3 = i3 - m_begins[3];
                const size_t j4 = i4 - m_begins[4];
                return m_map.reference(j0, j1, j2, j3, j4);
            }

            //------------------------------
            // Rank 6

            template<typename I0, typename I1, typename I2, typename I3, typename I4,
                    typename I5>
            FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
                    (flare::detail::are_integral<I0, I1, I2, I3, I4, I5>::value &&
                     (6 == Rank) && is_default_map),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
                       const I4 &i4, const I5 &i5) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY(
                        (m_track, m_map, m_begins, i0, i1, i2, i3, i4, i5))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                const size_t j2 = i2 - m_begins[2];
                const size_t j3 = i3 - m_begins[3];
                const size_t j4 = i4 - m_begins[4];
                const size_t j5 = i5 - m_begins[5];
                return m_map.m_impl_handle[m_map.m_impl_offset(j0, j1, j2, j3, j4, j5)];
            }

            template<typename I0, typename I1, typename I2, typename I3, typename I4,
                    typename I5>
            FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
                    (flare::detail::are_integral<I0, I1, I2, I3, I4, I5>::value &&
                     (6 == Rank) && !is_default_map),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
                       const I4 &i4, const I5 &i5) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY(
                        (m_track, m_map, m_begins, i0, i1, i2, i3, i4, i5))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                const size_t j2 = i2 - m_begins[2];
                const size_t j3 = i3 - m_begins[3];
                const size_t j4 = i4 - m_begins[4];
                const size_t j5 = i5 - m_begins[5];
                return m_map.reference(j0, j1, j2, j3, j4, j5);
            }

            //------------------------------
            // Rank 7

            template<typename I0, typename I1, typename I2, typename I3, typename I4,
                    typename I5, typename I6>
            FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
                    (flare::detail::are_integral<I0, I1, I2, I3, I4, I5, I6>::value &&
                     (7 == Rank) && is_default_map),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
                       const I4 &i4, const I5 &i5, const I6 &i6) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY(
                        (m_track, m_map, m_begins, i0, i1, i2, i3, i4, i5, i6))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                const size_t j2 = i2 - m_begins[2];
                const size_t j3 = i3 - m_begins[3];
                const size_t j4 = i4 - m_begins[4];
                const size_t j5 = i5 - m_begins[5];
                const size_t j6 = i6 - m_begins[6];
                return m_map.m_impl_handle[m_map.m_impl_offset(j0, j1, j2, j3, j4, j5, j6)];
            }

            template<typename I0, typename I1, typename I2, typename I3, typename I4,
                    typename I5, typename I6>
            FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
                    (flare::detail::are_integral<I0, I1, I2, I3, I4, I5, I6>::value &&
                     (7 == Rank) && !is_default_map),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
                       const I4 &i4, const I5 &i5, const I6 &i6) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY(
                        (m_track, m_map, m_begins, i0, i1, i2, i3, i4, i5, i6))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                const size_t j2 = i2 - m_begins[2];
                const size_t j3 = i3 - m_begins[3];
                const size_t j4 = i4 - m_begins[4];
                const size_t j5 = i5 - m_begins[5];
                const size_t j6 = i6 - m_begins[6];
                return m_map.reference(j0, j1, j2, j3, j4, j5, j6);
            }

            //------------------------------
            // Rank 8

            template<typename I0, typename I1, typename I2, typename I3, typename I4,
                    typename I5, typename I6, typename I7>
            FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
                    (flare::detail::are_integral<I0, I1, I2, I3, I4, I5, I6, I7>::value &&
                     (8 == Rank) && is_default_map),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
                       const I4 &i4, const I5 &i5, const I6 &i6, const I7 &i7) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY(
                        (m_track, m_map, m_begins, i0, i1, i2, i3, i4, i5, i6, i7))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                const size_t j2 = i2 - m_begins[2];
                const size_t j3 = i3 - m_begins[3];
                const size_t j4 = i4 - m_begins[4];
                const size_t j5 = i5 - m_begins[5];
                const size_t j6 = i6 - m_begins[6];
                const size_t j7 = i7 - m_begins[7];
                return m_map
                        .m_impl_handle[m_map.m_impl_offset(j0, j1, j2, j3, j4, j5, j6, j7)];
            }

            template<typename I0, typename I1, typename I2, typename I3, typename I4,
                    typename I5, typename I6, typename I7>
            FLARE_FORCEINLINE_FUNCTION std::enable_if_t<
                    (flare::detail::are_integral<I0, I1, I2, I3, I4, I5, I6, I7>::value &&
                     (8 == Rank) && !is_default_map),
                    reference_type>
            operator()(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
                       const I4 &i4, const I5 &i5, const I6 &i6, const I7 &i7) const {
                FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY(
                        (m_track, m_map, m_begins, i0, i1, i2, i3, i4, i5, i6, i7))
                const size_t j0 = i0 - m_begins[0];
                const size_t j1 = i1 - m_begins[1];
                const size_t j2 = i2 - m_begins[2];
                const size_t j3 = i3 - m_begins[3];
                const size_t j4 = i4 - m_begins[4];
                const size_t j5 = i5 - m_begins[5];
                const size_t j6 = i6 - m_begins[6];
                const size_t j7 = i7 - m_begins[7];
                return m_map.reference(j0, j1, j2, j3, j4, j5, j6, j7);
            }

#undef FLARE_IMPL_OFFSETTENSOR_OPERATOR_VERIFY

            //----------------------------------------
            // Standard destructor, constructors, and assignment operators

            FLARE_DEFAULTED_FUNCTION
            ~OffsetTensor() = default;

            FLARE_FUNCTION
            OffsetTensor() : m_track(), m_map() {
                for (size_t i = 0; i < Rank; ++i) m_begins[i] = FLARE_INVALID_OFFSET;
            }

            FLARE_FUNCTION
            OffsetTensor(const OffsetTensor &rhs)
                    : m_track(rhs.m_track, traits::is_managed),
                      m_map(rhs.m_map),
                      m_begins(rhs.m_begins) {}

            FLARE_FUNCTION
            OffsetTensor(OffsetTensor &&rhs)
                    : m_track(std::move(rhs.m_track)),
                      m_map(std::move(rhs.m_map)),
                      m_begins(std::move(rhs.m_begins)) {}

            FLARE_FUNCTION
            OffsetTensor &operator=(const OffsetTensor &rhs) {
                m_track = rhs.m_track;
                m_map = rhs.m_map;
                m_begins = rhs.m_begins;
                return *this;
            }

            FLARE_FUNCTION
            OffsetTensor &operator=(OffsetTensor &&rhs) {
                m_track = std::move(rhs.m_track);
                m_map = std::move(rhs.m_map);
                m_begins = std::move(rhs.m_begins);
                return *this;
            }

            // interoperability with Tensor
        private:
            using tensor_type =
                    Tensor<typename traits::scalar_array_type, typename traits::array_layout,
                            typename traits::device_type, typename traits::memory_traits>;

        public:
            FLARE_FUNCTION
            tensor_type tensor() const {
                tensor_type v(m_track, m_map);
                return v;
            }

            template<class RT, class... RP>
            FLARE_FUNCTION OffsetTensor(const Tensor<RT, RP...> &atensor)
                    : m_track(atensor.impl_track()), m_map() {
                using SrcTraits = typename OffsetTensor<RT, RP...>::traits;
                using Mapping = flare::detail::TensorMapping<traits, SrcTraits, void>;
                static_assert(Mapping::is_assignable,
                              "Incompatible OffsetTensor copy construction");
                Mapping::assign(m_map, atensor.impl_map(), m_track);

                for (size_t i = 0; i < Tensor<RT, RP...>::rank(); ++i) {
                    m_begins[i] = 0;
                }
            }

            template<class RT, class... RP>
            FLARE_FUNCTION OffsetTensor(const Tensor<RT, RP...> &atensor,
                                        const index_list_type &minIndices)
                    : m_track(atensor.impl_track()), m_map() {
                using SrcTraits = typename OffsetTensor<RT, RP...>::traits;
                using Mapping = flare::detail::TensorMapping<traits, SrcTraits, void>;
                static_assert(Mapping::is_assignable,
                              "Incompatible OffsetTensor copy construction");
                Mapping::assign(m_map, atensor.impl_map(), m_track);

                FLARE_IF_ON_HOST((flare::experimental::detail::runtime_check_rank_host(
                        traits::rank_dynamic, Rank, minIndices, label());))

                FLARE_IF_ON_DEVICE((flare::experimental::detail::runtime_check_rank_device(
                        traits::rank_dynamic, Rank, minIndices);))

                for (size_t i = 0; i < minIndices.size(); ++i) {
                    m_begins[i] = minIndices.begin()[i];
                }
            }

            template<class RT, class... RP>
            FLARE_FUNCTION OffsetTensor(const Tensor<RT, RP...> &atensor,
                                        const begins_type &beg)
                    : m_track(atensor.impl_track()), m_map(), m_begins(beg) {
                using SrcTraits = typename OffsetTensor<RT, RP...>::traits;
                using Mapping = flare::detail::TensorMapping<traits, SrcTraits, void>;
                static_assert(Mapping::is_assignable,
                              "Incompatible OffsetTensor copy construction");
                Mapping::assign(m_map, atensor.impl_map(), m_track);
            }

            // may assign unmanaged from managed.

            template<class RT, class... RP>
            FLARE_FUNCTION OffsetTensor(const OffsetTensor<RT, RP...> &rhs)
                    : m_track(rhs.m_track, traits::is_managed),
                      m_map(),
                      m_begins(rhs.m_begins) {
                using SrcTraits = typename OffsetTensor<RT, RP...>::traits;
                using Mapping = flare::detail::TensorMapping<traits, SrcTraits, void>;
                static_assert(Mapping::is_assignable,
                              "Incompatible OffsetTensor copy construction");
                Mapping::assign(m_map, rhs.m_map, rhs.m_track);  // swb what about assign?
            }

        private:
            enum class subtraction_failure {
                none,
                negative,
                overflow,
            };

            // Subtraction should return a non-negative number and not overflow
            FLARE_FUNCTION static subtraction_failure check_subtraction(int64_t lhs,
                                                                        int64_t rhs) {
                if (lhs < rhs) return subtraction_failure::negative;

                if (static_cast<uint64_t>(-1) / static_cast<uint64_t>(2) <
                    static_cast<uint64_t>(lhs) - static_cast<uint64_t>(rhs))
                    return subtraction_failure::overflow;

                return subtraction_failure::none;
            }

            // Need a way to get at an element from both begins_type (aka flare::Array
            // which doesn't have iterators) and index_list_type (aka
            // std::initializer_list which doesn't have .data() or operator[]).
            // Returns by value
            FLARE_FUNCTION
            static int64_t at(const begins_type &a, size_t pos) { return a[pos]; }

            FLARE_FUNCTION
            static int64_t at(index_list_type a, size_t pos) {
                return *(a.begin() + pos);
            }

            // Check that begins < ends for all elements
            // B, E can be begins_type and/or index_list_type
            template<typename B, typename E>
            static subtraction_failure runtime_check_begins_ends_host(const B &begins,
                                                                      const E &ends) {
                std::string message;
                if (begins.size() != Rank)
                    message +=
                            "begins.size() "
                            "(" +
                            std::to_string(begins.size()) +
                            ")"
                            " != Rank "
                            "(" +
                            std::to_string(Rank) +
                            ")"
                            "\n";

                if (ends.size() != Rank)
                    message +=
                            "ends.size() "
                            "(" +
                            std::to_string(begins.size()) +
                            ")"
                            " != Rank "
                            "(" +
                            std::to_string(Rank) +
                            ")"
                            "\n";

                // If there are no errors so far, then arg_rank == Rank
                // Otherwise, check as much as possible
                size_t arg_rank = begins.size() < ends.size() ? begins.size() : ends.size();
                for (size_t i = 0; i != arg_rank; ++i) {
                    subtraction_failure sf = check_subtraction(at(ends, i), at(begins, i));
                    if (sf != subtraction_failure::none) {
                        message +=
                                "("
                                "ends[" +
                                std::to_string(i) +
                                "]"
                                " "
                                "(" +
                                std::to_string(at(ends, i)) +
                                ")"
                                " - "
                                "begins[" +
                                std::to_string(i) +
                                "]"
                                " "
                                "(" +
                                std::to_string(at(begins, i)) +
                                ")"
                                ")";
                        switch (sf) {
                            case subtraction_failure::negative:
                                message += " must be non-negative\n";
                                break;
                            case subtraction_failure::overflow:
                                message += " overflows\n";
                                break;
                            default:
                                break;
                        }
                    }
                }

                if (!message.empty()) {
                    message =
                            "flare::experimental::OffsetTensor ERROR: for unmanaged OffsetTensor\n" +
                            message;
                    flare::detail::throw_runtime_exception(message);
                }

                return subtraction_failure::none;
            }

            // Check the begins < ends for all elements
            template<typename B, typename E>
            FLARE_FUNCTION static subtraction_failure runtime_check_begins_ends_device(
                    const B &begins, const E &ends) {
                if (begins.size() != Rank)
                    flare::abort(
                            "flare::experimental::OffsetTensor ERROR: for unmanaged "
                            "OffsetTensor: begins has bad Rank");
                if (ends.size() != Rank)
                    flare::abort(
                            "flare::experimental::OffsetTensor ERROR: for unmanaged "
                            "OffsetTensor: ends has bad Rank");

                for (size_t i = 0; i != begins.size(); ++i) {
                    switch (check_subtraction(at(ends, i), at(begins, i))) {
                        case subtraction_failure::negative:
                            flare::abort(
                                    "flare::experimental::OffsetTensor ERROR: for unmanaged "
                                    "OffsetTensor: bad range");
                            break;
                        case subtraction_failure::overflow:
                            flare::abort(
                                    "flare::experimental::OffsetTensor ERROR: for unmanaged "
                                    "OffsetTensor: range overflows");
                            break;
                        default:
                            break;
                    }
                }

                return subtraction_failure::none;
            }

            template<typename B, typename E>
            FLARE_FUNCTION static subtraction_failure runtime_check_begins_ends(
                    const B &begins, const E &ends) {
                FLARE_IF_ON_HOST((return runtime_check_begins_ends_host(begins, ends);))
                FLARE_IF_ON_DEVICE(
                (return runtime_check_begins_ends_device(begins, ends);))
            }

            // Constructor around unmanaged data after checking begins < ends for all
            // elements
            // Each of B, E can be begins_type and/or index_list_type
            // Precondition: begins.size() == ends.size() == m_begins.size() == Rank
            template<typename B, typename E>
            FLARE_FUNCTION OffsetTensor(const pointer_type &p, const B &begins_,
                                        const E &ends_,
                                        subtraction_failure)
                    : m_track()  // no tracking
                    ,
                      m_map(flare::detail::TensorCtorProp<pointer_type>(p),
                            typename traits::array_layout(
                                    Rank > 0 ? at(ends_, 0) - at(begins_, 0) : 0,
                                    Rank > 1 ? at(ends_, 1) - at(begins_, 1) : 0,
                                    Rank > 2 ? at(ends_, 2) - at(begins_, 2) : 0,
                                    Rank > 3 ? at(ends_, 3) - at(begins_, 3) : 0,
                                    Rank > 4 ? at(ends_, 4) - at(begins_, 4) : 0,
                                    Rank > 5 ? at(ends_, 5) - at(begins_, 5) : 0,
                                    Rank > 6 ? at(ends_, 6) - at(begins_, 6) : 0,
                                    Rank > 7 ? at(ends_, 7) - at(begins_, 7) : 0)) {
                for (size_t i = 0; i != m_begins.size(); ++i) {
                    m_begins[i] = at(begins_, i);
                };
            }

        public:
            // Constructor around unmanaged data
            // Four overloads, as both begins and ends can be either
            // begins_type or index_list_type
            FLARE_FUNCTION
            OffsetTensor(const pointer_type &p, const begins_type &begins_,
                         const begins_type &ends_)
                    : OffsetTensor(p, begins_, ends_,
                                   runtime_check_begins_ends(begins_, ends_)) {}

            FLARE_FUNCTION
            OffsetTensor(const pointer_type &p, const begins_type &begins_,
                         index_list_type ends_)
                    : OffsetTensor(p, begins_, ends_,
                                   runtime_check_begins_ends(begins_, ends_)) {}

            FLARE_FUNCTION
            OffsetTensor(const pointer_type &p, index_list_type begins_,
                         const begins_type &ends_)
                    : OffsetTensor(p, begins_, ends_,
                                   runtime_check_begins_ends(begins_, ends_)) {}

            FLARE_FUNCTION
            OffsetTensor(const pointer_type &p, index_list_type begins_,
                         index_list_type ends_)
                    : OffsetTensor(p, begins_, ends_,
                                   runtime_check_begins_ends(begins_, ends_)) {}

            //----------------------------------------
            // Allocation tracking properties
            FLARE_FUNCTION
            int use_count() const { return m_track.use_count(); }

            const std::string label() const {
                return m_track.template get_label<typename traits::memory_space>();
            }

            // Choosing std::pair as type for the arguments allows constructing an
            // OffsetTensor using list initialization syntax, e.g.,
            //   OffsetTensor dummy("dummy", {-1, 3}, {-2,2});
            // We could allow arbitrary types RangeType that support
            // std::get<{0,1}>(RangeType const&) with std::tuple_size<RangeType>::value==2
            // but this wouldn't allow using the syntax in the example above.
            template<typename Label>
            explicit OffsetTensor(
                    const Label &arg_label,
                    std::enable_if_t<flare::detail::is_tensor_label<Label>::value,
                            const std::pair<int64_t, int64_t>>
                    range0,
                    const std::pair<int64_t, int64_t> range1 = FLARE_INVALID_INDEX_RANGE,
                    const std::pair<int64_t, int64_t> range2 = FLARE_INVALID_INDEX_RANGE,
                    const std::pair<int64_t, int64_t> range3 = FLARE_INVALID_INDEX_RANGE,
                    const std::pair<int64_t, int64_t> range4 = FLARE_INVALID_INDEX_RANGE,
                    const std::pair<int64_t, int64_t> range5 = FLARE_INVALID_INDEX_RANGE,
                    const std::pair<int64_t, int64_t> range6 = FLARE_INVALID_INDEX_RANGE,
                    const std::pair<int64_t, int64_t> range7 = FLARE_INVALID_INDEX_RANGE

            )
                    : OffsetTensor(
                    flare::detail::TensorCtorProp<std::string>(arg_label),
                    typename traits::array_layout(range0.second - range0.first + 1,
                                                  range1.second - range1.first + 1,
                                                  range2.second - range2.first + 1,
                                                  range3.second - range3.first + 1,
                                                  range4.second - range4.first + 1,
                                                  range5.second - range5.first + 1,
                                                  range6.second - range6.first + 1,
                                                  range7.second - range7.first + 1),
                    {range0.first, range1.first, range2.first, range3.first,
                     range4.first, range5.first, range6.first, range7.first}) {}

            template<class... P>
            explicit OffsetTensor(
                    const flare::detail::TensorCtorProp<P...> &arg_prop,
                    const std::pair<int64_t, int64_t> range0 = FLARE_INVALID_INDEX_RANGE,
                    const std::pair<int64_t, int64_t> range1 = FLARE_INVALID_INDEX_RANGE,
                    const std::pair<int64_t, int64_t> range2 = FLARE_INVALID_INDEX_RANGE,
                    const std::pair<int64_t, int64_t> range3 = FLARE_INVALID_INDEX_RANGE,
                    const std::pair<int64_t, int64_t> range4 = FLARE_INVALID_INDEX_RANGE,
                    const std::pair<int64_t, int64_t> range5 = FLARE_INVALID_INDEX_RANGE,
                    const std::pair<int64_t, int64_t> range6 = FLARE_INVALID_INDEX_RANGE,
                    const std::pair<int64_t, int64_t> range7 = FLARE_INVALID_INDEX_RANGE)
                    : OffsetTensor(
                    arg_prop,
                    typename traits::array_layout(range0.second - range0.first + 1,
                                                  range1.second - range1.first + 1,
                                                  range2.second - range2.first + 1,
                                                  range3.second - range3.first + 1,
                                                  range4.second - range4.first + 1,
                                                  range5.second - range5.first + 1,
                                                  range6.second - range6.first + 1,
                                                  range7.second - range7.first + 1),
                    {range0.first, range1.first, range2.first, range3.first,
                     range4.first, range5.first, range6.first, range7.first}) {}

            template<class... P>
            explicit FLARE_FUNCTION OffsetTensor(
                    const flare::detail::TensorCtorProp<P...> &arg_prop,
                    std::enable_if_t<flare::detail::TensorCtorProp<P...>::has_pointer,
                            typename traits::array_layout> const &arg_layout,
                    const index_list_type minIndices)
                    : m_track()  // No memory tracking
                    ,
                      m_map(arg_prop, arg_layout) {
                for (size_t i = 0; i < minIndices.size(); ++i) {
                    m_begins[i] = minIndices.begin()[i];
                }
                static_assert(
                        std::is_same<pointer_type, typename flare::detail::TensorCtorProp<
                                P...>::pointer_type>::value,
                        "When constructing OffsetTensor to wrap user memory, you must supply "
                        "matching pointer type");
            }

            template<class... P>
            explicit OffsetTensor(
                    const flare::detail::TensorCtorProp<P...> &arg_prop,
                    std::enable_if_t<!flare::detail::TensorCtorProp<P...>::has_pointer,
                            typename traits::array_layout> const &arg_layout,
                    const index_list_type minIndices)
                    : m_track(),
                      m_map() {
                for (size_t i = 0; i < Rank; ++i) m_begins[i] = minIndices.begin()[i];

                // Copy the input allocation properties with possibly defaulted properties
                auto prop_copy = flare::detail::with_properties_if_unset(
                        arg_prop, std::string{}, typename traits::device_type::memory_space{},
                        typename traits::device_type::execution_space{});
                using alloc_prop = decltype(prop_copy);

                static_assert(traits::is_managed,
                              "OffsetTensor allocation constructor requires managed memory");

                if (alloc_prop::initialize &&
                    !alloc_prop::execution_space::impl_is_initialized()) {
                    // If initializing tensor data then
                    // the execution space must be initialized.
                    flare::detail::throw_runtime_exception(
                            "Constructing OffsetTensor and initializing data with uninitialized "
                            "execution space");
                }

                flare::detail::SharedAllocationRecord<> *record = m_map.allocate_shared(
                        prop_copy, arg_layout,
                        flare::detail::TensorCtorProp<P...>::has_execution_space);

                // Setup and initialization complete, start tracking
                m_track.assign_allocated_record_to_uninitialized(record);

                FLARE_IF_ON_HOST((flare::experimental::detail::runtime_check_rank_host(
                        traits::rank_dynamic, Rank, minIndices, label());))

                FLARE_IF_ON_DEVICE((flare::experimental::detail::runtime_check_rank_device(
                        traits::rank_dynamic, Rank, minIndices);))
            }
        };

/** \brief Temporary free function rank()
 *         until rank() is implemented
 *         in the Tensor
 */
        template<typename D, class... P>
        FLARE_INLINE_FUNCTION constexpr unsigned rank(const OffsetTensor<D, P...> &V) {
            return V.Rank;
        }  // Temporary until added to tensor

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
        namespace detail {

            template<class T>
            FLARE_INLINE_FUNCTION std::enable_if_t<std::is_integral<T>::value, T>
            shift_input(const T arg, const int64_t offset) {
                return arg - offset;
            }

            FLARE_INLINE_FUNCTION
            flare::ALL_t shift_input(const flare::ALL_t arg, const int64_t /*offset*/) {
                return arg;
            }

            template<class T>
            FLARE_INLINE_FUNCTION
            std::enable_if_t<std::is_integral<T>::value, flare::pair<T, T>>
            shift_input(const flare::pair<T, T> arg, const int64_t offset) {
                return flare::make_pair<T, T>(arg.first - offset, arg.second - offset);
            }

            template<class T>
            inline std::enable_if_t<std::is_integral<T>::value, std::pair<T, T>>
            shift_input(const std::pair<T, T> arg, const int64_t offset) {
                return std::make_pair<T, T>(arg.first - offset, arg.second - offset);
            }

            template<size_t N, class Arg, class A>
            FLARE_INLINE_FUNCTION void map_arg_to_new_begin(
                    const size_t i, flare::Array<int64_t, N> &subtensorBegins,
                    std::enable_if_t<N != 0, const Arg> shiftedArg, const Arg arg,
                    const A TensorBegins, size_t &counter) {
                if (!std::is_integral<Arg>::value) {
                    subtensorBegins[counter] = shiftedArg == arg ? TensorBegins[i] : 0;
                    counter++;
                }
            }

            template<size_t N, class Arg, class A>
            FLARE_INLINE_FUNCTION void map_arg_to_new_begin(
                    const size_t /*i*/, flare::Array<int64_t, N> & /*subtensorBegins*/,
                    std::enable_if_t<N == 0, const Arg> /*shiftedArg*/, const Arg /*arg*/,
                    const A /*TensorBegins*/, size_t & /*counter*/) {}

            template<class D, class... P, class T>
            FLARE_INLINE_FUNCTION
            typename flare::experimental::detail::GetOffsetTensorTypeFromTensorType<
                    typename flare::detail::TensorMapping<void /* deduce subtensor type from
                                                   source tensor traits */
                            ,
                            TensorTraits<D, P...>, T>::type>::type
            subtensor_offset(const OffsetTensor<D, P...> &src, T arg) {
                auto theTensor = src.tensor();
                auto begins = src.begins();

                T shiftedArg = shift_input(arg, begins[0]);

                constexpr size_t rank =
                        flare::detail::TensorMapping<void /* deduce subtensor type from source tensor
                                        traits */
                                ,
                                TensorTraits<D, P...>, T>::type::rank;

                auto theSubtensor = flare::subtensor(theTensor, shiftedArg);

                flare::Array<int64_t, rank> subtensorBegins;
                size_t counter = 0;
                flare::experimental::detail::map_arg_to_new_begin(0, subtensorBegins, shiftedArg,
                                                                  arg, begins, counter);

                typename flare::experimental::detail::GetOffsetTensorTypeFromTensorType<
                        typename flare::detail::TensorMapping<void /* deduce subtensor type from source
                                                 tensor traits */
                                ,
                                TensorTraits<D, P...>, T>::type>::type
                        offsetTensor(theSubtensor, subtensorBegins);

                return offsetTensor;
            }

            template<class D, class... P, class T0, class T1>
            FLARE_INLINE_FUNCTION
            typename flare::experimental::detail::GetOffsetTensorTypeFromTensorType<
                    typename flare::detail::TensorMapping<
                            void /* deduce subtensor type from source tensor traits */
                            ,
                            TensorTraits<D, P...>, T0, T1>::type>::type
            subtensor_offset(const flare::experimental::OffsetTensor<D, P...> &src,
                             T0 arg0, T1 arg1) {
                auto theTensor = src.tensor();
                auto begins = src.begins();

                T0 shiftedArg0 = shift_input(arg0, begins[0]);
                T1 shiftedArg1 = shift_input(arg1, begins[1]);

                auto theSubtensor = flare::subtensor(theTensor, shiftedArg0, shiftedArg1);
                constexpr size_t rank =
                        flare::detail::TensorMapping<void /* deduce subtensor type from source tensor
                                        traits */
                                ,
                                TensorTraits<D, P...>, T0, T1>::type::rank;

                flare::Array<int64_t, rank> subtensorBegins;
                size_t counter = 0;
                flare::experimental::detail::map_arg_to_new_begin(
                        0, subtensorBegins, shiftedArg0, arg0, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        1, subtensorBegins, shiftedArg1, arg1, begins, counter);

                typename flare::experimental::detail::GetOffsetTensorTypeFromTensorType<
                        typename flare::detail::TensorMapping<
                                void /* deduce subtensor type from source tensor traits */
                                ,
                                TensorTraits<D, P...>, T0, T1>::type>::type offsetTensor(theSubtensor,
                                                                                       subtensorBegins);

                return offsetTensor;
            }

            template<class D, class... P, class T0, class T1, class T2>
            FLARE_INLINE_FUNCTION
            typename flare::experimental::detail::GetOffsetTensorTypeFromTensorType<
                    typename flare::detail::TensorMapping<
                            void /* deduce subtensor type from source tensor traits */
                            ,
                            TensorTraits<D, P...>, T0, T1, T2>::type>::type
            subtensor_offset(const OffsetTensor<D, P...> &src, T0 arg0, T1 arg1, T2 arg2) {
                auto theTensor = src.tensor();
                auto begins = src.begins();

                T0 shiftedArg0 = shift_input(arg0, begins[0]);
                T1 shiftedArg1 = shift_input(arg1, begins[1]);
                T2 shiftedArg2 = shift_input(arg2, begins[2]);

                auto theSubtensor =
                        flare::subtensor(theTensor, shiftedArg0, shiftedArg1, shiftedArg2);

                constexpr size_t rank =
                        flare::detail::TensorMapping<void /* deduce subtensor type from source tensor
                                        traits */
                                ,
                                TensorTraits<D, P...>, T0, T1, T2>::type::rank;

                flare::Array<int64_t, rank> subtensorBegins;

                size_t counter = 0;
                flare::experimental::detail::map_arg_to_new_begin(
                        0, subtensorBegins, shiftedArg0, arg0, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        1, subtensorBegins, shiftedArg1, arg1, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        2, subtensorBegins, shiftedArg2, arg2, begins, counter);

                typename flare::experimental::detail::GetOffsetTensorTypeFromTensorType<
                        typename flare::detail::TensorMapping<
                                void /* deduce subtensor type from source tensor traits */
                                ,
                                TensorTraits<D, P...>, T0, T1, T2>::type>::type
                        offsetTensor(theSubtensor, subtensorBegins);

                return offsetTensor;
            }

            template<class D, class... P, class T0, class T1, class T2, class T3>
            FLARE_INLINE_FUNCTION
            typename flare::experimental::detail::GetOffsetTensorTypeFromTensorType<
                    typename flare::detail::TensorMapping<
                            void /* deduce subtensor type from source tensor traits */
                            ,
                            TensorTraits<D, P...>, T0, T1, T2, T3>::type>::type
            subtensor_offset(const OffsetTensor<D, P...> &src, T0 arg0, T1 arg1, T2 arg2,
                             T3 arg3) {
                auto theTensor = src.tensor();
                auto begins = src.begins();

                T0 shiftedArg0 = shift_input(arg0, begins[0]);
                T1 shiftedArg1 = shift_input(arg1, begins[1]);
                T2 shiftedArg2 = shift_input(arg2, begins[2]);
                T3 shiftedArg3 = shift_input(arg3, begins[3]);

                auto theSubtensor = flare::subtensor(theTensor, shiftedArg0, shiftedArg1,
                                                     shiftedArg2, shiftedArg3);

                constexpr size_t rank = flare::detail::TensorMapping<
                        void /* deduce subtensor type from source tensor traits */
                        ,
                        TensorTraits<D, P...>, T0, T1, T2, T3>::type::rank;
                flare::Array<int64_t, rank> subtensorBegins;

                size_t counter = 0;
                flare::experimental::detail::map_arg_to_new_begin(
                        0, subtensorBegins, shiftedArg0, arg0, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        1, subtensorBegins, shiftedArg1, arg1, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        2, subtensorBegins, shiftedArg2, arg2, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        3, subtensorBegins, shiftedArg3, arg3, begins, counter);

                typename flare::experimental::detail::GetOffsetTensorTypeFromTensorType<
                        typename flare::detail::TensorMapping<
                                void /* deduce subtensor type from source tensor traits */
                                ,
                                TensorTraits<D, P...>, T0, T1, T2, T3>::type>::type
                        offsetTensor(theSubtensor, subtensorBegins);

                return offsetTensor;
            }

            template<class D, class... P, class T0, class T1, class T2, class T3, class T4>
            FLARE_INLINE_FUNCTION
            typename flare::experimental::detail::GetOffsetTensorTypeFromTensorType<
                    typename flare::detail::TensorMapping<
                            void /* deduce subtensor type from source tensor traits */
                            ,
                            TensorTraits<D, P...>, T0, T1, T2, T3, T4>::type>::type
            subtensor_offset(const OffsetTensor<D, P...> &src, T0 arg0, T1 arg1, T2 arg2,
                             T3 arg3, T4 arg4) {
                auto theTensor = src.tensor();
                auto begins = src.begins();

                T0 shiftedArg0 = shift_input(arg0, begins[0]);
                T1 shiftedArg1 = shift_input(arg1, begins[1]);
                T2 shiftedArg2 = shift_input(arg2, begins[2]);
                T3 shiftedArg3 = shift_input(arg3, begins[3]);
                T4 shiftedArg4 = shift_input(arg4, begins[4]);

                auto theSubtensor = flare::subtensor(theTensor, shiftedArg0, shiftedArg1,
                                                     shiftedArg2, shiftedArg3, shiftedArg4);

                constexpr size_t rank = flare::detail::TensorMapping<
                        void /* deduce subtensor type from source tensor traits */
                        ,
                        TensorTraits<D, P...>, T0, T1, T2, T3, T4>::type::rank;
                flare::Array<int64_t, rank> subtensorBegins;

                size_t counter = 0;
                flare::experimental::detail::map_arg_to_new_begin(
                        0, subtensorBegins, shiftedArg0, arg0, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        1, subtensorBegins, shiftedArg1, arg1, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        2, subtensorBegins, shiftedArg2, arg2, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        3, subtensorBegins, shiftedArg3, arg3, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        4, subtensorBegins, shiftedArg4, arg4, begins, counter);

                typename flare::experimental::detail::GetOffsetTensorTypeFromTensorType<
                        typename flare::detail::TensorMapping<
                                void /* deduce subtensor type from source tensor traits */
                                ,
                                TensorTraits<D, P...>, T0, T1, T2, T3, T4>::type>::type
                        offsetTensor(theSubtensor, subtensorBegins);

                return offsetTensor;
            }

            template<class D, class... P, class T0, class T1, class T2, class T3, class T4,
                    class T5>
            FLARE_INLINE_FUNCTION
            typename flare::experimental::detail::GetOffsetTensorTypeFromTensorType<
                    typename flare::detail::TensorMapping<
                            void /* deduce subtensor type from source tensor traits */
                            ,
                            TensorTraits<D, P...>, T0, T1, T2, T3, T4, T5>::type>::type
            subtensor_offset(const OffsetTensor<D, P...> &src, T0 arg0, T1 arg1, T2 arg2,
                             T3 arg3, T4 arg4, T5 arg5) {
                auto theTensor = src.tensor();
                auto begins = src.begins();

                T0 shiftedArg0 = shift_input(arg0, begins[0]);
                T1 shiftedArg1 = shift_input(arg1, begins[1]);
                T2 shiftedArg2 = shift_input(arg2, begins[2]);
                T3 shiftedArg3 = shift_input(arg3, begins[3]);
                T4 shiftedArg4 = shift_input(arg4, begins[4]);
                T5 shiftedArg5 = shift_input(arg5, begins[5]);

                auto theSubtensor =
                        flare::subtensor(theTensor, shiftedArg0, shiftedArg1, shiftedArg2,
                                         shiftedArg3, shiftedArg4, shiftedArg5);

                constexpr size_t rank = flare::detail::TensorMapping<
                        void /* deduce subtensor type from source tensor traits */
                        ,
                        TensorTraits<D, P...>, T0, T1, T2, T3, T4, T5>::type::rank;

                flare::Array<int64_t, rank> subtensorBegins;

                size_t counter = 0;
                flare::experimental::detail::map_arg_to_new_begin(
                        0, subtensorBegins, shiftedArg0, arg0, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        1, subtensorBegins, shiftedArg1, arg1, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        2, subtensorBegins, shiftedArg2, arg2, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        3, subtensorBegins, shiftedArg3, arg3, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        4, subtensorBegins, shiftedArg4, arg4, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        5, subtensorBegins, shiftedArg5, arg5, begins, counter);

                typename flare::experimental::detail::GetOffsetTensorTypeFromTensorType<
                        typename flare::detail::TensorMapping<
                                void /* deduce subtensor type from source tensor traits */
                                ,
                                TensorTraits<D, P...>, T0, T1, T2, T3, T4, T5>::type>::type
                        offsetTensor(theSubtensor, subtensorBegins);

                return offsetTensor;
            }

            template<class D, class... P, class T0, class T1, class T2, class T3, class T4,
                    class T5, class T6>
            FLARE_INLINE_FUNCTION
            typename flare::experimental::detail::GetOffsetTensorTypeFromTensorType<
                    typename flare::detail::TensorMapping<
                            void /* deduce subtensor type from source tensor traits */
                            ,
                            TensorTraits<D, P...>, T0, T1, T2, T3, T4, T5, T6>::type>::type
            subtensor_offset(const OffsetTensor<D, P...> &src, T0 arg0, T1 arg1, T2 arg2,
                             T3 arg3, T4 arg4, T5 arg5, T6 arg6) {
                auto theTensor = src.tensor();
                auto begins = src.begins();

                T0 shiftedArg0 = shift_input(arg0, begins[0]);
                T1 shiftedArg1 = shift_input(arg1, begins[1]);
                T2 shiftedArg2 = shift_input(arg2, begins[2]);
                T3 shiftedArg3 = shift_input(arg3, begins[3]);
                T4 shiftedArg4 = shift_input(arg4, begins[4]);
                T5 shiftedArg5 = shift_input(arg5, begins[5]);
                T6 shiftedArg6 = shift_input(arg6, begins[6]);

                auto theSubtensor =
                        flare::subtensor(theTensor, shiftedArg0, shiftedArg1, shiftedArg2,
                                         shiftedArg3, shiftedArg4, shiftedArg5, shiftedArg6);

                constexpr size_t rank = flare::detail::TensorMapping<
                        void /* deduce subtensor type from source tensor traits */
                        ,
                        TensorTraits<D, P...>, T0, T1, T2, T3, T4, T5, T6>::type::rank;

                flare::Array<int64_t, rank> subtensorBegins;

                size_t counter = 0;
                flare::experimental::detail::map_arg_to_new_begin(
                        0, subtensorBegins, shiftedArg0, arg0, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        1, subtensorBegins, shiftedArg1, arg1, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        2, subtensorBegins, shiftedArg2, arg2, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        3, subtensorBegins, shiftedArg3, arg3, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        4, subtensorBegins, shiftedArg4, arg4, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        5, subtensorBegins, shiftedArg5, arg5, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        6, subtensorBegins, shiftedArg6, arg6, begins, counter);

                typename flare::experimental::detail::GetOffsetTensorTypeFromTensorType<
                        typename flare::detail::TensorMapping<
                                void /* deduce subtensor type from source tensor traits */
                                ,
                                TensorTraits<D, P...>, T0, T1, T2, T3, T4, T5, T6>::type>::type
                        offsetTensor(theSubtensor, subtensorBegins);

                return offsetTensor;
            }

            template<class D, class... P, class T0, class T1, class T2, class T3, class T4,
                    class T5, class T6, class T7>
            FLARE_INLINE_FUNCTION
            typename flare::experimental::detail::GetOffsetTensorTypeFromTensorType<
                    typename flare::detail::TensorMapping<
                            void /* deduce subtensor type from source tensor traits */
                            ,
                            TensorTraits<D, P...>, T0, T1, T2, T3, T4, T5, T6, T7>::type>::type
            subtensor_offset(const OffsetTensor<D, P...> &src, T0 arg0, T1 arg1, T2 arg2,
                             T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) {
                auto theTensor = src.tensor();
                auto begins = src.begins();

                T0 shiftedArg0 = shift_input(arg0, begins[0]);
                T1 shiftedArg1 = shift_input(arg1, begins[1]);
                T2 shiftedArg2 = shift_input(arg2, begins[2]);
                T3 shiftedArg3 = shift_input(arg3, begins[3]);
                T4 shiftedArg4 = shift_input(arg4, begins[4]);
                T5 shiftedArg5 = shift_input(arg5, begins[5]);
                T6 shiftedArg6 = shift_input(arg6, begins[6]);
                T7 shiftedArg7 = shift_input(arg7, begins[7]);

                auto theSubtensor = flare::subtensor(theTensor, shiftedArg0, shiftedArg1,
                                                     shiftedArg2, shiftedArg3, shiftedArg4,
                                                     shiftedArg5, shiftedArg6, shiftedArg7);

                constexpr size_t rank = flare::detail::TensorMapping<
                        void /* deduce subtensor type from source tensor traits */
                        ,
                        TensorTraits<D, P...>, T0, T1, T2, T3, T4, T5, T6, T7>::type::rank;

                flare::Array<int64_t, rank> subtensorBegins;

                size_t counter = 0;
                flare::experimental::detail::map_arg_to_new_begin(
                        0, subtensorBegins, shiftedArg0, arg0, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        1, subtensorBegins, shiftedArg1, arg1, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        2, subtensorBegins, shiftedArg2, arg2, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        3, subtensorBegins, shiftedArg3, arg3, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        4, subtensorBegins, shiftedArg4, arg4, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        5, subtensorBegins, shiftedArg5, arg5, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        6, subtensorBegins, shiftedArg6, arg6, begins, counter);
                flare::experimental::detail::map_arg_to_new_begin(
                        7, subtensorBegins, shiftedArg7, arg7, begins, counter);

                typename flare::experimental::detail::GetOffsetTensorTypeFromTensorType<
                        typename flare::detail::TensorMapping<
                                void /* deduce subtensor type from source tensor traits */
                                ,
                                TensorTraits<D, P...>, T0, T1, T2, T3, T4, T5, T6, T7>::type>::type
                        offsetTensor(theSubtensor, subtensorBegins);

                return offsetTensor;
            }
        }  // namespace detail

        template<class D, class... P, class... Args>
        FLARE_INLINE_FUNCTION
        typename flare::experimental::detail::GetOffsetTensorTypeFromTensorType<
                typename flare::detail::TensorMapping<
                        void /* deduce subtensor type from source tensor traits */
                        ,
                        TensorTraits<D, P...>, Args...>::type>::type
        subtensor(const OffsetTensor<D, P...> &src, Args... args) {
            static_assert(
                    OffsetTensor<D, P...>::Rank == sizeof...(Args),
                    "subtensor requires one argument for each source OffsetTensor rank");

            return flare::experimental::detail::subtensor_offset(src, args...);
        }

    }  // namespace experimental
}  // namespace flare
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {
    namespace experimental {
        template<class LT, class... LP, class RT, class... RP>
        FLARE_INLINE_FUNCTION bool operator==(const OffsetTensor<LT, LP...> &lhs,
                                              const OffsetTensor<RT, RP...> &rhs) {
            // Same data, layout, dimensions
            using lhs_traits = TensorTraits<LT, LP...>;
            using rhs_traits = TensorTraits<RT, RP...>;

            return std::is_same<typename lhs_traits::const_value_type,
                    typename rhs_traits::const_value_type>::value &&
                   std::is_same<typename lhs_traits::array_layout,
                           typename rhs_traits::array_layout>::value &&
                   std::is_same<typename lhs_traits::memory_space,
                           typename rhs_traits::memory_space>::value &&
                   unsigned(lhs_traits::rank) == unsigned(rhs_traits::rank) &&
                   lhs.data() == rhs.data() && lhs.span() == rhs.span() &&
                   lhs.extent(0) == rhs.extent(0) && lhs.extent(1) == rhs.extent(1) &&
                   lhs.extent(2) == rhs.extent(2) && lhs.extent(3) == rhs.extent(3) &&
                   lhs.extent(4) == rhs.extent(4) && lhs.extent(5) == rhs.extent(5) &&
                   lhs.extent(6) == rhs.extent(6) && lhs.extent(7) == rhs.extent(7) &&
                   lhs.begin(0) == rhs.begin(0) && lhs.begin(1) == rhs.begin(1) &&
                   lhs.begin(2) == rhs.begin(2) && lhs.begin(3) == rhs.begin(3) &&
                   lhs.begin(4) == rhs.begin(4) && lhs.begin(5) == rhs.begin(5) &&
                   lhs.begin(6) == rhs.begin(6) && lhs.begin(7) == rhs.begin(7);
        }

        template<class LT, class... LP, class RT, class... RP>
        FLARE_INLINE_FUNCTION bool operator!=(const OffsetTensor<LT, LP...> &lhs,
                                              const OffsetTensor<RT, RP...> &rhs) {
            return !(operator==(lhs, rhs));
        }

        template<class LT, class... LP, class RT, class... RP>
        FLARE_INLINE_FUNCTION bool operator==(const Tensor<LT, LP...> &lhs,
                                              const OffsetTensor<RT, RP...> &rhs) {
            // Same data, layout, dimensions
            using lhs_traits = TensorTraits<LT, LP...>;
            using rhs_traits = TensorTraits<RT, RP...>;

            return std::is_same<typename lhs_traits::const_value_type,
                    typename rhs_traits::const_value_type>::value &&
                   std::is_same<typename lhs_traits::array_layout,
                           typename rhs_traits::array_layout>::value &&
                   std::is_same<typename lhs_traits::memory_space,
                           typename rhs_traits::memory_space>::value &&
                   unsigned(lhs_traits::rank) == unsigned(rhs_traits::rank) &&
                   lhs.data() == rhs.data() && lhs.span() == rhs.span() &&
                   lhs.extent(0) == rhs.extent(0) && lhs.extent(1) == rhs.extent(1) &&
                   lhs.extent(2) == rhs.extent(2) && lhs.extent(3) == rhs.extent(3) &&
                   lhs.extent(4) == rhs.extent(4) && lhs.extent(5) == rhs.extent(5) &&
                   lhs.extent(6) == rhs.extent(6) && lhs.extent(7) == rhs.extent(7);
        }

        template<class LT, class... LP, class RT, class... RP>
        FLARE_INLINE_FUNCTION bool operator==(const OffsetTensor<LT, LP...> &lhs,
                                              const Tensor<RT, RP...> &rhs) {
            return rhs == lhs;
        }

    }  // namespace experimental
} /* namespace flare */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {

    template<class DT, class... DP>
    inline void deep_copy(
            const experimental::OffsetTensor<DT, DP...> &dst,
            typename TensorTraits<DT, DP...>::const_value_type &value,
            std::enable_if_t<std::is_same<typename TensorTraits<DT, DP...>::specialize,
                    void>::value> * = nullptr) {
        static_assert(
                std::is_same<typename TensorTraits<DT, DP...>::non_const_value_type,
                        typename TensorTraits<DT, DP...>::value_type>::value,
                "deep_copy requires non-const type");

        auto dstTensor = dst.tensor();
        flare::deep_copy(dstTensor, value);
    }

    template<class DT, class... DP, class ST, class... SP>
    inline void deep_copy(
            const experimental::OffsetTensor<DT, DP...> &dst,
            const experimental::OffsetTensor<ST, SP...> &value,
            std::enable_if_t<std::is_same<typename TensorTraits<DT, DP...>::specialize,
                    void>::value> * = nullptr) {
        static_assert(
                std::is_same<typename TensorTraits<DT, DP...>::value_type,
                        typename TensorTraits<ST, SP...>::non_const_value_type>::value,
                "deep_copy requires matching non-const destination type");

        auto dstTensor = dst.tensor();
        flare::deep_copy(dstTensor, value.tensor());
    }

    template<class DT, class... DP, class ST, class... SP>
    inline void deep_copy(
            const experimental::OffsetTensor<DT, DP...> &dst,
            const Tensor<ST, SP...> &value,
            std::enable_if_t<std::is_same<typename TensorTraits<DT, DP...>::specialize,
                    void>::value> * = nullptr) {
        static_assert(
                std::is_same<typename TensorTraits<DT, DP...>::value_type,
                        typename TensorTraits<ST, SP...>::non_const_value_type>::value,
                "deep_copy requires matching non-const destination type");

        auto dstTensor = dst.tensor();
        flare::deep_copy(dstTensor, value);
    }

    template<class DT, class... DP, class ST, class... SP>
    inline void deep_copy(
            const Tensor<DT, DP...> &dst,
            const experimental::OffsetTensor<ST, SP...> &value,
            std::enable_if_t<std::is_same<typename TensorTraits<DT, DP...>::specialize,
                    void>::value> * = nullptr) {
        static_assert(
                std::is_same<typename TensorTraits<DT, DP...>::value_type,
                        typename TensorTraits<ST, SP...>::non_const_value_type>::value,
                "deep_copy requires matching non-const destination type");

        flare::deep_copy(dst, value.tensor());
    }

    namespace detail {

// Deduce Mirror Types
        template<class Space, class T, class... P>
        struct MirrorOffsetTensorType {
            // The incoming tensor_type
            using src_tensor_type = typename flare::experimental::OffsetTensor<T, P...>;
            // The memory space for the mirror tensor
            using memory_space = typename Space::memory_space;
            // Check whether it is the same memory space
            enum {
                is_same_memspace =
                std::is_same<memory_space, typename src_tensor_type::memory_space>::value
            };
            // The array_layout
            using array_layout = typename src_tensor_type::array_layout;
            // The data type (we probably want it non-const since otherwise we can't even
            // deep_copy to it.)
            using data_type = typename src_tensor_type::non_const_data_type;
            // The destination tensor type if it is not the same memory space
            using dest_tensor_type =
                    flare::experimental::OffsetTensor<data_type, array_layout, Space>;
            // If it is the same memory_space return the existing tensor_type
            // This will also keep the unmanaged trait if necessary
            using tensor_type =
                    std::conditional_t<is_same_memspace, src_tensor_type, dest_tensor_type>;
        };

        template<class Space, class T, class... P>
        struct MirrorOffsetType {
            // The incoming tensor_type
            using src_tensor_type = typename flare::experimental::OffsetTensor<T, P...>;
            // The memory space for the mirror tensor
            using memory_space = typename Space::memory_space;
            // Check whether it is the same memory space
            enum {
                is_same_memspace =
                std::is_same<memory_space, typename src_tensor_type::memory_space>::value
            };
            // The array_layout
            using array_layout = typename src_tensor_type::array_layout;
            // The data type (we probably want it non-const since otherwise we can't even
            // deep_copy to it.)
            using data_type = typename src_tensor_type::non_const_data_type;
            // The destination tensor type if it is not the same memory space
            using tensor_type =
                    flare::experimental::OffsetTensor<data_type, array_layout, Space>;
        };

    }  // namespace detail

    namespace detail {
        template<class T, class... P, class... TensorCtorArgs>
        inline std::enable_if_t<
                !detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space,
                typename flare::experimental::OffsetTensor<T, P...>::HostMirror>
        create_mirror(const flare::experimental::OffsetTensor<T, P...> &src,
                      const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop) {
            return typename flare::experimental::OffsetTensor<T, P...>::HostMirror(
                    flare::create_mirror(arg_prop, src.tensor()), src.begins());
        }

        template<class T, class... P, class... TensorCtorArgs,
                class = std::enable_if_t<
                        detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space>>
        inline auto create_mirror(const flare::experimental::OffsetTensor<T, P...> &src,
                                  const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop) {
            using alloc_prop_input = detail::TensorCtorProp<TensorCtorArgs...>;
            using Space = typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space;

            static_assert(
                    !alloc_prop_input::has_label,
                    "The tensor constructor arguments passed to flare::create_mirror "
                    "must not include a label!");
            static_assert(
                    !alloc_prop_input::has_pointer,
                    "The tensor constructor arguments passed to flare::create_mirror must "
                    "not include a pointer!");
            static_assert(
                    !alloc_prop_input::allow_padding,
                    "The tensor constructor arguments passed to flare::create_mirror must "
                    "not explicitly allow padding!");

            auto prop_copy = detail::with_properties_if_unset(
                    arg_prop, std::string(src.label()).append("_mirror"));

            return typename flare::detail::MirrorOffsetType<Space, T, P...>::tensor_type(
                    prop_copy, src.layout(),
                    {src.begin(0), src.begin(1), src.begin(2), src.begin(3), src.begin(4),
                     src.begin(5), src.begin(6), src.begin(7)});
        }
    }  // namespace detail

// Create a mirror in host space
    template<class T, class... P>
    inline auto create_mirror(
            const flare::experimental::OffsetTensor<T, P...> &src) {
        return detail::create_mirror(src, detail::TensorCtorProp<>{});
    }

    template<class T, class... P>
    inline auto create_mirror(
            flare::detail::WithoutInitializing_t wi,
            const flare::experimental::OffsetTensor<T, P...> &src) {
        return detail::create_mirror(src, flare::tensor_alloc(wi));
    }

// Create a mirror in a new space
    template<class Space, class T, class... P,
            typename Enable = std::enable_if_t<flare::is_space<Space>::value>>
    inline auto create_mirror(
            const Space &, const flare::experimental::OffsetTensor<T, P...> &src) {
        return detail::create_mirror(
                src, flare::tensor_alloc(typename Space::memory_space{}));
    }

    template<class Space, class T, class... P>
    typename flare::detail::MirrorOffsetType<Space, T, P...>::tensor_type
    create_mirror(flare::detail::WithoutInitializing_t wi, const Space &,
                  const flare::experimental::OffsetTensor<T, P...> &src) {
        return detail::create_mirror(
                src, flare::tensor_alloc(typename Space::memory_space{}, wi));
    }

    template<class T, class... P, class... TensorCtorArgs>
    inline auto create_mirror(
            const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop,
            const flare::experimental::OffsetTensor<T, P...> &src) {
        return detail::create_mirror(src, arg_prop);
    }

    namespace detail {
        template<class T, class... P, class... TensorCtorArgs>
        inline std::enable_if_t<
                !detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space &&
                (std::is_same<
                        typename flare::experimental::OffsetTensor<T, P...>::memory_space,
                        typename flare::experimental::OffsetTensor<
                                T, P...>::HostMirror::memory_space>::value &&
                 std::is_same<
                         typename flare::experimental::OffsetTensor<T, P...>::data_type,
                         typename flare::experimental::OffsetTensor<
                                 T, P...>::HostMirror::data_type>::value),
                typename flare::experimental::OffsetTensor<T, P...>::HostMirror>
        create_mirror_tensor(const flare::experimental::OffsetTensor<T, P...> &src,
                             const detail::TensorCtorProp<TensorCtorArgs...> &) {
            return src;
        }

        template<class T, class... P, class... TensorCtorArgs>
        inline std::enable_if_t<
                !detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space &&
                !(std::is_same<
                        typename flare::experimental::OffsetTensor<T, P...>::memory_space,
                        typename flare::experimental::OffsetTensor<
                                T, P...>::HostMirror::memory_space>::value &&
                  std::is_same<
                          typename flare::experimental::OffsetTensor<T, P...>::data_type,
                          typename flare::experimental::OffsetTensor<
                                  T, P...>::HostMirror::data_type>::value),
                typename flare::experimental::OffsetTensor<T, P...>::HostMirror>
        create_mirror_tensor(const flare::experimental::OffsetTensor<T, P...> &src,
                             const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop) {
            return flare::create_mirror(arg_prop, src);
        }

        template<class T, class... P, class... TensorCtorArgs,
                class = std::enable_if_t<
                        detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space>>
        std::enable_if_t<detail::MirrorOffsetTensorType<
                typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space,
                T, P...>::is_same_memspace,
                typename detail::MirrorOffsetTensorType<
                        typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space,
                        T, P...>::tensor_type>
        create_mirror_tensor(const flare::experimental::OffsetTensor<T, P...> &src,
                             const detail::TensorCtorProp<TensorCtorArgs...> &) {
            return src;
        }

        template<class T, class... P, class... TensorCtorArgs,
                class = std::enable_if_t<
                        detail::TensorCtorProp<TensorCtorArgs...>::has_memory_space>>
        std::enable_if_t<!detail::MirrorOffsetTensorType<
                typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space,
                T, P...>::is_same_memspace,
                typename detail::MirrorOffsetTensorType<
                        typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space,
                        T, P...>::tensor_type>
        create_mirror_tensor(const flare::experimental::OffsetTensor<T, P...> &src,
                             const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop) {
            return flare::detail::create_mirror(src, arg_prop);
        }
    }  // namespace detail

// Create a mirror tensor in host space
    template<class T, class... P>
    inline auto create_mirror_tensor(
            const typename flare::experimental::OffsetTensor<T, P...> &src) {
        return detail::create_mirror_tensor(src, detail::TensorCtorProp<>{});
    }

    template<class T, class... P>
    inline auto create_mirror_tensor(
            flare::detail::WithoutInitializing_t wi,
            const typename flare::experimental::OffsetTensor<T, P...> &src) {
        return detail::create_mirror_tensor(src, flare::tensor_alloc(wi));
    }

// Create a mirror tensor in a new space
    template<class Space, class T, class... P,
            typename Enable = std::enable_if_t<flare::is_space<Space>::value>>
    inline auto create_mirror_tensor(
            const Space &, const flare::experimental::OffsetTensor<T, P...> &src) {
        return detail::create_mirror_tensor(
                src, flare::tensor_alloc(typename Space::memory_space{}));
    }

    template<class Space, class T, class... P>
    inline auto create_mirror_tensor(
            flare::detail::WithoutInitializing_t wi, const Space &,
            const flare::experimental::OffsetTensor<T, P...> &src) {
        return detail::create_mirror_tensor(
                src, flare::tensor_alloc(typename Space::memory_space{}, wi));
    }

    template<class T, class... P, class... TensorCtorArgs>
    inline auto create_mirror_tensor(
            const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop,
            const flare::experimental::OffsetTensor<T, P...> &src) {
        return detail::create_mirror_tensor(src, arg_prop);
    }

// Create a mirror tensor and deep_copy in a new space
    template<class... TensorCtorArgs, class T, class... P>
    typename flare::detail::MirrorOffsetTensorType<
            typename detail::TensorCtorProp<TensorCtorArgs...>::memory_space, T,
            P...>::tensor_type
    create_mirror_tensor_and_copy(
            const detail::TensorCtorProp<TensorCtorArgs...> &arg_prop,
            const flare::experimental::OffsetTensor<T, P...> &src) {
        return {create_mirror_tensor_and_copy(arg_prop, src.tensor()), src.begins()};
    }

    template<class Space, class T, class... P>
    typename flare::detail::MirrorOffsetTensorType<Space, T, P...>::tensor_type
    create_mirror_tensor_and_copy(
            const Space &space, const flare::experimental::OffsetTensor<T, P...> &src,
            std::string const &name = "") {
        return {create_mirror_tensor_and_copy(space, src.tensor(), name), src.begins()};
    }
}  // namespace flare

#endif  // FLARE_OFFSET_TENSOR_H_
