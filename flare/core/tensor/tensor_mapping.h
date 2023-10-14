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

#ifndef FLARE_CORE_TENSOR_TENSOR_MAPPING_H_
#define FLARE_CORE_TENSOR_TENSOR_MAPPING_H_

#include <type_traits>
#include <initializer_list>

#include <flare/core_fwd.h>
#include <flare/core/common/detection_idiom.h>
#include <flare/core/pair.h>
#include <flare/core/memory/layout.h>
#include <flare/core/common/extents.h>
#include <flare/core/common/error.h>
#include <flare/core/common/traits.h>
#include <flare/core/tensor/tensor_tracker.h>
#include <flare/core/tensor/tensor_ctor.h>
#include <flare/core/common/atomic_tensor.h>
#include <flare/core/profile/tools.h>
#include <flare/core/common/string_manipulation.h>
#include <flare/core/common/zero_memset_fwd.h>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare::detail {

    template<unsigned I, size_t... Args>
    struct variadic_size_t {
        enum : size_t {
            value = FLARE_INVALID_INDEX
        };
    };

    template<size_t Val, size_t... Args>
    struct variadic_size_t<0, Val, Args...> {
        enum : size_t {
            value = Val
        };
    };

    template<unsigned I, size_t Val, size_t... Args>
    struct variadic_size_t<I, Val, Args...> {
        enum : size_t {
            value = variadic_size_t<I - 1, Args...>::value
        };
    };

    template<size_t... Args>
    struct rank_dynamic;

    template<>
    struct rank_dynamic<> {
        enum : unsigned {
            value = 0
        };
    };

    template<size_t Val, size_t... Args>
    struct rank_dynamic<Val, Args...> {
        enum : unsigned {
            value = (Val == 0 ? 1 : 0) + rank_dynamic<Args...>::value
        };
    };

#define FLARE_IMPL_TENSOR_DIMENSION(R)                                       \
  template <size_t V, unsigned>                                             \
  struct TensorDimension##R {                                                 \
    static constexpr size_t ArgN##R = (V != FLARE_INVALID_INDEX ? V : 1);  \
    static constexpr size_t N##R    = (V != FLARE_INVALID_INDEX ? V : 1);  \
    FLARE_INLINE_FUNCTION explicit TensorDimension##R(size_t) {}             \
    TensorDimension##R()                        = default;                    \
    TensorDimension##R(const TensorDimension##R&) = default;                    \
    TensorDimension##R& operator=(const TensorDimension##R&) = default;         \
  };                                                                        \
  template <size_t V, unsigned RD>                                          \
  constexpr size_t TensorDimension##R<V, RD>::ArgN##R;                        \
  template <size_t V, unsigned RD>                                          \
  constexpr size_t TensorDimension##R<V, RD>::N##R;                           \
  template <unsigned RD>                                                    \
  struct TensorDimension##R<0u, RD> {                                         \
    static constexpr size_t ArgN##R = 0;                                    \
    std::conditional_t<(RD < 3), size_t, unsigned> N##R;                    \
    TensorDimension##R()                        = default;                    \
    TensorDimension##R(const TensorDimension##R&) = default;                    \
    TensorDimension##R& operator=(const TensorDimension##R&) = default;         \
    FLARE_INLINE_FUNCTION explicit TensorDimension##R(size_t V) : N##R(V) {} \
  };                                                                        \
  template <unsigned RD>                                                    \
  constexpr size_t TensorDimension##R<0u, RD>::ArgN##R;

    FLARE_IMPL_TENSOR_DIMENSION(0)

    FLARE_IMPL_TENSOR_DIMENSION(1)

    FLARE_IMPL_TENSOR_DIMENSION(2)

    FLARE_IMPL_TENSOR_DIMENSION(3)

    FLARE_IMPL_TENSOR_DIMENSION(4)

    FLARE_IMPL_TENSOR_DIMENSION(5)

    FLARE_IMPL_TENSOR_DIMENSION(6)

    FLARE_IMPL_TENSOR_DIMENSION(7)

#undef FLARE_IMPL_TENSOR_DIMENSION

// MSVC does not do empty base class optimization by default.
// Per standard it is required for standard layout types
    template<size_t... Vals>
    struct FLARE_IMPL_ENFORCE_EMPTY_BASE_OPTIMIZATION TensorDimension
            : public TensorDimension0<variadic_size_t<0u, Vals...>::value,
                    rank_dynamic<Vals...>::value>,
              public TensorDimension1<variadic_size_t<1u, Vals...>::value,
                      rank_dynamic<Vals...>::value>,
              public TensorDimension2<variadic_size_t<2u, Vals...>::value,
                      rank_dynamic<Vals...>::value>,
              public TensorDimension3<variadic_size_t<3u, Vals...>::value,
                      rank_dynamic<Vals...>::value>,
              public TensorDimension4<variadic_size_t<4u, Vals...>::value,
                      rank_dynamic<Vals...>::value>,
              public TensorDimension5<variadic_size_t<5u, Vals...>::value,
                      rank_dynamic<Vals...>::value>,
              public TensorDimension6<variadic_size_t<6u, Vals...>::value,
                      rank_dynamic<Vals...>::value>,
              public TensorDimension7<variadic_size_t<7u, Vals...>::value,
                      rank_dynamic<Vals...>::value> {
        using D0 = TensorDimension0<variadic_size_t<0U, Vals...>::value,
                rank_dynamic<Vals...>::value>;
        using D1 = TensorDimension1<variadic_size_t<1U, Vals...>::value,
                rank_dynamic<Vals...>::value>;
        using D2 = TensorDimension2<variadic_size_t<2U, Vals...>::value,
                rank_dynamic<Vals...>::value>;
        using D3 = TensorDimension3<variadic_size_t<3U, Vals...>::value,
                rank_dynamic<Vals...>::value>;
        using D4 = TensorDimension4<variadic_size_t<4U, Vals...>::value,
                rank_dynamic<Vals...>::value>;
        using D5 = TensorDimension5<variadic_size_t<5U, Vals...>::value,
                rank_dynamic<Vals...>::value>;
        using D6 = TensorDimension6<variadic_size_t<6U, Vals...>::value,
                rank_dynamic<Vals...>::value>;
        using D7 = TensorDimension7<variadic_size_t<7U, Vals...>::value,
                rank_dynamic<Vals...>::value>;

        using D0::ArgN0;
        using D1::ArgN1;
        using D2::ArgN2;
        using D3::ArgN3;
        using D4::ArgN4;
        using D5::ArgN5;
        using D6::ArgN6;
        using D7::ArgN7;

        using D0::N0;
        using D1::N1;
        using D2::N2;
        using D3::N3;
        using D4::N4;
        using D5::N5;
        using D6::N6;
        using D7::N7;

        static constexpr unsigned rank = sizeof...(Vals);
        static constexpr unsigned rank_dynamic = detail::rank_dynamic<Vals...>::value;

        TensorDimension() = default;

        TensorDimension(const TensorDimension &) = default;

        TensorDimension &operator=(const TensorDimension &) = default;

        FLARE_INLINE_FUNCTION
        constexpr TensorDimension(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4,
                                size_t n5, size_t n6, size_t n7)
                : D0(n0 == FLARE_INVALID_INDEX ? 1 : n0),
                  D1(n1 == FLARE_INVALID_INDEX ? 1 : n1),
                  D2(n2 == FLARE_INVALID_INDEX ? 1 : n2),
                  D3(n3 == FLARE_INVALID_INDEX ? 1 : n3),
                  D4(n4 == FLARE_INVALID_INDEX ? 1 : n4),
                  D5(n5 == FLARE_INVALID_INDEX ? 1 : n5),
                  D6(n6 == FLARE_INVALID_INDEX ? 1 : n6),
                  D7(n7 == FLARE_INVALID_INDEX ? 1 : n7) {}

        FLARE_INLINE_FUNCTION
        constexpr size_t extent(const unsigned r) const noexcept {
            return r == 0
                   ? N0
                   : (r == 1
                      ? N1
                      : (r == 2
                         ? N2
                         : (r == 3
                            ? N3
                            : (r == 4
                               ? N4
                               : (r == 5
                                  ? N5
                                  : (r == 6
                                     ? N6
                                     : (r == 7 ? N7
                                               : 0)))))));
        }

        static FLARE_INLINE_FUNCTION constexpr size_t static_extent(
                const unsigned r) noexcept {
            return r == 0
                   ? ArgN0
                   : (r == 1
                      ? ArgN1
                      : (r == 2
                         ? ArgN2
                         : (r == 3
                            ? ArgN3
                            : (r == 4
                               ? ArgN4
                               : (r == 5
                                  ? ArgN5
                                  : (r == 6
                                     ? ArgN6
                                     : (r == 7 ? ArgN7
                                               : 0)))))));
        }

        template<size_t N>
        struct prepend {
            using type = TensorDimension<N, Vals...>;
        };

        template<size_t N>
        struct append {
            using type = TensorDimension<Vals..., N>;
        };
    };

    template<class A, class B>
    struct TensorDimensionJoin;

    template<size_t... A, size_t... B>
    struct TensorDimensionJoin<TensorDimension<A...>, TensorDimension<B...>> {
        using type = TensorDimension<A..., B...>;
    };

//----------------------------------------------------------------------------

    template<class DstDim, class SrcDim>
    struct TensorDimensionAssignable;

    template<size_t... DstArgs, size_t... SrcArgs>
    struct TensorDimensionAssignable<TensorDimension<DstArgs...>,
            TensorDimension<SrcArgs...>> {
        using dst = TensorDimension<DstArgs...>;
        using src = TensorDimension<SrcArgs...>;

        enum {
            value = unsigned(dst::rank) == unsigned(src::rank) &&
                    (
                            // Compile time check that potential static dimensions match
                            ((1 > dst::rank_dynamic && 1 > src::rank_dynamic)
                             ? (size_t(dst::ArgN0) == size_t(src::ArgN0))
                             : true) &&
                            ((2 > dst::rank_dynamic && 2 > src::rank_dynamic)
                             ? (size_t(dst::ArgN1) == size_t(src::ArgN1))
                             : true) &&
                            ((3 > dst::rank_dynamic && 3 > src::rank_dynamic)
                             ? (size_t(dst::ArgN2) == size_t(src::ArgN2))
                             : true) &&
                            ((4 > dst::rank_dynamic && 4 > src::rank_dynamic)
                             ? (size_t(dst::ArgN3) == size_t(src::ArgN3))
                             : true) &&
                            ((5 > dst::rank_dynamic && 5 > src::rank_dynamic)
                             ? (size_t(dst::ArgN4) == size_t(src::ArgN4))
                             : true) &&
                            ((6 > dst::rank_dynamic && 6 > src::rank_dynamic)
                             ? (size_t(dst::ArgN5) == size_t(src::ArgN5))
                             : true) &&
                            ((7 > dst::rank_dynamic && 7 > src::rank_dynamic)
                             ? (size_t(dst::ArgN6) == size_t(src::ArgN6))
                             : true) &&
                            ((8 > dst::rank_dynamic && 8 > src::rank_dynamic)
                             ? (size_t(dst::ArgN7) == size_t(src::ArgN7))
                             : true))
        };
    };

}  // namespace flare::detail

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace flare {

    struct ALL_t {
        FLARE_INLINE_FUNCTION
        constexpr const ALL_t &operator()() const { return *this; }

        FLARE_INLINE_FUNCTION
        constexpr bool operator==(const ALL_t &) const { return true; }
    };

}  // namespace flare

namespace flare::detail {

    template<class T>
    struct is_integral_extent_type {
        enum : bool {
            value = std::is_same<T, flare::ALL_t>::value ? 1 : 0
        };
    };

    template<class iType>
    struct is_integral_extent_type<std::pair<iType, iType>> {
        enum : bool {
            value = std::is_integral<iType>::value ? 1 : 0
        };
    };

    template<class iType>
    struct is_integral_extent_type<flare::pair<iType, iType>> {
        enum : bool {
            value = std::is_integral<iType>::value ? 1 : 0
        };
    };

// Assuming '2 == initializer_list<iType>::size()'
    template<class iType>
    struct is_integral_extent_type<std::initializer_list<iType>> {
        enum : bool {
            value = std::is_integral<iType>::value ? 1 : 0
        };
    };

    template<unsigned I, class... Args>
    struct is_integral_extent {
        // get_type is void when sizeof...(Args) <= I
        using type = std::remove_cv_t<std::remove_reference_t<
                typename flare::detail::get_type<I, Args...>::type>>;

        enum : bool {
            value = is_integral_extent_type<type>::value
        };

        static_assert(value || std::is_integral<type>::value ||
                      std::is_void<type>::value,
                      "subtensor argument must be either integral or integral extent");
    };

// Rules for subtensor arguments and layouts matching

    template<class LayoutDest, class LayoutSrc, int RankDest, int RankSrc,
            int CurrentArg, class... SubTensorArgs>
    struct SubtensorLegalArgsCompileTime;

// Rules which allow LayoutLeft to LayoutLeft assignment

    template<int RankDest, int RankSrc, int CurrentArg, class Arg,
            class... SubTensorArgs>
    struct SubtensorLegalArgsCompileTime<flare::LayoutLeft, flare::LayoutLeft,
            RankDest, RankSrc, CurrentArg, Arg,
            SubTensorArgs...> {
        enum {
            value = (((CurrentArg == RankDest - 1) &&
                      (flare::detail::is_integral_extent_type<Arg>::value)) ||
                     ((CurrentArg >= RankDest) && (std::is_integral<Arg>::value)) ||
                     ((CurrentArg < RankDest) &&
                      (std::is_same<Arg, flare::ALL_t>::value)) ||
                     ((CurrentArg == 0) &&
                      (flare::detail::is_integral_extent_type<Arg>::value))) &&
                    (SubtensorLegalArgsCompileTime<flare::LayoutLeft, flare::LayoutLeft,
                            RankDest, RankSrc, CurrentArg + 1,
                            SubTensorArgs...>::value)
        };
    };

    template<int RankDest, int RankSrc, int CurrentArg, class Arg>
    struct SubtensorLegalArgsCompileTime<flare::LayoutLeft, flare::LayoutLeft,
            RankDest, RankSrc, CurrentArg, Arg> {
        enum {
            value = ((CurrentArg == RankDest - 1) || (std::is_integral<Arg>::value)) &&
                    (CurrentArg == RankSrc - 1)
        };
    };

// Rules which allow LayoutRight to LayoutRight assignment

    template<int RankDest, int RankSrc, int CurrentArg, class Arg,
            class... SubTensorArgs>
    struct SubtensorLegalArgsCompileTime<flare::LayoutRight, flare::LayoutRight,
            RankDest, RankSrc, CurrentArg, Arg,
            SubTensorArgs...> {
        enum {
            value = (((CurrentArg == RankSrc - RankDest) &&
                      (flare::detail::is_integral_extent_type<Arg>::value)) ||
                     ((CurrentArg < RankSrc - RankDest) &&
                      (std::is_integral<Arg>::value)) ||
                     ((CurrentArg >= RankSrc - RankDest) &&
                      (std::is_same<Arg, flare::ALL_t>::value))) &&
                    (SubtensorLegalArgsCompileTime<flare::LayoutRight,
                            flare::LayoutRight, RankDest, RankSrc,
                            CurrentArg + 1, SubTensorArgs...>::value)
        };
    };

    template<int RankDest, int RankSrc, int CurrentArg, class Arg>
    struct SubtensorLegalArgsCompileTime<flare::LayoutRight, flare::LayoutRight,
            RankDest, RankSrc, CurrentArg, Arg> {
        enum {
            value = ((CurrentArg == RankSrc - 1) &&
                     (std::is_same<Arg, flare::ALL_t>::value))
        };
    };

// Rules which allow assignment to LayoutStride

    template<int RankDest, int RankSrc, int CurrentArg, class... SubTensorArgs>
    struct SubtensorLegalArgsCompileTime<flare::LayoutStride, flare::LayoutLeft,
            RankDest, RankSrc, CurrentArg,
            SubTensorArgs...> {
        enum : bool {
            value = true
        };
    };

    template<int RankDest, int RankSrc, int CurrentArg, class... SubTensorArgs>
    struct SubtensorLegalArgsCompileTime<flare::LayoutStride, flare::LayoutRight,
            RankDest, RankSrc, CurrentArg,
            SubTensorArgs...> {
        enum : bool {
            value = true
        };
    };

    template<int RankDest, int RankSrc, int CurrentArg, class... SubTensorArgs>
    struct SubtensorLegalArgsCompileTime<flare::LayoutStride, flare::LayoutStride,
            RankDest, RankSrc, CurrentArg,
            SubTensorArgs...> {
        enum : bool {
            value = true
        };
    };

    template<unsigned DomainRank, unsigned RangeRank>
    struct SubTensorExtents {
    private:
        // Cannot declare zero-length arrays
        // '+' is used to silence GCC 7.2.0 -Wduplicated-branches warning when
        // RangeRank=1
        enum {
            InternalRangeRank = RangeRank ? RangeRank : +1u
        };

        size_t m_begin[DomainRank];
        size_t m_length[InternalRangeRank];
        unsigned m_index[InternalRangeRank];

        template<size_t... DimArgs>
        FLARE_FORCEINLINE_FUNCTION bool set(unsigned, unsigned,
                                            const TensorDimension<DimArgs...> &) {
            return true;
        }

        template<class T, size_t... DimArgs, class... Args>
        FLARE_FORCEINLINE_FUNCTION bool set(unsigned domain_rank,
                                            unsigned range_rank,
                                            const TensorDimension<DimArgs...> &dim,
                                            const T &val, Args... args) {
            const size_t v = static_cast<size_t>(val);

            m_begin[domain_rank] = v;

            return set(domain_rank + 1, range_rank, dim, args...)
#if defined(FLARE_ENABLE_DEBUG_BOUNDS_CHECK)
                && (v < dim.extent(domain_rank))
#endif
                    ;
        }

        // ALL_t
        template<size_t... DimArgs, class... Args>
        FLARE_FORCEINLINE_FUNCTION bool set(unsigned domain_rank,
                                            unsigned range_rank,
                                            const TensorDimension<DimArgs...> &dim,
                                            flare::ALL_t, Args... args) {
            m_begin[domain_rank] = 0;
            m_length[range_rank] = dim.extent(domain_rank);
            m_index[range_rank] = domain_rank;

            return set(domain_rank + 1, range_rank + 1, dim, args...);
        }

        // std::pair range
        template<class T, size_t... DimArgs, class... Args>
        FLARE_FORCEINLINE_FUNCTION bool set(unsigned domain_rank,
                                            unsigned range_rank,
                                            const TensorDimension<DimArgs...> &dim,
                                            const std::pair<T, T> &val,
                                            Args... args) {
            const size_t b = static_cast<size_t>(val.first);
            const size_t e = static_cast<size_t>(val.second);

            m_begin[domain_rank] = b;
            m_length[range_rank] = e - b;
            m_index[range_rank] = domain_rank;

            return set(domain_rank + 1, range_rank + 1, dim, args...)
#if defined(FLARE_ENABLE_DEBUG_BOUNDS_CHECK)
                && (e <= b + dim.extent(domain_rank))
#endif
                    ;
        }

        // flare::pair range
        template<class T, size_t... DimArgs, class... Args>
        FLARE_FORCEINLINE_FUNCTION bool set(unsigned domain_rank,
                                            unsigned range_rank,
                                            const TensorDimension<DimArgs...> &dim,
                                            const flare::pair<T, T> &val,
                                            Args... args) {
            const size_t b = static_cast<size_t>(val.first);
            const size_t e = static_cast<size_t>(val.second);

            m_begin[domain_rank] = b;
            m_length[range_rank] = e - b;
            m_index[range_rank] = domain_rank;

            return set(domain_rank + 1, range_rank + 1, dim, args...)
#if defined(FLARE_ENABLE_DEBUG_BOUNDS_CHECK)
                && (e <= b + dim.extent(domain_rank))
#endif
                    ;
        }

        // { begin , end } range
        template<class T, size_t... DimArgs, class... Args>
        FLARE_FORCEINLINE_FUNCTION bool set(unsigned domain_rank,
                                            unsigned range_rank,
                                            const TensorDimension<DimArgs...> &dim,
                                            const std::initializer_list<T> &val,
                                            Args... args) {
            const size_t b = static_cast<size_t>(val.begin()[0]);
            const size_t e = static_cast<size_t>(val.begin()[1]);

            m_begin[domain_rank] = b;
            m_length[range_rank] = e - b;
            m_index[range_rank] = domain_rank;

            return set(domain_rank + 1, range_rank + 1, dim, args...)
#if defined(FLARE_ENABLE_DEBUG_BOUNDS_CHECK)
                && (val.size() == 2) && (e <= b + dim.extent(domain_rank))
#endif
                    ;
        }

        //------------------------------

#if defined(FLARE_ENABLE_DEBUG_BOUNDS_CHECK)

                                                                                                                                template <size_t... DimArgs>
  void error(char*, int, unsigned, unsigned,
             const TensorDimension<DimArgs...>&) const {}

  template <class T, size_t... DimArgs, class... Args>
  void error(char* buf, int buf_len, unsigned domain_rank, unsigned range_rank,
             const TensorDimension<DimArgs...>& dim, const T& val,
             Args... args) const {
    const int n = std::min(
        buf_len,
        snprintf(buf, buf_len, " %lu < %lu %c", static_cast<unsigned long>(val),
                 static_cast<unsigned long>(dim.extent(domain_rank)),
                 int(sizeof...(Args) ? ',' : ')')));

    error(buf + n, buf_len - n, domain_rank + 1, range_rank, dim, args...);
  }

  // std::pair range
  template <size_t... DimArgs, class... Args>
  void error(char* buf, int buf_len, unsigned domain_rank, unsigned range_rank,
             const TensorDimension<DimArgs...>& dim, flare::ALL_t,
             Args... args) const {
    const int n = std::min(buf_len, snprintf(buf, buf_len, " flare::ALL %c",
                                             int(sizeof...(Args) ? ',' : ')')));

    error(buf + n, buf_len - n, domain_rank + 1, range_rank + 1, dim, args...);
  }

  // std::pair range
  template <class T, size_t... DimArgs, class... Args>
  void error(char* buf, int buf_len, unsigned domain_rank, unsigned range_rank,
             const TensorDimension<DimArgs...>& dim, const std::pair<T, T>& val,
             Args... args) const {
    // d <= e - b
    const int n = std::min(
        buf_len, snprintf(buf, buf_len, " %lu <= %lu - %lu %c",
                          static_cast<unsigned long>(dim.extent(domain_rank)),
                          static_cast<unsigned long>(val.second),
                          static_cast<unsigned long>(val.first),
                          int(sizeof...(Args) ? ',' : ')')));

    error(buf + n, buf_len - n, domain_rank + 1, range_rank + 1, dim, args...);
  }

  // flare::pair range
  template <class T, size_t... DimArgs, class... Args>
  void error(char* buf, int buf_len, unsigned domain_rank, unsigned range_rank,
             const TensorDimension<DimArgs...>& dim,
             const flare::pair<T, T>& val, Args... args) const {
    // d <= e - b
    const int n = std::min(
        buf_len, snprintf(buf, buf_len, " %lu <= %lu - %lu %c",
                          static_cast<unsigned long>(dim.extent(domain_rank)),
                          static_cast<unsigned long>(val.second),
                          static_cast<unsigned long>(val.first),
                          int(sizeof...(Args) ? ',' : ')')));

    error(buf + n, buf_len - n, domain_rank + 1, range_rank + 1, dim, args...);
  }

  // { begin , end } range
  template <class T, size_t... DimArgs, class... Args>
  void error(char* buf, int buf_len, unsigned domain_rank, unsigned range_rank,
             const TensorDimension<DimArgs...>& dim,
             const std::initializer_list<T>& val, Args... args) const {
    // d <= e - b
    int n = 0;
    if (val.size() == 2) {
      n = std::min(buf_len,
                   snprintf(buf, buf_len, " %lu <= %lu - %lu %c",
                            static_cast<unsigned long>(dim.extent(domain_rank)),
                            static_cast<unsigned long>(val.begin()[0]),
                            static_cast<unsigned long>(val.begin()[1]),
                            int(sizeof...(Args) ? ',' : ')')));
    } else {
      n = std::min(buf_len, snprintf(buf, buf_len, " { ... }.size() == %u %c",
                                     unsigned(val.size()),
                                     int(sizeof...(Args) ? ',' : ')')));
    }

    error(buf + n, buf_len - n, domain_rank + 1, range_rank + 1, dim, args...);
  }

  template <size_t... DimArgs, class... Args>
  FLARE_FORCEINLINE_FUNCTION void error(const TensorDimension<DimArgs...>& dim,
                                         Args... args) const {
    FLARE_IF_ON_HOST(
        (enum {LEN = 1024}; char buffer[LEN];

         const int n = snprintf(buffer, LEN, "flare::subtensor bounds error (");
         error(buffer + n, LEN - n, 0, 0, dim, args...);

         flare::detail::throw_runtime_exception(std::string(buffer));))

    FLARE_IF_ON_DEVICE(((void)dim;
                         flare::abort("flare::subtensor bounds error");
                         [](Args...) {}(args...);))
  }

#else

        template<size_t... DimArgs, class... Args>
        FLARE_FORCEINLINE_FUNCTION void error(const TensorDimension<DimArgs...> &,
                                              Args...) const {}

#endif

    public:
        template<size_t... DimArgs, class... Args>
        FLARE_INLINE_FUNCTION SubTensorExtents(const TensorDimension<DimArgs...> &dim,
                                             Args... args) {
            static_assert(DomainRank == sizeof...(DimArgs), "");
            static_assert(DomainRank == sizeof...(Args), "");

            // Verifies that all arguments, up to 8, are integral types,
            // integral extents, or don't exist.
            static_assert(
                    RangeRank == unsigned(is_integral_extent<0, Args...>::value) +
                                 unsigned(is_integral_extent<1, Args...>::value) +
                                 unsigned(is_integral_extent<2, Args...>::value) +
                                 unsigned(is_integral_extent<3, Args...>::value) +
                                 unsigned(is_integral_extent<4, Args...>::value) +
                                 unsigned(is_integral_extent<5, Args...>::value) +
                                 unsigned(is_integral_extent<6, Args...>::value) +
                                 unsigned(is_integral_extent<7, Args...>::value),
                    "");

            if (RangeRank == 0) {
                m_length[0] = 0;
                m_index[0] = ~0u;
            }

            if (!set(0, 0, dim, args...)) error(dim, args...);
        }

        template<typename iType>
        FLARE_FORCEINLINE_FUNCTION constexpr size_t domain_offset(
                const iType i) const {
            return unsigned(i) < DomainRank ? m_begin[i] : 0;
        }

        template<typename iType>
        FLARE_FORCEINLINE_FUNCTION constexpr size_t range_extent(
                const iType i) const {
            return unsigned(i) < InternalRangeRank ? m_length[i] : 0;
        }

        template<typename iType>
        FLARE_FORCEINLINE_FUNCTION constexpr unsigned range_index(
                const iType i) const {
            return unsigned(i) < InternalRangeRank ? m_index[i] : ~0u;
        }
    };


/** \brief  Given a value type and dimension generate the Tensor data type */
    template<class T, class Dim>
    struct TensorDataType_;

    template<class T>
    struct TensorDataType_<T, TensorDimension<>> {
        using type = T;
    };

    template<class T, size_t... Args>
    struct TensorDataType_<T, TensorDimension<0, Args...>> {
        using type = typename TensorDataType_<T *, TensorDimension<Args...>>::type;
    };

    template<class T, size_t N, size_t... Args>
    struct TensorDataType_<T, TensorDimension<N, Args...>> {
        using type = typename TensorDataType_<T, TensorDimension<Args...>>::type[N];
    };

/**\brief  Analysis of Tensor data type.
 *
 *  Data type conforms to one of the following patterns :
 *    {const} value_type [][#][#][#]
 *    {const} value_type ***[#][#][#]
 *  Where the sum of counts of '*' and '[#]' is at most ten.
 *
 *  Provide alias for TensorDimension<...> and value_type.
 */
    template<class T>
    struct TensorArrayAnalysis {
        using value_type = T;
        using const_value_type = std::add_const_t<T>;
        using non_const_value_type = std::remove_const_t<T>;
        using static_dimension = TensorDimension<>;
        using dynamic_dimension = TensorDimension<>;
        using dimension = TensorDimension<>;
    };

    template<class T, size_t N>
    struct TensorArrayAnalysis<T[N]> {
    private:
        using nested = TensorArrayAnalysis<T>;

    public:
        using value_type = typename nested::value_type;
        using const_value_type = typename nested::const_value_type;
        using non_const_value_type = typename nested::non_const_value_type;

        using static_dimension =
                typename nested::static_dimension::template prepend<N>::type;

        using dynamic_dimension = typename nested::dynamic_dimension;

        using dimension =
                typename TensorDimensionJoin<dynamic_dimension, static_dimension>::type;
    };

    template<class T>
    struct TensorArrayAnalysis<T[]> {
    private:
        using nested = TensorArrayAnalysis<T>;
        using nested_dimension = typename nested::dimension;

    public:
        using value_type = typename nested::value_type;
        using const_value_type = typename nested::const_value_type;
        using non_const_value_type = typename nested::non_const_value_type;

        using dynamic_dimension =
                typename nested::dynamic_dimension::template prepend<0>::type;

        using static_dimension = typename nested::static_dimension;

        using dimension =
                typename TensorDimensionJoin<dynamic_dimension, static_dimension>::type;
    };

    template<class T>
    struct TensorArrayAnalysis<T *> {
    private:
        using nested = TensorArrayAnalysis<T>;

    public:
        using value_type = typename nested::value_type;
        using const_value_type = typename nested::const_value_type;
        using non_const_value_type = typename nested::non_const_value_type;

        using dynamic_dimension =
                typename nested::dynamic_dimension::template prepend<0>::type;

        using static_dimension = typename nested::static_dimension;

        using dimension =
                typename TensorDimensionJoin<dynamic_dimension, static_dimension>::type;
    };

    template<class DataType, class ArrayLayout, class ValueType>
    struct TensorDataAnalysis {
    private:
        using array_analysis = TensorArrayAnalysis<DataType>;

        // ValueType is opportunity for partial specialization.
        // Must match array analysis when this default template is used.
        static_assert(
                std::is_same<ValueType,
                        typename array_analysis::non_const_value_type>::value,
                "");

    public:
        using specialize = void;  // No specialization

        using dimension = typename array_analysis::dimension;
        using value_type = typename array_analysis::value_type;
        using const_value_type = typename array_analysis::const_value_type;
        using non_const_value_type = typename array_analysis::non_const_value_type;

        // Generate analogous multidimensional array specification type.
        using type = typename TensorDataType_<value_type, dimension>::type;
        using const_type = typename TensorDataType_<const_value_type, dimension>::type;
        using non_const_type =
                typename TensorDataType_<non_const_value_type, dimension>::type;

        // Generate "flattened" multidimensional array specification type.
        using scalar_array_type = type;
        using const_scalar_array_type = const_type;
        using non_const_scalar_array_type = non_const_type;
    };


    template<class Dimension, class Layout, class Enable = void>
    struct TensorOffset {
        using is_mapping_plugin = std::false_type;
    };

    //----------------------------------------------------------------------------
    // LayoutLeft AND ( 1 >= rank OR 0 == rank_dynamic ) : no padding / striding
    template<class Dimension>
    struct TensorOffset<
            Dimension, flare::LayoutLeft,
            std::enable_if_t<(1 >= Dimension::rank || 0 == Dimension::rank_dynamic)>> {
        using is_mapping_plugin = std::true_type;
        using is_regular = std::true_type;

        using size_type = size_t;
        using dimension_type = Dimension;
        using array_layout = flare::LayoutLeft;

        dimension_type m_dim;

        //----------------------------------------

        // rank 1
        template<typename I0>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0) const {
            return i0;
        }

        // rank 2
        template<typename I0, typename I1>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1) const {
            return i0 + m_dim.N0 * i1;
        }

        // rank 3
        template<typename I0, typename I1, typename I2>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1,
                                                             I2 const &i2) const {
            return i0 + m_dim.N0 * (i1 + m_dim.N1 * i2);
        }

        // rank 4
        template<typename I0, typename I1, typename I2, typename I3>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1,
                                                             I2 const &i2,
                                                             I3 const &i3) const {
            return i0 + m_dim.N0 * (i1 + m_dim.N1 * (i2 + m_dim.N2 * i3));
        }

        // rank 5
        template<typename I0, typename I1, typename I2, typename I3, typename I4>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1,
                                                             I2 const &i2,
                                                             I3 const &i3,
                                                             I4 const &i4) const {
            return i0 +
                   m_dim.N0 * (i1 + m_dim.N1 * (i2 + m_dim.N2 * (i3 + m_dim.N3 * i4)));
        }

        // rank 6
        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(
                I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
                I5 const &i5) const {
            return i0 +
                   m_dim.N0 *
                   (i1 +
                    m_dim.N1 *
                    (i2 + m_dim.N2 * (i3 + m_dim.N3 * (i4 + m_dim.N4 * i5))));
        }

        // rank 7
        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5, typename I6>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(
                I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
                I5 const &i5, I6 const &i6) const {
            return i0 +
                   m_dim.N0 *
                   (i1 + m_dim.N1 *
                         (i2 + m_dim.N2 *
                               (i3 + m_dim.N3 *
                                     (i4 + m_dim.N4 *
                                           (i5 + m_dim.N5 * i6)))));
        }

        // rank 8
        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5, typename I6, typename I7>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(
                I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
                I5 const &i5, I6 const &i6, I7 const &i7) const {
            return i0 +
                   m_dim.N0 *
                   (i1 +
                    m_dim.N1 *
                    (i2 + m_dim.N2 *
                          (i3 + m_dim.N3 *
                                (i4 + m_dim.N4 *
                                      (i5 + m_dim.N5 *
                                            (i6 + m_dim.N6 *
                                                  i7))))));
        }

        //----------------------------------------

        FLARE_INLINE_FUNCTION
        constexpr array_layout layout() const {
            constexpr auto r = dimension_type::rank;
            return array_layout((r > 0 ? m_dim.N0 : FLARE_INVALID_INDEX),
                                (r > 1 ? m_dim.N1 : FLARE_INVALID_INDEX),
                                (r > 2 ? m_dim.N2 : FLARE_INVALID_INDEX),
                                (r > 3 ? m_dim.N3 : FLARE_INVALID_INDEX),
                                (r > 4 ? m_dim.N4 : FLARE_INVALID_INDEX),
                                (r > 5 ? m_dim.N5 : FLARE_INVALID_INDEX),
                                (r > 6 ? m_dim.N6 : FLARE_INVALID_INDEX),
                                (r > 7 ? m_dim.N7 : FLARE_INVALID_INDEX));
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_0() const {
            return m_dim.N0;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_1() const {
            return m_dim.N1;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_2() const {
            return m_dim.N2;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_3() const {
            return m_dim.N3;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_4() const {
            return m_dim.N4;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_5() const {
            return m_dim.N5;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_6() const {
            return m_dim.N6;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_7() const {
            return m_dim.N7;
        }

        /* Cardinality of the domain index space */
        FLARE_INLINE_FUNCTION
        constexpr size_type size() const {
            return size_type(m_dim.N0) * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 *
                   m_dim.N5 * m_dim.N6 * m_dim.N7;
        }

        /* Span of the range space */
        FLARE_INLINE_FUNCTION
        constexpr size_type span() const {
            return size_type(m_dim.N0) * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 *
                   m_dim.N5 * m_dim.N6 * m_dim.N7;
        }

        FLARE_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
            return true;
        }

        /* Strides of dimensions */
        FLARE_INLINE_FUNCTION constexpr size_type stride_0() const { return 1; }

        FLARE_INLINE_FUNCTION constexpr size_type stride_1() const {
            return m_dim.N0;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_2() const {
            return size_type(m_dim.N0) * m_dim.N1;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_3() const {
            return size_type(m_dim.N0) * m_dim.N1 * m_dim.N2;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_4() const {
            return size_type(m_dim.N0) * m_dim.N1 * m_dim.N2 * m_dim.N3;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_5() const {
            return size_type(m_dim.N0) * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_6() const {
            return size_type(m_dim.N0) * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 *
                   m_dim.N5;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_7() const {
            return size_type(m_dim.N0) * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 *
                   m_dim.N5 * m_dim.N6;
        }

        // Stride with [ rank ] value is the total length
        template<typename iType>
        FLARE_INLINE_FUNCTION void stride(iType *const s) const {
            s[0] = 1;
            if (0 < dimension_type::rank) {
                s[1] = m_dim.N0;
            }
            if (1 < dimension_type::rank) {
                s[2] = s[1] * m_dim.N1;
            }
            if (2 < dimension_type::rank) {
                s[3] = s[2] * m_dim.N2;
            }
            if (3 < dimension_type::rank) {
                s[4] = s[3] * m_dim.N3;
            }
            if (4 < dimension_type::rank) {
                s[5] = s[4] * m_dim.N4;
            }
            if (5 < dimension_type::rank) {
                s[6] = s[5] * m_dim.N5;
            }
            if (6 < dimension_type::rank) {
                s[7] = s[6] * m_dim.N6;
            }
            if (7 < dimension_type::rank) {
                s[8] = s[7] * m_dim.N7;
            }
        }

        //----------------------------------------

        // MSVC (16.5.5) + CUDA (10.2) did not generate the defaulted functions
        // correct and errors out during compilation. Same for the other places where
        // I changed this.
#ifdef FLARE_IMPL_WINDOWS_CUDA
                                                                                                                                FLARE_FUNCTION TensorOffset() : m_dim(dimension_type()) {}
  FLARE_FUNCTION TensorOffset(const TensorOffset& src) { m_dim = src.m_dim; }
  FLARE_FUNCTION TensorOffset& operator=(const TensorOffset& src) {
    m_dim = src.m_dim;
    return *this;
  }
#else

        TensorOffset() = default;

        TensorOffset(const TensorOffset &) = default;

        TensorOffset &operator=(const TensorOffset &) = default;

#endif

        template<unsigned TrivialScalarSize>
        FLARE_INLINE_FUNCTION constexpr TensorOffset(
                std::integral_constant<unsigned, TrivialScalarSize> const &,
                flare::LayoutLeft const &arg_layout)
                : m_dim(arg_layout.dimension[0], 0, 0, 0, 0, 0, 0, 0) {}

        template<class DimRHS>
        FLARE_INLINE_FUNCTION constexpr TensorOffset(
                const TensorOffset<DimRHS, flare::LayoutLeft, void> &rhs)
                : m_dim(rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3,
                        rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7) {
            static_assert(int(DimRHS::rank) == int(dimension_type::rank),
                          "TensorOffset assignment requires equal rank");
            // Also requires equal static dimensions ...
        }

        template<class DimRHS>
        FLARE_INLINE_FUNCTION constexpr TensorOffset(
                const TensorOffset<DimRHS, flare::LayoutRight, void> &rhs)
                : m_dim(rhs.m_dim.N0, 0, 0, 0, 0, 0, 0, 0) {
            static_assert(((DimRHS::rank == 0 && dimension_type::rank == 0) ||
                           (DimRHS::rank == 1 && dimension_type::rank == 1)),
                          "TensorOffset LayoutLeft and LayoutRight are only compatible "
                          "when rank <= 1");
        }

        template<class DimRHS>
        FLARE_INLINE_FUNCTION TensorOffset(
                const TensorOffset<DimRHS, flare::LayoutStride, void> &rhs)
                : m_dim(rhs.m_dim.N0, 0, 0, 0, 0, 0, 0, 0) {
            if (rhs.m_stride.S0 != 1) {
                flare::abort(
                        "flare::detail::TensorOffset assignment of LayoutLeft from LayoutStride "
                        " requires stride == 1");
            }
        }

        //----------------------------------------
        // Subtensor construction

        template<class DimRHS>
        FLARE_INLINE_FUNCTION constexpr TensorOffset(
                const TensorOffset<DimRHS, flare::LayoutLeft, void> &,
                const SubTensorExtents<DimRHS::rank, dimension_type::rank> &sub)
                : m_dim(sub.range_extent(0), 0, 0, 0, 0, 0, 0, 0) {
            static_assert((0 == dimension_type::rank_dynamic) ||
                          (1 == dimension_type::rank &&
                           1 == dimension_type::rank_dynamic && 1 <= DimRHS::rank),
                          "TensorOffset subtensor construction requires compatible rank");
        }
    };

//----------------------------------------------------------------------------
// LayoutLeft AND ( 1 < rank AND 0 < rank_dynamic ) : has padding / striding
    template<class Dimension>
    struct TensorOffset<
            Dimension, flare::LayoutLeft,
            std::enable_if_t<(1 < Dimension::rank && 0 < Dimension::rank_dynamic)>> {
        using is_mapping_plugin = std::true_type;
        using is_regular = std::true_type;

        using size_type = size_t;
        using dimension_type = Dimension;
        using array_layout = flare::LayoutLeft;

        dimension_type m_dim;
        size_type m_stride;

        //----------------------------------------

        // rank 1
        template<typename I0>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0) const {
            return i0;
        }

        // rank 2
        template<typename I0, typename I1>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1) const {
            return i0 + m_stride * i1;
        }

        // rank 3
        template<typename I0, typename I1, typename I2>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1,
                                                             I2 const &i2) const {
            return i0 + m_stride * (i1 + m_dim.N1 * i2);
        }

        // rank 4
        template<typename I0, typename I1, typename I2, typename I3>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1,
                                                             I2 const &i2,
                                                             I3 const &i3) const {
            return i0 + m_stride * (i1 + m_dim.N1 * (i2 + m_dim.N2 * i3));
        }

        // rank 5
        template<typename I0, typename I1, typename I2, typename I3, typename I4>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1,
                                                             I2 const &i2,
                                                             I3 const &i3,
                                                             I4 const &i4) const {
            return i0 +
                   m_stride * (i1 + m_dim.N1 * (i2 + m_dim.N2 * (i3 + m_dim.N3 * i4)));
        }

        // rank 6
        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(
                I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
                I5 const &i5) const {
            return i0 +
                   m_stride *
                   (i1 +
                    m_dim.N1 *
                    (i2 + m_dim.N2 * (i3 + m_dim.N3 * (i4 + m_dim.N4 * i5))));
        }

        // rank 7
        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5, typename I6>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(
                I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
                I5 const &i5, I6 const &i6) const {
            return i0 +
                   m_stride *
                   (i1 + m_dim.N1 *
                         (i2 + m_dim.N2 *
                               (i3 + m_dim.N3 *
                                     (i4 + m_dim.N4 *
                                           (i5 + m_dim.N5 * i6)))));
        }

        // rank 8
        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5, typename I6, typename I7>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(
                I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
                I5 const &i5, I6 const &i6, I7 const &i7) const {
            return i0 +
                   m_stride *
                   (i1 +
                    m_dim.N1 *
                    (i2 + m_dim.N2 *
                          (i3 + m_dim.N3 *
                                (i4 + m_dim.N4 *
                                      (i5 + m_dim.N5 *
                                            (i6 + m_dim.N6 *
                                                  i7))))));
        }

        //----------------------------------------

        FLARE_INLINE_FUNCTION
        constexpr array_layout layout() const {
            constexpr auto r = dimension_type::rank;
            return array_layout((r > 0 ? m_dim.N0 : FLARE_INVALID_INDEX),
                                (r > 1 ? m_dim.N1 : FLARE_INVALID_INDEX),
                                (r > 2 ? m_dim.N2 : FLARE_INVALID_INDEX),
                                (r > 3 ? m_dim.N3 : FLARE_INVALID_INDEX),
                                (r > 4 ? m_dim.N4 : FLARE_INVALID_INDEX),
                                (r > 5 ? m_dim.N5 : FLARE_INVALID_INDEX),
                                (r > 6 ? m_dim.N6 : FLARE_INVALID_INDEX),
                                (r > 7 ? m_dim.N7 : FLARE_INVALID_INDEX));
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_0() const {
            return m_dim.N0;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_1() const {
            return m_dim.N1;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_2() const {
            return m_dim.N2;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_3() const {
            return m_dim.N3;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_4() const {
            return m_dim.N4;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_5() const {
            return m_dim.N5;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_6() const {
            return m_dim.N6;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_7() const {
            return m_dim.N7;
        }

        /* Cardinality of the domain index space */
        FLARE_INLINE_FUNCTION
        constexpr size_type size() const {
            return size_type(m_dim.N0) * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 *
                   m_dim.N5 * m_dim.N6 * m_dim.N7;
        }

        /* Span of the range space */
        FLARE_INLINE_FUNCTION
        constexpr size_type span() const {
            return (m_dim.N0 > size_type(0) ? m_stride : size_type(0)) * m_dim.N1 *
                   m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 * m_dim.N6 * m_dim.N7;
        }

        FLARE_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
            return m_stride == m_dim.N0;
        }

        /* Strides of dimensions */
        FLARE_INLINE_FUNCTION constexpr size_type stride_0() const { return 1; }

        FLARE_INLINE_FUNCTION constexpr size_type stride_1() const {
            return m_stride;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_2() const {
            return m_stride * m_dim.N1;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_3() const {
            return m_stride * m_dim.N1 * m_dim.N2;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_4() const {
            return m_stride * m_dim.N1 * m_dim.N2 * m_dim.N3;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_5() const {
            return m_stride * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_6() const {
            return m_stride * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_7() const {
            return m_stride * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 *
                   m_dim.N6;
        }

        // Stride with [ rank ] value is the total length
        template<typename iType>
        FLARE_INLINE_FUNCTION void stride(iType *const s) const {
            s[0] = 1;
            if (0 < dimension_type::rank) {
                s[1] = m_stride;
            }
            if (1 < dimension_type::rank) {
                s[2] = s[1] * m_dim.N1;
            }
            if (2 < dimension_type::rank) {
                s[3] = s[2] * m_dim.N2;
            }
            if (3 < dimension_type::rank) {
                s[4] = s[3] * m_dim.N3;
            }
            if (4 < dimension_type::rank) {
                s[5] = s[4] * m_dim.N4;
            }
            if (5 < dimension_type::rank) {
                s[6] = s[5] * m_dim.N5;
            }
            if (6 < dimension_type::rank) {
                s[7] = s[6] * m_dim.N6;
            }
            if (7 < dimension_type::rank) {
                s[8] = s[7] * m_dim.N7;
            }
        }

        //----------------------------------------

    private:
        template<unsigned TrivialScalarSize>
        struct Padding {
            enum {
                div = TrivialScalarSize == 0
                      ? 0
                      : flare::detail::MEMORY_ALIGNMENT /
                        (TrivialScalarSize ? TrivialScalarSize : 1)
            };
            enum {
                mod = TrivialScalarSize == 0
                      ? 0
                      : flare::detail::MEMORY_ALIGNMENT %
                        (TrivialScalarSize ? TrivialScalarSize : 1)
            };

            // If memory alignment is a multiple of the trivial scalar size then attempt
            // to align.
            enum {
                align = 0 != TrivialScalarSize && 0 == mod ? div : 0
            };
            enum {
                div_ok = (div != 0) ? div : 1
            };  // To valid modulo zero in constexpr

            FLARE_INLINE_FUNCTION
            static constexpr size_t stride(size_t const N) {
                return ((align != 0) &&
                        ((static_cast<int>(flare::detail::MEMORY_ALIGNMENT_THRESHOLD) *
                          static_cast<int>(align)) < N) &&
                        ((N % div_ok) != 0))
                       ? N + align - (N % div_ok)
                       : N;
            }
        };

    public:
        // MSVC (16.5.5) + CUDA (10.2) did not generate the defaulted functions
        // correct and errors out during compilation. Same for the other places where
        // I changed this.
#ifdef FLARE_IMPL_WINDOWS_CUDA
                                                                                                                                FLARE_FUNCTION TensorOffset() : m_dim(dimension_type()), m_stride(0) {}
  FLARE_FUNCTION TensorOffset(const TensorOffset& src) {
    m_dim    = src.m_dim;
    m_stride = src.m_stride;
  }
  FLARE_FUNCTION TensorOffset& operator=(const TensorOffset& src) {
    m_dim    = src.m_dim;
    m_stride = src.m_stride;
    return *this;
  }
#else

        TensorOffset() = default;

        TensorOffset(const TensorOffset &) = default;

        TensorOffset &operator=(const TensorOffset &) = default;

#endif

        /* Enable padding for trivial scalar types with non-zero trivial scalar size
   */
        template<unsigned TrivialScalarSize>
        FLARE_INLINE_FUNCTION constexpr TensorOffset(
                std::integral_constant<unsigned, TrivialScalarSize> const &,
                flare::LayoutLeft const &arg_layout)
                : m_dim(arg_layout.dimension[0], arg_layout.dimension[1],
                        arg_layout.dimension[2], arg_layout.dimension[3],
                        arg_layout.dimension[4], arg_layout.dimension[5],
                        arg_layout.dimension[6], arg_layout.dimension[7]),
                  m_stride(Padding<TrivialScalarSize>::stride(arg_layout.dimension[0])) {}

        template<class DimRHS>
        FLARE_INLINE_FUNCTION constexpr TensorOffset(
                const TensorOffset<DimRHS, flare::LayoutLeft, void> &rhs)
                : m_dim(rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3,
                        rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7),
                  m_stride(rhs.stride_1()) {
            static_assert(int(DimRHS::rank) == int(dimension_type::rank),
                          "TensorOffset assignment requires equal rank");
            // Also requires equal static dimensions ...
        }

        template<class DimRHS>
        FLARE_INLINE_FUNCTION TensorOffset(
                const TensorOffset<DimRHS, flare::LayoutStride, void> &rhs)
                : m_dim(rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3,
                        rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7),
                  m_stride(rhs.stride_1()) {
            if (rhs.m_stride.S0 != 1) {
                flare::abort(
                        "flare::detail::TensorOffset assignment of LayoutLeft from LayoutStride "
                        "requires stride == 1");
            }
        }

        //----------------------------------------
        // Subtensor construction
        // This subtensor must be 2 == rank and 2 == rank_dynamic
        // due to only having stride #0.
        // The source dimension #0 must be non-zero for stride-one leading dimension.
        // At most subsequent dimension can be non-zero.

        template<class DimRHS>
        FLARE_INLINE_FUNCTION constexpr TensorOffset(
                const TensorOffset<DimRHS, flare::LayoutLeft, void> &rhs,
                const SubTensorExtents<DimRHS::rank, dimension_type::rank> &sub)
                : m_dim(sub.range_extent(0), sub.range_extent(1), sub.range_extent(2),
                        sub.range_extent(3), sub.range_extent(4), sub.range_extent(5),
                        sub.range_extent(6), sub.range_extent(7)),
                  m_stride(
                          (1 == sub.range_index(1)
                           ? rhs.stride_1()
                           : (2 == sub.range_index(1)
                              ? rhs.stride_2()
                              : (3 == sub.range_index(1)
                                 ? rhs.stride_3()
                                 : (4 == sub.range_index(1)
                                    ? rhs.stride_4()
                                    : (5 == sub.range_index(1)
                                       ? rhs.stride_5()
                                       : (6 == sub.range_index(1)
                                          ? rhs.stride_6()
                                          : (7 == sub.range_index(1)
                                             ? rhs.stride_7()
                                             : 0)))))))) {
            // static_assert( ( 2 == dimension_type::rank ) &&
            //               ( 2 == dimension_type::rank_dynamic ) &&
            //               ( 2 <= DimRHS::rank )
            //             , "TensorOffset subtensor construction requires compatible rank"
            //             );
        }
    };

//----------------------------------------------------------------------------
// LayoutRight AND ( 1 >= rank OR 0 == rank_dynamic ) : no padding / striding
    template<class Dimension>
    struct TensorOffset<
            Dimension, flare::LayoutRight,
            std::enable_if_t<(1 >= Dimension::rank || 0 == Dimension::rank_dynamic)>> {
        using is_mapping_plugin = std::true_type;
        using is_regular = std::true_type;

        using size_type = size_t;
        using dimension_type = Dimension;
        using array_layout = flare::LayoutRight;

        dimension_type m_dim;

        //----------------------------------------

        // rank 1
        template<typename I0>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0) const {
            return i0;
        }

        // rank 2
        template<typename I0, typename I1>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1) const {
            return i1 + m_dim.N1 * i0;
        }

        // rank 3
        template<typename I0, typename I1, typename I2>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1,
                                                             I2 const &i2) const {
            return i2 + m_dim.N2 * (i1 + m_dim.N1 * (i0));
        }

        // rank 4
        template<typename I0, typename I1, typename I2, typename I3>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1,
                                                             I2 const &i2,
                                                             I3 const &i3) const {
            return i3 + m_dim.N3 * (i2 + m_dim.N2 * (i1 + m_dim.N1 * (i0)));
        }

        // rank 5
        template<typename I0, typename I1, typename I2, typename I3, typename I4>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1,
                                                             I2 const &i2,
                                                             I3 const &i3,
                                                             I4 const &i4) const {
            return i4 + m_dim.N4 *
                        (i3 + m_dim.N3 * (i2 + m_dim.N2 * (i1 + m_dim.N1 * (i0))));
        }

        // rank 6
        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(
                I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
                I5 const &i5) const {
            return i5 +
                   m_dim.N5 *
                   (i4 +
                    m_dim.N4 *
                    (i3 + m_dim.N3 * (i2 + m_dim.N2 * (i1 + m_dim.N1 * (i0)))));
        }

        // rank 7
        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5, typename I6>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(
                I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
                I5 const &i5, I6 const &i6) const {
            return i6 +
                   m_dim.N6 *
                   (i5 +
                    m_dim.N5 *
                    (i4 +
                     m_dim.N4 *
                     (i3 + m_dim.N3 *
                           (i2 + m_dim.N2 * (i1 + m_dim.N1 * (i0))))));
        }

        // rank 8
        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5, typename I6, typename I7>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(
                I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
                I5 const &i5, I6 const &i6, I7 const &i7) const {
            return i7 +
                   m_dim.N7 *
                   (i6 +
                    m_dim.N6 *
                    (i5 +
                     m_dim.N5 *
                     (i4 +
                      m_dim.N4 *
                      (i3 +
                       m_dim.N3 *
                       (i2 + m_dim.N2 * (i1 + m_dim.N1 * (i0)))))));
        }

        //----------------------------------------

        FLARE_INLINE_FUNCTION
        constexpr array_layout layout() const {
            constexpr auto r = dimension_type::rank;
            return array_layout((r > 0 ? m_dim.N0 : FLARE_INVALID_INDEX),
                                (r > 1 ? m_dim.N1 : FLARE_INVALID_INDEX),
                                (r > 2 ? m_dim.N2 : FLARE_INVALID_INDEX),
                                (r > 3 ? m_dim.N3 : FLARE_INVALID_INDEX),
                                (r > 4 ? m_dim.N4 : FLARE_INVALID_INDEX),
                                (r > 5 ? m_dim.N5 : FLARE_INVALID_INDEX),
                                (r > 6 ? m_dim.N6 : FLARE_INVALID_INDEX),
                                (r > 7 ? m_dim.N7 : FLARE_INVALID_INDEX));
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_0() const {
            return m_dim.N0;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_1() const {
            return m_dim.N1;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_2() const {
            return m_dim.N2;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_3() const {
            return m_dim.N3;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_4() const {
            return m_dim.N4;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_5() const {
            return m_dim.N5;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_6() const {
            return m_dim.N6;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_7() const {
            return m_dim.N7;
        }

        /* Cardinality of the domain index space */
        FLARE_INLINE_FUNCTION
        constexpr size_type size() const {
            return size_type(m_dim.N0) * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 *
                   m_dim.N5 * m_dim.N6 * m_dim.N7;
        }

        /* Span of the range space */
        FLARE_INLINE_FUNCTION
        constexpr size_type span() const {
            return size_type(m_dim.N0) * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 *
                   m_dim.N5 * m_dim.N6 * m_dim.N7;
        }

        FLARE_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
            return true;
        }

        /* Strides of dimensions */
        FLARE_INLINE_FUNCTION constexpr size_type stride_7() const { return 1; }

        FLARE_INLINE_FUNCTION constexpr size_type stride_6() const {
            return m_dim.N7;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_5() const {
            return m_dim.N7 * m_dim.N6;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_4() const {
            return m_dim.N7 * m_dim.N6 * m_dim.N5;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_3() const {
            return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_2() const {
            return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_1() const {
            return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3 * m_dim.N2;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_0() const {
            return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3 * m_dim.N2 *
                   m_dim.N1;
        }

        // Stride with [ rank ] value is the total length
        template<typename iType>
        FLARE_INLINE_FUNCTION void stride(iType *const s) const {
            size_type n = 1;
            if (7 < dimension_type::rank) {
                s[7] = n;
                n *= m_dim.N7;
            }
            if (6 < dimension_type::rank) {
                s[6] = n;
                n *= m_dim.N6;
            }
            if (5 < dimension_type::rank) {
                s[5] = n;
                n *= m_dim.N5;
            }
            if (4 < dimension_type::rank) {
                s[4] = n;
                n *= m_dim.N4;
            }
            if (3 < dimension_type::rank) {
                s[3] = n;
                n *= m_dim.N3;
            }
            if (2 < dimension_type::rank) {
                s[2] = n;
                n *= m_dim.N2;
            }
            if (1 < dimension_type::rank) {
                s[1] = n;
                n *= m_dim.N1;
            }
            if (0 < dimension_type::rank) {
                s[0] = n;
            }
            s[dimension_type::rank] = n * m_dim.N0;
        }

        //----------------------------------------
        // MSVC (16.5.5) + CUDA (10.2) did not generate the defaulted functions
        // correct and errors out during compilation. Same for the other places where
        // I changed this.

#ifdef FLARE_IMPL_WINDOWS_CUDA
                                                                                                                                FLARE_FUNCTION TensorOffset() : m_dim(dimension_type()) {}
  FLARE_FUNCTION TensorOffset(const TensorOffset& src) { m_dim = src.m_dim; }
  FLARE_FUNCTION TensorOffset& operator=(const TensorOffset& src) {
    m_dim = src.m_dim;
    return *this;
  }
#else

        TensorOffset() = default;

        TensorOffset(const TensorOffset &) = default;

        TensorOffset &operator=(const TensorOffset &) = default;

#endif

        template<unsigned TrivialScalarSize>
        FLARE_INLINE_FUNCTION constexpr TensorOffset(
                std::integral_constant<unsigned, TrivialScalarSize> const &,
                flare::LayoutRight const &arg_layout)
                : m_dim(arg_layout.dimension[0], 0, 0, 0, 0, 0, 0, 0) {}

        template<class DimRHS>
        FLARE_INLINE_FUNCTION constexpr TensorOffset(
                const TensorOffset<DimRHS, flare::LayoutRight, void> &rhs)
                : m_dim(rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3,
                        rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7) {
            static_assert(int(DimRHS::rank) == int(dimension_type::rank),
                          "TensorOffset assignment requires equal rank");
            // Also requires equal static dimensions ...
        }

        template<class DimRHS>
        FLARE_INLINE_FUNCTION constexpr TensorOffset(
                const TensorOffset<DimRHS, flare::LayoutLeft, void> &rhs)
                : m_dim(rhs.m_dim.N0, 0, 0, 0, 0, 0, 0, 0) {
            static_assert((DimRHS::rank == 0 && dimension_type::rank == 0) ||
                          (DimRHS::rank == 1 && dimension_type::rank == 1),
                          "TensorOffset LayoutRight and LayoutLeft are only compatible "
                          "when rank <= 1");
        }

        template<class DimRHS>
        FLARE_INLINE_FUNCTION TensorOffset(
                const TensorOffset<DimRHS, flare::LayoutStride, void> &rhs)
                : m_dim(rhs.m_dim.N0, 0, 0, 0, 0, 0, 0, 0) {}

        //----------------------------------------
        // Subtensor construction

        template<class DimRHS>
        FLARE_INLINE_FUNCTION constexpr TensorOffset(
                const TensorOffset<DimRHS, flare::LayoutRight, void> &,
                const SubTensorExtents<DimRHS::rank, dimension_type::rank> &sub)
                : m_dim(sub.range_extent(0), 0, 0, 0, 0, 0, 0, 0) {
            static_assert((0 == dimension_type::rank_dynamic) ||
                          (1 == dimension_type::rank &&
                           1 == dimension_type::rank_dynamic && 1 <= DimRHS::rank),
                          "TensorOffset subtensor construction requires compatible rank");
        }
    };

//----------------------------------------------------------------------------
// LayoutRight AND ( 1 < rank AND 0 < rank_dynamic ) : has padding / striding
    template<class Dimension>
    struct TensorOffset<
            Dimension, flare::LayoutRight,
            std::enable_if_t<(1 < Dimension::rank && 0 < Dimension::rank_dynamic)>> {
        using is_mapping_plugin = std::true_type;
        using is_regular = std::true_type;

        using size_type = size_t;
        using dimension_type = Dimension;
        using array_layout = flare::LayoutRight;

        dimension_type m_dim;
        size_type m_stride;

        //----------------------------------------

        // rank 1
        template<typename I0>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0) const {
            return i0;
        }

        // rank 2
        template<typename I0, typename I1>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1) const {
            return i1 + i0 * m_stride;
        }

        // rank 3
        template<typename I0, typename I1, typename I2>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1,
                                                             I2 const &i2) const {
            return i2 + m_dim.N2 * (i1) + i0 * m_stride;
        }

        // rank 4
        template<typename I0, typename I1, typename I2, typename I3>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1,
                                                             I2 const &i2,
                                                             I3 const &i3) const {
            return i3 + m_dim.N3 * (i2 + m_dim.N2 * (i1)) + i0 * m_stride;
        }

        // rank 5
        template<typename I0, typename I1, typename I2, typename I3, typename I4>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1,
                                                             I2 const &i2,
                                                             I3 const &i3,
                                                             I4 const &i4) const {
            return i4 + m_dim.N4 * (i3 + m_dim.N3 * (i2 + m_dim.N2 * (i1))) +
                   i0 * m_stride;
        }

        // rank 6
        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(
                I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
                I5 const &i5) const {
            return i5 +
                   m_dim.N5 *
                   (i4 + m_dim.N4 * (i3 + m_dim.N3 * (i2 + m_dim.N2 * (i1)))) +
                   i0 * m_stride;
        }

        // rank 7
        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5, typename I6>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(
                I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
                I5 const &i5, I6 const &i6) const {
            return i6 +
                   m_dim.N6 *
                   (i5 + m_dim.N5 *
                         (i4 + m_dim.N4 *
                               (i3 + m_dim.N3 * (i2 + m_dim.N2 * (i1))))) +
                   i0 * m_stride;
        }

        // rank 8
        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5, typename I6, typename I7>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(
                I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
                I5 const &i5, I6 const &i6, I7 const &i7) const {
            return i7 +
                   m_dim.N7 *
                   (i6 +
                    m_dim.N6 *
                    (i5 +
                     m_dim.N5 *
                     (i4 + m_dim.N4 *
                           (i3 + m_dim.N3 * (i2 + m_dim.N2 * (i1)))))) +
                   i0 * m_stride;
        }

        //----------------------------------------

        FLARE_INLINE_FUNCTION
        constexpr array_layout layout() const {
            constexpr auto r = dimension_type::rank;
            return array_layout((r > 0 ? m_dim.N0 : FLARE_INVALID_INDEX),
                                (r > 1 ? m_dim.N1 : FLARE_INVALID_INDEX),
                                (r > 2 ? m_dim.N2 : FLARE_INVALID_INDEX),
                                (r > 3 ? m_dim.N3 : FLARE_INVALID_INDEX),
                                (r > 4 ? m_dim.N4 : FLARE_INVALID_INDEX),
                                (r > 5 ? m_dim.N5 : FLARE_INVALID_INDEX),
                                (r > 6 ? m_dim.N6 : FLARE_INVALID_INDEX),
                                (r > 7 ? m_dim.N7 : FLARE_INVALID_INDEX));
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_0() const {
            return m_dim.N0;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_1() const {
            return m_dim.N1;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_2() const {
            return m_dim.N2;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_3() const {
            return m_dim.N3;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_4() const {
            return m_dim.N4;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_5() const {
            return m_dim.N5;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_6() const {
            return m_dim.N6;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_7() const {
            return m_dim.N7;
        }

        /* Cardinality of the domain index space */
        FLARE_INLINE_FUNCTION
        constexpr size_type size() const {
            return size_type(m_dim.N0) * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 *
                   m_dim.N5 * m_dim.N6 * m_dim.N7;
        }

        /* Span of the range space */
        FLARE_INLINE_FUNCTION
        constexpr size_type span() const {
            return size() > 0 ? size_type(m_dim.N0) * m_stride : 0;
        }

        FLARE_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
            return m_stride == m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3 *
                               m_dim.N2 * m_dim.N1;
        }

        /* Strides of dimensions */
        FLARE_INLINE_FUNCTION constexpr size_type stride_7() const { return 1; }

        FLARE_INLINE_FUNCTION constexpr size_type stride_6() const {
            return m_dim.N7;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_5() const {
            return m_dim.N7 * m_dim.N6;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_4() const {
            return m_dim.N7 * m_dim.N6 * m_dim.N5;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_3() const {
            return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_2() const {
            return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_1() const {
            return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3 * m_dim.N2;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_0() const {
            return m_stride;
        }

        // Stride with [ rank ] value is the total length
        template<typename iType>
        FLARE_INLINE_FUNCTION void stride(iType *const s) const {
            size_type n = 1;
            if (7 < dimension_type::rank) {
                s[7] = n;
                n *= m_dim.N7;
            }
            if (6 < dimension_type::rank) {
                s[6] = n;
                n *= m_dim.N6;
            }
            if (5 < dimension_type::rank) {
                s[5] = n;
                n *= m_dim.N5;
            }
            if (4 < dimension_type::rank) {
                s[4] = n;
                n *= m_dim.N4;
            }
            if (3 < dimension_type::rank) {
                s[3] = n;
                n *= m_dim.N3;
            }
            if (2 < dimension_type::rank) {
                s[2] = n;
                n *= m_dim.N2;
            }
            if (1 < dimension_type::rank) {
                s[1] = n;
            }
            if (0 < dimension_type::rank) {
                s[0] = m_stride;
            }
            s[dimension_type::rank] = m_stride * m_dim.N0;
        }

        //----------------------------------------

    private:
        template<unsigned TrivialScalarSize>
        struct Padding {
            enum {
                div = TrivialScalarSize == 0
                      ? 0
                      : flare::detail::MEMORY_ALIGNMENT /
                        (TrivialScalarSize ? TrivialScalarSize : 1)
            };
            enum {
                mod = TrivialScalarSize == 0
                      ? 0
                      : flare::detail::MEMORY_ALIGNMENT %
                        (TrivialScalarSize ? TrivialScalarSize : 1)
            };

            // If memory alignment is a multiple of the trivial scalar size then attempt
            // to align.
            enum {
                align = 0 != TrivialScalarSize && 0 == mod ? div : 0
            };
            enum {
                div_ok = (div != 0) ? div : 1
            };  // To valid modulo zero in constexpr

            FLARE_INLINE_FUNCTION
            static constexpr size_t stride(size_t const N) {
                return ((align != 0) &&
                        ((static_cast<int>(flare::detail::MEMORY_ALIGNMENT_THRESHOLD) *
                          static_cast<int>(align)) < N) &&
                        ((N % div_ok) != 0))
                       ? N + align - (N % div_ok)
                       : N;
            }
        };

    public:
        // MSVC (16.5.5) + CUDA (10.2) did not generate the defaulted functions
        // correct and errors out during compilation. Same for the other places where
        // I changed this.

#ifdef FLARE_IMPL_WINDOWS_CUDA
                                                                                                                                FLARE_FUNCTION TensorOffset() : m_dim(dimension_type()), m_stride(0) {}
  FLARE_FUNCTION TensorOffset(const TensorOffset& src) {
    m_dim    = src.m_dim;
    m_stride = src.m_stride;
  }
  FLARE_FUNCTION TensorOffset& operator=(const TensorOffset& src) {
    m_dim    = src.m_dim;
    m_stride = src.m_stride;
    return *this;
  }
#else

        TensorOffset() = default;

        TensorOffset(const TensorOffset &) = default;

        TensorOffset &operator=(const TensorOffset &) = default;

#endif

        /* Enable padding for trivial scalar types with non-zero trivial scalar size.
   */
        template<unsigned TrivialScalarSize>
        FLARE_INLINE_FUNCTION constexpr TensorOffset(
                std::integral_constant<unsigned, TrivialScalarSize> const &,
                flare::LayoutRight const &arg_layout)
                : m_dim(arg_layout.dimension[0], arg_layout.dimension[1],
                        arg_layout.dimension[2], arg_layout.dimension[3],
                        arg_layout.dimension[4], arg_layout.dimension[5],
                        arg_layout.dimension[6], arg_layout.dimension[7]),
                  m_stride(
                          Padding<TrivialScalarSize>::
                          stride(/* 2 <= rank */
                                  m_dim.N1 *
                                  (dimension_type::rank == 2
                                   ? size_t(1)
                                   : m_dim.N2 *
                                     (dimension_type::rank == 3
                                      ? size_t(1)
                                      : m_dim.N3 *
                                        (dimension_type::rank == 4
                                         ? size_t(1)
                                         : m_dim.N4 *
                                           (dimension_type::rank ==
                                            5
                                            ? size_t(1)
                                            : m_dim.N5 *
                                              (dimension_type::
                                               rank ==
                                               6
                                               ? size_t(
                                                              1)
                                               : m_dim.N6 *
                                                 (dimension_type::
                                                  rank ==
                                                  7
                                                  ? size_t(
                                                                 1)
                                                  : m_dim
                                                          .N7)))))))) {
        }

        template<class DimRHS>
        FLARE_INLINE_FUNCTION constexpr TensorOffset(
                const TensorOffset<DimRHS, flare::LayoutRight, void> &rhs)
                : m_dim(rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3,
                        rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7),
                  m_stride(rhs.stride_0()) {
            static_assert(int(DimRHS::rank) == int(dimension_type::rank),
                          "TensorOffset assignment requires equal rank");
            // Also requires equal static dimensions ...
        }

        template<class DimRHS>
        FLARE_INLINE_FUNCTION TensorOffset(
                const TensorOffset<DimRHS, flare::LayoutStride, void> &rhs)
                : m_dim(rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3,
                        rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7),
                  m_stride(rhs.stride_0()) {
            if (((dimension_type::rank == 2)
                 ? rhs.m_stride.S1
                 : ((dimension_type::rank == 3)
                    ? rhs.m_stride.S2
                    : ((dimension_type::rank == 4)
                       ? rhs.m_stride.S3
                       : ((dimension_type::rank == 5)
                          ? rhs.m_stride.S4
                          : ((dimension_type::rank == 6)
                             ? rhs.m_stride.S5
                             : ((dimension_type::rank == 7)
                                ? rhs.m_stride.S6
                                : rhs.m_stride.S7)))))) != 1) {
                flare::abort(
                        "flare::detail::TensorOffset assignment of LayoutRight from "
                        "LayoutStride requires right-most stride == 1");
            }
        }

        //----------------------------------------
        // Subtensor construction
        // Last dimension must be non-zero

        template<class DimRHS>
        FLARE_INLINE_FUNCTION constexpr TensorOffset(
                const TensorOffset<DimRHS, flare::LayoutRight, void> &rhs,
                const SubTensorExtents<DimRHS::rank, dimension_type::rank> &sub)
                : m_dim(sub.range_extent(0), sub.range_extent(1), sub.range_extent(2),
                        sub.range_extent(3), sub.range_extent(4), sub.range_extent(5),
                        sub.range_extent(6), sub.range_extent(7)),
                  m_stride(
                          0 == sub.range_index(0)
                          ? rhs.stride_0()
                          : (1 == sub.range_index(0)
                             ? rhs.stride_1()
                             : (2 == sub.range_index(0)
                                ? rhs.stride_2()
                                : (3 == sub.range_index(0)
                                   ? rhs.stride_3()
                                   : (4 == sub.range_index(0)
                                      ? rhs.stride_4()
                                      : (5 == sub.range_index(0)
                                         ? rhs.stride_5()
                                         : (6 == sub.range_index(0)
                                            ? rhs.stride_6()
                                            : 0))))))) {
            /*      // This subtensor must be 2 == rank and 2 == rank_dynamic
          // due to only having stride #0.
          // The source dimension #0 must be non-zero for stride-one leading
       dimension.
          // At most subsequent dimension can be non-zero.

          static_assert( (( 2 == dimension_type::rank ) &&
                          ( 2 <= DimRHS::rank )) ||
                         ()
                       , "TensorOffset subtensor construction requires compatible
       rank" );
    */
        }
    };

//----------------------------------------------------------------------------
/* Strided array layout only makes sense for 0 < rank */
/* rank = 0 included for DynRankTensor case */

    template<unsigned Rank>
    struct TensorStride;

    template<>
    struct TensorStride<0> {
        static constexpr size_t S0 = 0, S1 = 0, S2 = 0, S3 = 0, S4 = 0, S5 = 0,
                S6 = 0, S7 = 0;

        TensorStride() = default;

        TensorStride(const TensorStride &) = default;

        TensorStride &operator=(const TensorStride &) = default;

        FLARE_INLINE_FUNCTION
        constexpr TensorStride(size_t, size_t, size_t, size_t, size_t, size_t, size_t,
                             size_t) {}
    };

    template<>
    struct TensorStride<1> {
        size_t S0;
        static constexpr size_t S1 = 0, S2 = 0, S3 = 0, S4 = 0, S5 = 0, S6 = 0,
                S7 = 0;

        TensorStride() = default;

        TensorStride(const TensorStride &) = default;

        TensorStride &operator=(const TensorStride &) = default;

        FLARE_INLINE_FUNCTION
        constexpr TensorStride(size_t aS0, size_t, size_t, size_t, size_t, size_t,
                             size_t, size_t)
                : S0(aS0) {}
    };

    template<>
    struct TensorStride<2> {
        size_t S0, S1;
        static constexpr size_t S2 = 0, S3 = 0, S4 = 0, S5 = 0, S6 = 0, S7 = 0;

        TensorStride() = default;

        TensorStride(const TensorStride &) = default;

        TensorStride &operator=(const TensorStride &) = default;

        FLARE_INLINE_FUNCTION
        constexpr TensorStride(size_t aS0, size_t aS1, size_t, size_t, size_t, size_t,
                             size_t, size_t)
                : S0(aS0), S1(aS1) {}
    };

    template<>
    struct TensorStride<3> {
        size_t S0, S1, S2;
        static constexpr size_t S3 = 0, S4 = 0, S5 = 0, S6 = 0, S7 = 0;

        TensorStride() = default;

        TensorStride(const TensorStride &) = default;

        TensorStride &operator=(const TensorStride &) = default;

        FLARE_INLINE_FUNCTION
        constexpr TensorStride(size_t aS0, size_t aS1, size_t aS2, size_t, size_t,
                             size_t, size_t, size_t)
                : S0(aS0), S1(aS1), S2(aS2) {}
    };

    template<>
    struct TensorStride<4> {
        size_t S0, S1, S2, S3;
        static constexpr size_t S4 = 0, S5 = 0, S6 = 0, S7 = 0;

        TensorStride() = default;

        TensorStride(const TensorStride &) = default;

        TensorStride &operator=(const TensorStride &) = default;

        FLARE_INLINE_FUNCTION
        constexpr TensorStride(size_t aS0, size_t aS1, size_t aS2, size_t aS3, size_t,
                             size_t, size_t, size_t)
                : S0(aS0), S1(aS1), S2(aS2), S3(aS3) {}
    };

    template<>
    struct TensorStride<5> {
        size_t S0, S1, S2, S3, S4;
        static constexpr size_t S5 = 0, S6 = 0, S7 = 0;

        TensorStride() = default;

        TensorStride(const TensorStride &) = default;

        TensorStride &operator=(const TensorStride &) = default;

        FLARE_INLINE_FUNCTION
        constexpr TensorStride(size_t aS0, size_t aS1, size_t aS2, size_t aS3,
                             size_t aS4, size_t, size_t, size_t)
                : S0(aS0), S1(aS1), S2(aS2), S3(aS3), S4(aS4) {}
    };

    template<>
    struct TensorStride<6> {
        size_t S0, S1, S2, S3, S4, S5;
        static constexpr size_t S6 = 0, S7 = 0;

        TensorStride() = default;

        TensorStride(const TensorStride &) = default;

        TensorStride &operator=(const TensorStride &) = default;

        FLARE_INLINE_FUNCTION
        constexpr TensorStride(size_t aS0, size_t aS1, size_t aS2, size_t aS3,
                             size_t aS4, size_t aS5, size_t, size_t)
                : S0(aS0), S1(aS1), S2(aS2), S3(aS3), S4(aS4), S5(aS5) {}
    };

    template<>
    struct TensorStride<7> {
        size_t S0, S1, S2, S3, S4, S5, S6;
        static constexpr size_t S7 = 0;

        TensorStride() = default;

        TensorStride(const TensorStride &) = default;

        TensorStride &operator=(const TensorStride &) = default;

        FLARE_INLINE_FUNCTION
        constexpr TensorStride(size_t aS0, size_t aS1, size_t aS2, size_t aS3,
                             size_t aS4, size_t aS5, size_t aS6, size_t)
                : S0(aS0), S1(aS1), S2(aS2), S3(aS3), S4(aS4), S5(aS5), S6(aS6) {}
    };

    template<>
    struct TensorStride<8> {
        size_t S0, S1, S2, S3, S4, S5, S6, S7;

        TensorStride() = default;

        TensorStride(const TensorStride &) = default;

        TensorStride &operator=(const TensorStride &) = default;

        FLARE_INLINE_FUNCTION
        constexpr TensorStride(size_t aS0, size_t aS1, size_t aS2, size_t aS3,
                             size_t aS4, size_t aS5, size_t aS6, size_t aS7)
                : S0(aS0),
                  S1(aS1),
                  S2(aS2),
                  S3(aS3),
                  S4(aS4),
                  S5(aS5),
                  S6(aS6),
                  S7(aS7) {}
    };

    template<class Dimension>
    struct TensorOffset<Dimension, flare::LayoutStride, void> {
    private:
        using stride_type = TensorStride<Dimension::rank>;

    public:
        using is_mapping_plugin = std::true_type;
        using is_regular = std::true_type;

        using size_type = size_t;
        using dimension_type = Dimension;
        using array_layout = flare::LayoutStride;

        dimension_type m_dim;
        stride_type m_stride;

        //----------------------------------------

        // rank 1
        template<typename I0>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0) const {
            return i0 * m_stride.S0;
        }

        // rank 2
        template<typename I0, typename I1>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1) const {
            return i0 * m_stride.S0 + i1 * m_stride.S1;
        }

        // rank 3
        template<typename I0, typename I1, typename I2>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1,
                                                             I2 const &i2) const {
            return i0 * m_stride.S0 + i1 * m_stride.S1 + i2 * m_stride.S2;
        }

        // rank 4
        template<typename I0, typename I1, typename I2, typename I3>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1,
                                                             I2 const &i2,
                                                             I3 const &i3) const {
            return i0 * m_stride.S0 + i1 * m_stride.S1 + i2 * m_stride.S2 +
                   i3 * m_stride.S3;
        }

        // rank 5
        template<typename I0, typename I1, typename I2, typename I3, typename I4>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(I0 const &i0,
                                                             I1 const &i1,
                                                             I2 const &i2,
                                                             I3 const &i3,
                                                             I4 const &i4) const {
            return i0 * m_stride.S0 + i1 * m_stride.S1 + i2 * m_stride.S2 +
                   i3 * m_stride.S3 + i4 * m_stride.S4;
        }

        // rank 6
        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(
                I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
                I5 const &i5) const {
            return i0 * m_stride.S0 + i1 * m_stride.S1 + i2 * m_stride.S2 +
                   i3 * m_stride.S3 + i4 * m_stride.S4 + i5 * m_stride.S5;
        }

        // rank 7
        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5, typename I6>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(
                I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
                I5 const &i5, I6 const &i6) const {
            return i0 * m_stride.S0 + i1 * m_stride.S1 + i2 * m_stride.S2 +
                   i3 * m_stride.S3 + i4 * m_stride.S4 + i5 * m_stride.S5 +
                   i6 * m_stride.S6;
        }

        // rank 8
        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5, typename I6, typename I7>
        FLARE_INLINE_FUNCTION constexpr size_type operator()(
                I0 const &i0, I1 const &i1, I2 const &i2, I3 const &i3, I4 const &i4,
                I5 const &i5, I6 const &i6, I7 const &i7) const {
            return i0 * m_stride.S0 + i1 * m_stride.S1 + i2 * m_stride.S2 +
                   i3 * m_stride.S3 + i4 * m_stride.S4 + i5 * m_stride.S5 +
                   i6 * m_stride.S6 + i7 * m_stride.S7;
        }

        //----------------------------------------

        FLARE_INLINE_FUNCTION
        constexpr array_layout layout() const {
            constexpr auto r = dimension_type::rank;
            return array_layout((r > 0 ? m_dim.N0 : FLARE_INVALID_INDEX), m_stride.S0,
                                (r > 1 ? m_dim.N1 : FLARE_INVALID_INDEX), m_stride.S1,
                                (r > 2 ? m_dim.N2 : FLARE_INVALID_INDEX), m_stride.S2,
                                (r > 3 ? m_dim.N3 : FLARE_INVALID_INDEX), m_stride.S3,
                                (r > 4 ? m_dim.N4 : FLARE_INVALID_INDEX), m_stride.S4,
                                (r > 5 ? m_dim.N5 : FLARE_INVALID_INDEX), m_stride.S5,
                                (r > 6 ? m_dim.N6 : FLARE_INVALID_INDEX), m_stride.S6,
                                (r > 7 ? m_dim.N7 : FLARE_INVALID_INDEX), m_stride.S7);
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_0() const {
            return m_dim.N0;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_1() const {
            return m_dim.N1;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_2() const {
            return m_dim.N2;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_3() const {
            return m_dim.N3;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_4() const {
            return m_dim.N4;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_5() const {
            return m_dim.N5;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_6() const {
            return m_dim.N6;
        }

        FLARE_INLINE_FUNCTION constexpr size_type dimension_7() const {
            return m_dim.N7;
        }

        /* Cardinality of the domain index space */
        FLARE_INLINE_FUNCTION
        constexpr size_type size() const {
            return dimension_type::rank == 0
                   ? 1
                   : size_type(m_dim.N0) * m_dim.N1 * m_dim.N2 * m_dim.N3 *
                     m_dim.N4 * m_dim.N5 * m_dim.N6 * m_dim.N7;
        }

    private:
        FLARE_INLINE_FUNCTION
        static constexpr size_type Max(size_type lhs, size_type rhs) {
            return lhs < rhs ? rhs : lhs;
        }

    public:
        /* Span of the range space, largest stride * dimension */
        FLARE_INLINE_FUNCTION
        constexpr size_type span() const {
            return dimension_type::rank == 0
                   ? 1
                   : (size() == size_type(0)
                      ? size_type(0)
                      : Max(m_dim.N0 * m_stride.S0,
                            Max(m_dim.N1 * m_stride.S1,
                                Max(m_dim.N2 * m_stride.S2,
                                    Max(m_dim.N3 * m_stride.S3,
                                        Max(m_dim.N4 * m_stride.S4,
                                            Max(m_dim.N5 * m_stride.S5,
                                                Max(m_dim.N6 * m_stride.S6,
                                                    m_dim.N7 *
                                                    m_stride.S7))))))));
        }

        FLARE_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
            return span() == size();
        }

        /* Strides of dimensions */
        FLARE_INLINE_FUNCTION constexpr size_type stride_0() const {
            return m_stride.S0;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_1() const {
            return m_stride.S1;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_2() const {
            return m_stride.S2;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_3() const {
            return m_stride.S3;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_4() const {
            return m_stride.S4;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_5() const {
            return m_stride.S5;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_6() const {
            return m_stride.S6;
        }

        FLARE_INLINE_FUNCTION constexpr size_type stride_7() const {
            return m_stride.S7;
        }

        // Stride with [ rank ] value is the total length
        template<typename iType>
        FLARE_INLINE_FUNCTION void stride(iType *const s) const {
            if (0 < dimension_type::rank) {
                s[0] = m_stride.S0;
            }
            if (1 < dimension_type::rank) {
                s[1] = m_stride.S1;
            }
            if (2 < dimension_type::rank) {
                s[2] = m_stride.S2;
            }
            if (3 < dimension_type::rank) {
                s[3] = m_stride.S3;
            }
            if (4 < dimension_type::rank) {
                s[4] = m_stride.S4;
            }
            if (5 < dimension_type::rank) {
                s[5] = m_stride.S5;
            }
            if (6 < dimension_type::rank) {
                s[6] = m_stride.S6;
            }
            if (7 < dimension_type::rank) {
                s[7] = m_stride.S7;
            }
            s[dimension_type::rank] = span();
        }

        //----------------------------------------
        // MSVC (16.5.5) + CUDA (10.2) did not generate the defaulted functions
        // correct and errors out during compilation. Same for the other places where
        // I changed this.

#ifdef FLARE_IMPL_WINDOWS_CUDA
                                                                                                                                FLARE_FUNCTION TensorOffset()
      : m_dim(dimension_type()), m_stride(stride_type()) {}
  FLARE_FUNCTION TensorOffset(const TensorOffset& src) {
    m_dim    = src.m_dim;
    m_stride = src.m_stride;
  }
  FLARE_FUNCTION TensorOffset& operator=(const TensorOffset& src) {
    m_dim    = src.m_dim;
    m_stride = src.m_stride;
    return *this;
  }
#else

        TensorOffset() = default;

        TensorOffset(const TensorOffset &) = default;

        TensorOffset &operator=(const TensorOffset &) = default;

#endif

        FLARE_INLINE_FUNCTION
        constexpr TensorOffset(std::integral_constant<unsigned, 0> const &,
                             flare::LayoutStride const &rhs)
                : m_dim(rhs.dimension[0], rhs.dimension[1], rhs.dimension[2],
                        rhs.dimension[3], rhs.dimension[4], rhs.dimension[5],
                        rhs.dimension[6], rhs.dimension[7]),
                  m_stride(rhs.stride[0], rhs.stride[1], rhs.stride[2], rhs.stride[3],
                           rhs.stride[4], rhs.stride[5], rhs.stride[6], rhs.stride[7]) {}

        template<class DimRHS, class LayoutRHS>
        FLARE_INLINE_FUNCTION constexpr TensorOffset(
                const TensorOffset<DimRHS, LayoutRHS, void> &rhs)
                : m_dim(rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3,
                        rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7),
                  m_stride(rhs.stride_0(), rhs.stride_1(), rhs.stride_2(), rhs.stride_3(),
                           rhs.stride_4(), rhs.stride_5(), rhs.stride_6(),
                           rhs.stride_7()) {
            static_assert(int(DimRHS::rank) == int(dimension_type::rank),
                          "TensorOffset assignment requires equal rank");
            // Also requires equal static dimensions ...
        }

        //----------------------------------------
        // Subtensor construction

    private:
        template<class DimRHS, class LayoutRHS>
        FLARE_INLINE_FUNCTION static constexpr size_t stride(
                unsigned r, const TensorOffset<DimRHS, LayoutRHS, void> &rhs) {
            return r > 7
                   ? 0
                   : (r == 0
                      ? rhs.stride_0()
                      : (r == 1
                         ? rhs.stride_1()
                         : (r == 2
                            ? rhs.stride_2()
                            : (r == 3
                               ? rhs.stride_3()
                               : (r == 4
                                  ? rhs.stride_4()
                                  : (r == 5
                                     ? rhs.stride_5()
                                     : (r == 6
                                        ? rhs.stride_6()
                                        : rhs.stride_7())))))));
        }

    public:
        template<class DimRHS, class LayoutRHS>
        FLARE_INLINE_FUNCTION constexpr TensorOffset(
                const TensorOffset<DimRHS, LayoutRHS, void> &rhs,
                const SubTensorExtents<DimRHS::rank, dimension_type::rank> &sub)
        // range_extent(r) returns 0 when dimension_type::rank <= r
                : m_dim(sub.range_extent(0), sub.range_extent(1), sub.range_extent(2),
                        sub.range_extent(3), sub.range_extent(4), sub.range_extent(5),
                        sub.range_extent(6), sub.range_extent(7))
                // range_index(r) returns ~0u when dimension_type::rank <= r
                ,
                  m_stride(
                          stride(sub.range_index(0), rhs), stride(sub.range_index(1), rhs),
                          stride(sub.range_index(2), rhs), stride(sub.range_index(3), rhs),
                          stride(sub.range_index(4), rhs), stride(sub.range_index(5), rhs),
                          stride(sub.range_index(6), rhs), stride(sub.range_index(7), rhs)) {}
    };


/** \brief  TensorDataHandle provides the type of the 'data handle' which the tensor
 *          uses to access data with the [] operator. It also provides
 *          an allocate function and a function to extract a raw ptr from the
 *          data handle. TensorDataHandle also defines an enum ReferenceAble which
 *          specifies whether references/pointers to elements can be taken and a
 *          'return_type' which is what the tensor operators will give back.
 *          Specialisation of this object allows three things depending
 *          on TensorTraits and compiler options:
 *          (i)   Use special allocator (e.g. huge pages/small pages and pinned
 * memory) (ii)  Use special data handle type (e.g. add Cuda Texture Object)
 *          (iii) Use special access intrinsics (e.g. texture fetch and
 * non-caching loads)
 */
    template<class Traits, class Enable = void>
    struct TensorDataHandle {
        using value_type = typename Traits::value_type;
        using handle_type = typename Traits::value_type *;
        using return_type = typename Traits::value_type &;
        using track_type = flare::detail::SharedAllocationTracker;

        FLARE_INLINE_FUNCTION
        static handle_type assign(value_type *arg_data_ptr,
                                  track_type const & /*arg_tracker*/) {
            return handle_type(arg_data_ptr);
        }

        FLARE_INLINE_FUNCTION
        static handle_type assign(handle_type const arg_data_ptr, size_t offset) {
            return handle_type(arg_data_ptr + offset);
        }
    };

    template<class Traits>
    struct TensorDataHandle<
            Traits,
            std::enable_if_t<(std::is_same<typename Traits::non_const_value_type,
                    typename Traits::value_type>::value &&
                              std::is_void<typename Traits::specialize>::value &&
                              Traits::memory_traits::is_atomic)>> {
        using value_type = typename Traits::value_type;
        using handle_type = typename flare::detail::AtomicTensorDataHandle<Traits>;
        using return_type = typename flare::detail::AtomicDataElement<Traits>;
        using track_type = flare::detail::SharedAllocationTracker;

        FLARE_INLINE_FUNCTION
        static handle_type assign(value_type *arg_data_ptr,
                                  track_type const & /*arg_tracker*/) {
            return handle_type(arg_data_ptr);
        }

        template<class SrcHandleType>
        FLARE_INLINE_FUNCTION static handle_type assign(
                const SrcHandleType &arg_handle, size_t offset) {
            return handle_type(arg_handle + offset);
        }
    };

    template<class Traits>
    struct TensorDataHandle<
            Traits,
            std::enable_if_t<(std::is_void<typename Traits::specialize>::value &&
                              (!Traits::memory_traits::is_aligned) &&
                              Traits::memory_traits::is_restrict &&
                              (!Traits::memory_traits::is_atomic))>> {
        using value_type = typename Traits::value_type;
        using handle_type = typename Traits::value_type *FLARE_RESTRICT;
        using return_type = typename Traits::value_type &FLARE_RESTRICT;
        using track_type = flare::detail::SharedAllocationTracker;

        FLARE_INLINE_FUNCTION
        static value_type *assign(value_type *arg_data_ptr,
                                  track_type const & /*arg_tracker*/) {
            return (value_type *) (arg_data_ptr);
        }

        FLARE_INLINE_FUNCTION
        static value_type *assign(handle_type const arg_data_ptr, size_t offset) {
            return (value_type *) (arg_data_ptr + offset);
        }
    };

    template<class Traits>
    struct TensorDataHandle<
            Traits,
            std::enable_if_t<(std::is_void<typename Traits::specialize>::value &&
                              Traits::memory_traits::is_aligned &&
                              (!Traits::memory_traits::is_restrict) &&
                              (!Traits::memory_traits::is_atomic))>> {
        using value_type = typename Traits::value_type;
        // typedef work-around for intel compilers error #3186: expected typedef
        // declaration
        // NOLINTNEXTLINE(modernize-use-using)
        typedef value_type *FLARE_IMPL_ALIGN_PTR(FLARE_MEMORY_ALIGNMENT)
                handle_type;
        using return_type = typename Traits::value_type &;
        using track_type = flare::detail::SharedAllocationTracker;

        FLARE_INLINE_FUNCTION
        static handle_type assign(value_type *arg_data_ptr,
                                  track_type const & /*arg_tracker*/) {
            if (reinterpret_cast<uintptr_t>(arg_data_ptr) % detail::MEMORY_ALIGNMENT) {
                flare::abort(
                        "Assigning NonAligned Tensor or Pointer to flare::Tensor with Aligned "
                        "attribute");
            }
            return handle_type(arg_data_ptr);
        }

        FLARE_INLINE_FUNCTION
        static handle_type assign(handle_type const arg_data_ptr, size_t offset) {
            if (reinterpret_cast<uintptr_t>(arg_data_ptr + offset) %
                detail::MEMORY_ALIGNMENT) {
                flare::abort(
                        "Assigning NonAligned Tensor or Pointer to flare::Tensor with Aligned "
                        "attribute");
            }
            return handle_type(arg_data_ptr + offset);
        }
    };

    template<class Traits>
    struct TensorDataHandle<
            Traits,
            std::enable_if_t<(std::is_void<typename Traits::specialize>::value &&
                              Traits::memory_traits::is_aligned &&
                              Traits::memory_traits::is_restrict &&
                              (!Traits::memory_traits::is_atomic))>> {
        using value_type = typename Traits::value_type;
        // typedef work-around for intel compilers error #3186: expected typedef
        // declaration
        // NOLINTNEXTLINE(modernize-use-using)
        typedef value_type *FLARE_IMPL_ALIGN_PTR(FLARE_MEMORY_ALIGNMENT)
                handle_type;
        using return_type = typename Traits::value_type &FLARE_RESTRICT;
        using track_type = flare::detail::SharedAllocationTracker;

        FLARE_INLINE_FUNCTION
        static value_type *assign(value_type *arg_data_ptr,
                                  track_type const & /*arg_tracker*/) {
            if (reinterpret_cast<uintptr_t>(arg_data_ptr) % detail::MEMORY_ALIGNMENT) {
                flare::abort(
                        "Assigning NonAligned Tensor or Pointer to flare::Tensor with Aligned "
                        "attribute");
            }
            return (value_type *) (arg_data_ptr);
        }

        FLARE_INLINE_FUNCTION
        static value_type *assign(handle_type const arg_data_ptr, size_t offset) {
            if (reinterpret_cast<uintptr_t>(arg_data_ptr + offset) %
                detail::MEMORY_ALIGNMENT) {
                flare::abort(
                        "Assigning NonAligned Tensor or Pointer to flare::Tensor with Aligned "
                        "attribute");
            }
            return (value_type *) (arg_data_ptr + offset);
        }
    };


    template<typename T>
    inline bool is_zero_byte(const T &t) {
        using comparison_type = std::conditional_t<
                sizeof(T) % sizeof(long long int) == 0, long long int,
                std::conditional_t<
                        sizeof(T) % sizeof(long int) == 0, long int,
                        std::conditional_t<
                                sizeof(T) % sizeof(int) == 0, int,
                                std::conditional_t<sizeof(T) % sizeof(short int) == 0, short int,
                                        char>>>>;
        const auto *const ptr = reinterpret_cast<const comparison_type *>(&t);
        for (std::size_t i = 0; i < sizeof(T) / sizeof(comparison_type); ++i)
            if (ptr[i] != 0) return false;
        return true;
    }

//----------------------------------------------------------------------------

/*
 *  The construction, assignment to default, and destruction
 *  are merged into a single functor.
 *  Primarily to work around an unresolved CUDA back-end bug
 *  that would lose the destruction cuda device function when
 *  called from the shared memory tracking destruction.
 *  Secondarily to have two fewer partial specializations.
 */
    template<class DeviceType, class ValueType,
            bool IsScalar = std::is_scalar<ValueType>::value>
    struct TensorValueFunctor;

    template<class DeviceType, class ValueType>
    struct TensorValueFunctor<DeviceType, ValueType, false /* is_scalar */> {
        using ExecSpace = typename DeviceType::execution_space;

        struct DestroyTag {
        };
        struct ConstructTag {
        };

        ExecSpace space;
        ValueType *ptr;
        size_t n;
        std::string name;
        bool default_exec_space;

        template<class _ValueType = ValueType>
        FLARE_INLINE_FUNCTION
        std::enable_if_t<std::is_default_constructible<_ValueType>::value>
        operator()(ConstructTag const &, const size_t i) const {
            new(ptr + i) ValueType();
        }

        FLARE_INLINE_FUNCTION void operator()(DestroyTag const &,
                                              const size_t i) const {
            (ptr + i)->~ValueType();
        }

        TensorValueFunctor() = default;

        TensorValueFunctor(const TensorValueFunctor &) = default;

        TensorValueFunctor &operator=(const TensorValueFunctor &) = default;

        TensorValueFunctor(ExecSpace const &arg_space, ValueType *const arg_ptr,
                         size_t const arg_n, std::string arg_name)
                : space(arg_space),
                  ptr(arg_ptr),
                  n(arg_n),
                  name(std::move(arg_name)),
                  default_exec_space(false) {
            functor_instantiate_workaround();
        }

        TensorValueFunctor(ValueType *const arg_ptr, size_t const arg_n,
                         std::string arg_name)
                : space(ExecSpace{}),
                  ptr(arg_ptr),
                  n(arg_n),
                  name(std::move(arg_name)),
                  default_exec_space(true) {
            functor_instantiate_workaround();
        }

        template<typename Dummy = ValueType>
        std::enable_if_t<std::is_trivial<Dummy>::value &&
                         std::is_trivially_copy_assignable<ValueType>::value>
        construct_dispatch() {
            ValueType value{};
// On A64FX memset seems to do the wrong thing with regards to first touch
// leading to the significant performance issues
#ifndef FLARE_ARCH_A64FX
            if (detail::is_zero_byte(value)) {
                uint64_t kpID = 0;
                if (flare::Profiling::profileLibraryLoaded()) {
                    // We are not really using parallel_for here but using beginParallelFor
                    // instead of begin_parallel_for (and adding "via memset") is the best
                    // we can do to indicate that this is not supposed to be tunable (and
                    // doesn't really execute a parallel_for).
                    flare::Profiling::beginParallelFor(
                            "flare::Tensor::initialization [" + name + "] via memset",
                            flare::Profiling::experimental::device_id(space), &kpID);
                }
                (void) ZeroMemset<
                        ExecSpace, flare::Tensor<ValueType *, typename DeviceType::memory_space,
                                flare::MemoryTraits<flare::Unmanaged>>>(
                        space,
                        flare::Tensor<ValueType *, typename DeviceType::memory_space,
                                flare::MemoryTraits<flare::Unmanaged>>(ptr, n),
                        value);

                if (flare::Profiling::profileLibraryLoaded()) {
                    flare::Profiling::endParallelFor(kpID);
                }
                if (default_exec_space)
                    space.fence("flare::detail::TensorValueFunctor: Tensor init/destroy fence");
            } else {
#endif
                parallel_for_implementation<ConstructTag>();
#ifndef FLARE_ARCH_A64FX
            }
#endif
        }

        template<typename Dummy = ValueType>
        std::enable_if_t<!(std::is_trivial<Dummy>::value &&
                           std::is_trivially_copy_assignable<ValueType>::value)>
        construct_dispatch() {
            parallel_for_implementation<ConstructTag>();
        }

        template<typename Tag>
        void parallel_for_implementation() {
            if (!space.in_parallel()) {
                using PolicyType =
                        flare::RangePolicy<ExecSpace, flare::IndexType<int64_t>, Tag>;
                PolicyType policy(space, 0, n);
                uint64_t kpID = 0;
                if (flare::Profiling::profileLibraryLoaded()) {
                    const std::string functor_name =
                            (std::is_same_v<Tag, DestroyTag>
                             ? "flare::Tensor::destruction [" + name + "]"
                             : "flare::Tensor::initialization [" + name + "]");
                    flare::Profiling::beginParallelFor(
                            functor_name, flare::Profiling::experimental::device_id(space),
                            &kpID);
                }

#ifdef FLARE_ON_CUDA_DEVICE
                                                                                                                                        if (std::is_same<ExecSpace, flare::Cuda>::value) {
        flare::detail::cuda_prefetch_pointer(space, ptr, sizeof(ValueType) * n,
                                            true);
      }
#endif
                const flare::detail::ParallelFor<TensorValueFunctor, PolicyType> closure(
                        *this, policy);
                closure.execute();
                if (default_exec_space || std::is_same_v<Tag, DestroyTag>)
                    space.fence("flare::detail::TensorValueFunctor: Tensor init/destroy fence");
                if (flare::Profiling::profileLibraryLoaded()) {
                    flare::Profiling::endParallelFor(kpID);
                }
            } else {
                for (size_t i = 0; i < n; ++i) operator()(Tag{}, i);
            }
        }

        void construct_shared_allocation() { construct_dispatch(); }

        void destroy_shared_allocation() {
            parallel_for_implementation<DestroyTag>();
        }

        // This function is to ensure that the functor with DestroyTag is instantiated
        // This is a workaround to avoid "cudaErrorInvalidDeviceFunction" error later
        // when the function is queried with cudaFuncGetAttributes
        void functor_instantiate_workaround() {
#if defined(FLARE_ON_CUDA_DEVICE)
                                                                                                                                    if (false) {
      parallel_for_implementation<DestroyTag>();
    }
#endif
        }
    };

    template<class DeviceType, class ValueType>
    struct TensorValueFunctor<DeviceType, ValueType, true /* is_scalar */> {
        using ExecSpace = typename DeviceType::execution_space;
        using PolicyType = flare::RangePolicy<ExecSpace, flare::IndexType<int64_t>>;

        ExecSpace space;
        ValueType *ptr;
        size_t n;
        std::string name;
        bool default_exec_space;

        FLARE_INLINE_FUNCTION
        void operator()(const size_t i) const { ptr[i] = ValueType(); }

        TensorValueFunctor() = default;

        TensorValueFunctor(const TensorValueFunctor &) = default;

        TensorValueFunctor &operator=(const TensorValueFunctor &) = default;

        TensorValueFunctor(ExecSpace const &arg_space, ValueType *const arg_ptr,
                         size_t const arg_n, std::string arg_name)
                : space(arg_space),
                  ptr(arg_ptr),
                  n(arg_n),
                  name(std::move(arg_name)),
                  default_exec_space(false) {}

        TensorValueFunctor(ValueType *const arg_ptr, size_t const arg_n,
                         std::string arg_name)
                : space(ExecSpace{}),
                  ptr(arg_ptr),
                  n(arg_n),
                  name(std::move(arg_name)),
                  default_exec_space(true) {}

        template<typename Dummy = ValueType>
        std::enable_if_t<std::is_trivial<Dummy>::value &&
                         std::is_trivially_copy_assignable<Dummy>::value>
        construct_shared_allocation() {
            // Shortcut for zero initialization
// On A64FX memset seems to do the wrong thing with regards to first touch
// leading to the significant performance issues
#ifndef FLARE_ARCH_A64FX
            ValueType value{};
            if (detail::is_zero_byte(value)) {
                uint64_t kpID = 0;
                if (flare::Profiling::profileLibraryLoaded()) {
                    // We are not really using parallel_for here but using beginParallelFor
                    // instead of begin_parallel_for (and adding "via memset") is the best
                    // we can do to indicate that this is not supposed to be tunable (and
                    // doesn't really execute a parallel_for).
                    flare::Profiling::beginParallelFor(
                            "flare::Tensor::initialization [" + name + "] via memset",
                            flare::Profiling::experimental::device_id(space), &kpID);
                }

                (void) ZeroMemset<
                        ExecSpace, flare::Tensor<ValueType *, typename DeviceType::memory_space,
                                flare::MemoryTraits<flare::Unmanaged>>>(
                        space,
                        flare::Tensor<ValueType *, typename DeviceType::memory_space,
                                flare::MemoryTraits<flare::Unmanaged>>(ptr, n),
                        value);

                if (flare::Profiling::profileLibraryLoaded()) {
                    flare::Profiling::endParallelFor(kpID);
                }
                if (default_exec_space)
                    space.fence("flare::detail::TensorValueFunctor: Tensor init/destroy fence");
            } else {
#endif
                parallel_for_implementation();
#ifndef FLARE_ARCH_A64FX
            }
#endif
        }

        template<typename Dummy = ValueType>
        std::enable_if_t<!(std::is_trivial<Dummy>::value &&
                           std::is_trivially_copy_assignable<Dummy>::value)>
        construct_shared_allocation() {
            parallel_for_implementation();
        }

        void parallel_for_implementation() {
            if (!space.in_parallel()) {
                PolicyType policy(0, n);
                uint64_t kpID = 0;
                if (flare::Profiling::profileLibraryLoaded()) {
                    flare::Profiling::beginParallelFor(
                            "flare::Tensor::initialization [" + name + "]",
                            flare::Profiling::experimental::device_id(space), &kpID);
                }
#ifdef FLARE_ON_CUDA_DEVICE
                                                                                                                                        if (std::is_same<ExecSpace, flare::Cuda>::value) {
        flare::detail::cuda_prefetch_pointer(space, ptr, sizeof(ValueType) * n,
                                            true);
      }
#endif
                const flare::detail::ParallelFor<TensorValueFunctor, PolicyType> closure(
                        *this, PolicyType(0, n));
                closure.execute();
                if (default_exec_space)
                    space.fence(
                            "flare::detail::TensorValueFunctor: Fence after setting values in tensor");
                if (flare::Profiling::profileLibraryLoaded()) {
                    flare::Profiling::endParallelFor(kpID);
                }
            } else {
                for (size_t i = 0; i < n; ++i) operator()(i);
            }
        }

        void destroy_shared_allocation() {}
    };

    //----------------------------------------------------------------------------
    /** \brief  Tensor mapping for non-specialized data type and standard layout */
    template<class Traits>
    class TensorMapping<
            Traits,
            std::enable_if_t<(
                    std::is_void<typename Traits::specialize>::value &&
                    TensorOffset<typename Traits::dimension, typename Traits::array_layout,
                            void>::is_mapping_plugin::value)>> {
    public:
        using offset_type = TensorOffset<typename Traits::dimension,
                typename Traits::array_layout, void>;

        using handle_type = typename TensorDataHandle<Traits>::handle_type;

        handle_type m_impl_handle;
        offset_type m_impl_offset;

    private:
        template<class, class...>
        friend
        class TensorMapping;

        FLARE_INLINE_FUNCTION
        TensorMapping(const handle_type &arg_handle, const offset_type &arg_offset)
                : m_impl_handle(arg_handle), m_impl_offset(arg_offset) {}

    public:
        using printable_label_typedef = void;
        enum {
            is_managed = Traits::is_managed
        };

        //----------------------------------------
        // Domain dimensions

        static constexpr unsigned Rank = Traits::dimension::rank;

        template<typename iType>
        FLARE_INLINE_FUNCTION constexpr size_t extent(const iType &r) const {
            return m_impl_offset.m_dim.extent(r);
        }

        static FLARE_INLINE_FUNCTION constexpr size_t static_extent(
                const unsigned r) noexcept {
            using dim_type = typename offset_type::dimension_type;
            return dim_type::static_extent(r);
        }

        FLARE_INLINE_FUNCTION constexpr typename Traits::array_layout layout()
        const {
            return m_impl_offset.layout();
        }

        FLARE_INLINE_FUNCTION constexpr size_t dimension_0() const {
            return m_impl_offset.dimension_0();
        }

        FLARE_INLINE_FUNCTION constexpr size_t dimension_1() const {
            return m_impl_offset.dimension_1();
        }

        FLARE_INLINE_FUNCTION constexpr size_t dimension_2() const {
            return m_impl_offset.dimension_2();
        }

        FLARE_INLINE_FUNCTION constexpr size_t dimension_3() const {
            return m_impl_offset.dimension_3();
        }

        FLARE_INLINE_FUNCTION constexpr size_t dimension_4() const {
            return m_impl_offset.dimension_4();
        }

        FLARE_INLINE_FUNCTION constexpr size_t dimension_5() const {
            return m_impl_offset.dimension_5();
        }

        FLARE_INLINE_FUNCTION constexpr size_t dimension_6() const {
            return m_impl_offset.dimension_6();
        }

        FLARE_INLINE_FUNCTION constexpr size_t dimension_7() const {
            return m_impl_offset.dimension_7();
        }

        // Is a regular layout with uniform striding for each index.
        using is_regular = typename offset_type::is_regular;

        FLARE_INLINE_FUNCTION constexpr size_t stride_0() const {
            return m_impl_offset.stride_0();
        }

        FLARE_INLINE_FUNCTION constexpr size_t stride_1() const {
            return m_impl_offset.stride_1();
        }

        FLARE_INLINE_FUNCTION constexpr size_t stride_2() const {
            return m_impl_offset.stride_2();
        }

        FLARE_INLINE_FUNCTION constexpr size_t stride_3() const {
            return m_impl_offset.stride_3();
        }

        FLARE_INLINE_FUNCTION constexpr size_t stride_4() const {
            return m_impl_offset.stride_4();
        }

        FLARE_INLINE_FUNCTION constexpr size_t stride_5() const {
            return m_impl_offset.stride_5();
        }

        FLARE_INLINE_FUNCTION constexpr size_t stride_6() const {
            return m_impl_offset.stride_6();
        }

        FLARE_INLINE_FUNCTION constexpr size_t stride_7() const {
            return m_impl_offset.stride_7();
        }

        template<typename iType>
        FLARE_INLINE_FUNCTION void stride(iType *const s) const {
            m_impl_offset.stride(s);
        }

        //----------------------------------------
        // Range span

        /** \brief  Span of the mapped range */
        FLARE_INLINE_FUNCTION constexpr size_t span() const {
            return m_impl_offset.span();
        }

        /** \brief  Is the mapped range span contiguous */
        FLARE_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
            return m_impl_offset.span_is_contiguous();
        }

        using reference_type = typename TensorDataHandle<Traits>::return_type;
        using pointer_type = typename Traits::value_type *;

        /** \brief  Query raw pointer to memory */
        FLARE_INLINE_FUNCTION constexpr pointer_type data() const {
            return m_impl_handle;
        }

        //----------------------------------------
        // The Tensor class performs all rank and bounds checking before
        // calling these element reference methods.

        FLARE_FORCEINLINE_FUNCTION
        reference_type reference() const { return m_impl_handle[0]; }

        template<typename I0>
        FLARE_FORCEINLINE_FUNCTION
        std::enable_if_t<(std::is_integral<I0>::value &&
                          // if layout is neither stride nor irregular,
                          // then just use the handle directly
                          !(std::is_same<typename Traits::array_layout,
                                  flare::LayoutStride>::value ||
                            !is_regular::value)),
                reference_type>
        reference(const I0 &i0) const {
            return m_impl_handle[i0];
        }

        template<typename I0>
        FLARE_FORCEINLINE_FUNCTION
        std::enable_if_t<(std::is_integral<I0>::value &&
                          // if the layout is strided or irregular, then
                          // we have to use the offset
                          (std::is_same<typename Traits::array_layout,
                                  flare::LayoutStride>::value ||
                           !is_regular::value)),
                reference_type>
        reference(const I0 &i0) const {
            return m_impl_handle[m_impl_offset(i0)];
        }

        template<typename I0, typename I1>
        FLARE_FORCEINLINE_FUNCTION reference_type reference(const I0 &i0,
                                                            const I1 &i1) const {
            return m_impl_handle[m_impl_offset(i0, i1)];
        }

        template<typename I0, typename I1, typename I2>
        FLARE_FORCEINLINE_FUNCTION reference_type reference(const I0 &i0,
                                                            const I1 &i1,
                                                            const I2 &i2) const {
            return m_impl_handle[m_impl_offset(i0, i1, i2)];
        }

        template<typename I0, typename I1, typename I2, typename I3>
        FLARE_FORCEINLINE_FUNCTION reference_type
        reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3) const {
            return m_impl_handle[m_impl_offset(i0, i1, i2, i3)];
        }

        template<typename I0, typename I1, typename I2, typename I3, typename I4>
        FLARE_FORCEINLINE_FUNCTION reference_type reference(const I0 &i0,
                                                            const I1 &i1,
                                                            const I2 &i2,
                                                            const I3 &i3,
                                                            const I4 &i4) const {
            return m_impl_handle[m_impl_offset(i0, i1, i2, i3, i4)];
        }

        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5>
        FLARE_FORCEINLINE_FUNCTION reference_type
        reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
                  const I4 &i4, const I5 &i5) const {
            return m_impl_handle[m_impl_offset(i0, i1, i2, i3, i4, i5)];
        }

        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5, typename I6>
        FLARE_FORCEINLINE_FUNCTION reference_type
        reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
                  const I4 &i4, const I5 &i5, const I6 &i6) const {
            return m_impl_handle[m_impl_offset(i0, i1, i2, i3, i4, i5, i6)];
        }

        template<typename I0, typename I1, typename I2, typename I3, typename I4,
                typename I5, typename I6, typename I7>
        FLARE_FORCEINLINE_FUNCTION reference_type
        reference(const I0 &i0, const I1 &i1, const I2 &i2, const I3 &i3,
                  const I4 &i4, const I5 &i5, const I6 &i6, const I7 &i7) const {
            return m_impl_handle[m_impl_offset(i0, i1, i2, i3, i4, i5, i6, i7)];
        }

        //----------------------------------------

    private:
        enum {
            MemorySpanMask = 8 - 1 /* Force alignment on 8 byte boundary */ };
        enum {
            MemorySpanSize = sizeof(typename Traits::value_type)
        };

    public:
        /** \brief  Span, in bytes, of the referenced memory */
        FLARE_INLINE_FUNCTION constexpr size_t memory_span() const {
            return (m_impl_offset.span() * sizeof(typename Traits::value_type) +
                    MemorySpanMask) &
                   ~size_t(MemorySpanMask);
        }

        //----------------------------------------

        FLARE_DEFAULTED_FUNCTION ~TensorMapping() = default;

        FLARE_INLINE_FUNCTION TensorMapping() : m_impl_handle(), m_impl_offset() {}

        FLARE_DEFAULTED_FUNCTION TensorMapping(const TensorMapping &) = default;

        FLARE_DEFAULTED_FUNCTION TensorMapping &operator=(const TensorMapping &) =
        default;

        FLARE_DEFAULTED_FUNCTION TensorMapping(TensorMapping &&) = default;

        FLARE_DEFAULTED_FUNCTION TensorMapping &operator=(TensorMapping &&) = default;

        //----------------------------------------

        /**\brief  Span, in bytes, of the required memory */
        FLARE_INLINE_FUNCTION
        static constexpr size_t memory_span(
                typename Traits::array_layout const &arg_layout) {
            using padding = std::integral_constant<unsigned int, 0>;
            return (offset_type(padding(), arg_layout).span() * MemorySpanSize +
                    MemorySpanMask) &
                   ~size_t(MemorySpanMask);
        }

        /**\brief  Wrap a span of memory */
        template<class... P>
        FLARE_INLINE_FUNCTION TensorMapping(
                flare::detail::TensorCtorProp<P...> const &arg_prop,
                typename Traits::array_layout const &arg_layout)
                : m_impl_handle(detail::get_property<detail::PointerTag>(arg_prop)),
                  m_impl_offset(std::integral_constant<unsigned, 0>(), arg_layout) {}

        /**\brief  Assign data */
        FLARE_INLINE_FUNCTION
        void assign_data(pointer_type arg_ptr) {
            m_impl_handle = handle_type(arg_ptr);
        }

        //----------------------------------------
        /*  Allocate and construct mapped array.
   *  Allocate via shared allocation record and
   *  return that record for allocation tracking.
   */
        template<class... P>
        flare::detail::SharedAllocationRecord<> *allocate_shared(
                flare::detail::TensorCtorProp<P...> const &arg_prop,
                typename Traits::array_layout const &arg_layout,
                bool execution_space_specified) {
            using alloc_prop = flare::detail::TensorCtorProp<P...>;

            using execution_space = typename alloc_prop::execution_space;
            using memory_space = typename Traits::memory_space;
            static_assert(
                    SpaceAccessibility<execution_space, memory_space>::accessible);
            using value_type = typename Traits::value_type;
            using functor_type =
                    TensorValueFunctor<flare::Device<execution_space, memory_space>,
                            value_type>;
            using record_type =
                    flare::detail::SharedAllocationRecord<memory_space, functor_type>;

            // Query the mapping for byte-size of allocation.
            // If padding is allowed then pass in sizeof value type
            // for padding computation.
            using padding = std::integral_constant<
                    unsigned int, alloc_prop::allow_padding ? sizeof(value_type) : 0>;

            m_impl_offset = offset_type(padding(), arg_layout);

            const size_t alloc_size =
                    (m_impl_offset.span() * MemorySpanSize + MemorySpanMask) &
                    ~size_t(MemorySpanMask);
            const std::string &alloc_name =
                    detail::get_property<detail::LabelTag>(arg_prop);
            const execution_space &exec_space =
                    detail::get_property<detail::ExecutionSpaceTag>(arg_prop);
            const memory_space &mem_space =
                    detail::get_property<detail::MemorySpaceTag>(arg_prop);

            // Create shared memory tracking record with allocate memory from the memory
            // space
            record_type *const record =
                    execution_space_specified
                    ? record_type::allocate(exec_space, mem_space, alloc_name,
                                            alloc_size)
                    : record_type::allocate(mem_space, alloc_name, alloc_size);

            m_impl_handle = handle_type(reinterpret_cast<pointer_type>(record->data()));

            functor_type functor =
                    execution_space_specified
                    ? functor_type(exec_space, (value_type *) m_impl_handle,
                                   m_impl_offset.span(), alloc_name)
                    : functor_type((value_type *) m_impl_handle, m_impl_offset.span(),
                                   alloc_name);

            //  Only initialize if the allocation is non-zero.
            //  May be zero if one of the dimensions is zero.
            if constexpr (alloc_prop::initialize)
                if (alloc_size) {
                    // Assume destruction is only required when construction is requested.
                    // The TensorValueFunctor has both value construction and destruction
                    // operators.
                    record->m_destroy = std::move(functor);

                    // Construct values
                    record->m_destroy.construct_shared_allocation();
                }

            return record;
        }
    };

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/** \brief  Assign compatible default mappings */

    template<class DstTraits, class SrcTraits>
    class TensorMapping<
            DstTraits, SrcTraits,
            std::enable_if_t<(
                    !(std::is_same<typename SrcTraits::array_layout, LayoutStride>::
                    value) &&  // Added to have a new specialization for SrcType of
                    // LayoutStride
                    // default mappings
                    std::is_void<typename DstTraits::specialize>::value &&
                    std::is_void<typename SrcTraits::specialize>::value &&
                    (
                            // same layout
                            std::is_same<typename DstTraits::array_layout,
                                    typename SrcTraits::array_layout>::value ||
                            // known layout
                            ((std::is_same<typename DstTraits::array_layout,
                                    flare::LayoutLeft>::value ||
                              std::is_same<typename DstTraits::array_layout,
                                      flare::LayoutRight>::value ||
                              std::is_same<typename DstTraits::array_layout,
                                      flare::LayoutStride>::value) &&
                             (std::is_same<typename SrcTraits::array_layout,
                                     flare::LayoutLeft>::value ||
                              std::is_same<typename SrcTraits::array_layout,
                                      flare::LayoutRight>::value ||
                              std::is_same<typename SrcTraits::array_layout,
                                      flare::LayoutStride>::value))))>> {
    private:
        enum {
            is_assignable_space = flare::detail::MemorySpaceAccess<
                    typename DstTraits::memory_space,
                    typename SrcTraits::memory_space>::assignable
        };

        enum {
            is_assignable_value_type =
            std::is_same<typename DstTraits::value_type,
                    typename SrcTraits::value_type>::value ||
            std::is_same<typename DstTraits::value_type,
                    typename SrcTraits::const_value_type>::value
        };

        enum {
            is_assignable_dimension =
            TensorDimensionAssignable<typename DstTraits::dimension,
                    typename SrcTraits::dimension>::value
        };

        enum {
            is_assignable_layout =
            std::is_same<typename DstTraits::array_layout,
                    typename SrcTraits::array_layout>::value ||
            std::is_same<typename DstTraits::array_layout,
                    flare::LayoutStride>::value ||
            (DstTraits::dimension::rank == 0) || (DstTraits::dimension::rank == 1)
        };

    public:
        enum {
            is_assignable_data_type =
            is_assignable_value_type && is_assignable_dimension
        };
        enum {
            is_assignable = is_assignable_space && is_assignable_value_type &&
                            is_assignable_dimension && is_assignable_layout
        };

        using TrackType = flare::detail::SharedAllocationTracker;
        using DstType = TensorMapping<DstTraits, void>;
        using SrcType = TensorMapping<SrcTraits, void>;

        FLARE_INLINE_FUNCTION
        static void assign(DstType &dst, const SrcType &src,
                           const TrackType &src_track) {
            static_assert(is_assignable_space,
                          "Tensor assignment must have compatible spaces");

            static_assert(
                    is_assignable_value_type,
                    "Tensor assignment must have same value type or const = non-const");

            static_assert(is_assignable_dimension,
                          "Tensor assignment must have compatible dimensions");

            static_assert(
                    is_assignable_layout,
                    "Tensor assignment must have compatible layout or have rank <= 1");

            using dst_offset_type = typename DstType::offset_type;

            if (size_t(DstTraits::dimension::rank_dynamic) <
                size_t(SrcTraits::dimension::rank_dynamic)) {
                using dst_dim = typename DstTraits::dimension;
                bool assignable = ((1 > DstTraits::dimension::rank_dynamic &&
                                    1 <= SrcTraits::dimension::rank_dynamic)
                                   ? dst_dim::ArgN0 == src.dimension_0()
                                   : true) &&
                                  ((2 > DstTraits::dimension::rank_dynamic &&
                                    2 <= SrcTraits::dimension::rank_dynamic)
                                   ? dst_dim::ArgN1 == src.dimension_1()
                                   : true) &&
                                  ((3 > DstTraits::dimension::rank_dynamic &&
                                    3 <= SrcTraits::dimension::rank_dynamic)
                                   ? dst_dim::ArgN2 == src.dimension_2()
                                   : true) &&
                                  ((4 > DstTraits::dimension::rank_dynamic &&
                                    4 <= SrcTraits::dimension::rank_dynamic)
                                   ? dst_dim::ArgN3 == src.dimension_3()
                                   : true) &&
                                  ((5 > DstTraits::dimension::rank_dynamic &&
                                    5 <= SrcTraits::dimension::rank_dynamic)
                                   ? dst_dim::ArgN4 == src.dimension_4()
                                   : true) &&
                                  ((6 > DstTraits::dimension::rank_dynamic &&
                                    6 <= SrcTraits::dimension::rank_dynamic)
                                   ? dst_dim::ArgN5 == src.dimension_5()
                                   : true) &&
                                  ((7 > DstTraits::dimension::rank_dynamic &&
                                    7 <= SrcTraits::dimension::rank_dynamic)
                                   ? dst_dim::ArgN6 == src.dimension_6()
                                   : true) &&
                                  ((8 > DstTraits::dimension::rank_dynamic &&
                                    8 <= SrcTraits::dimension::rank_dynamic)
                                   ? dst_dim::ArgN7 == src.dimension_7()
                                   : true);
                if (!assignable)
                    flare::abort(
                            "Tensor Assignment: trying to assign runtime dimension to non "
                            "matching compile time dimension.");
            }
            dst.m_impl_offset = dst_offset_type(src.m_impl_offset);
            dst.m_impl_handle = flare::detail::TensorDataHandle<DstTraits>::assign(
                    src.m_impl_handle, src_track);
        }
    };

//----------------------------------------------------------------------------
// Create new specialization for SrcType of LayoutStride. Runtime check for
// compatible layout
    template<class DstTraits, class SrcTraits>
    class TensorMapping<
            DstTraits, SrcTraits,
            std::enable_if_t<(
                    std::is_same<typename SrcTraits::array_layout,
                            flare::LayoutStride>::value &&
                    std::is_void<typename DstTraits::specialize>::value &&
                    std::is_void<typename SrcTraits::specialize>::value &&
                    (
                            // same layout
                            std::is_same<typename DstTraits::array_layout,
                                    typename SrcTraits::array_layout>::value ||
                            // known layout
                            (std::is_same<typename DstTraits::array_layout,
                                    flare::LayoutLeft>::value ||
                             std::is_same<typename DstTraits::array_layout,
                                     flare::LayoutRight>::value ||
                             std::is_same<typename DstTraits::array_layout,
                                     flare::LayoutStride>::value)))>> {
    private:
        enum {
            is_assignable_space = flare::detail::MemorySpaceAccess<
                    typename DstTraits::memory_space,
                    typename SrcTraits::memory_space>::assignable
        };

        enum {
            is_assignable_value_type =
            std::is_same<typename DstTraits::value_type,
                    typename SrcTraits::value_type>::value ||
            std::is_same<typename DstTraits::value_type,
                    typename SrcTraits::const_value_type>::value
        };

        enum {
            is_assignable_dimension =
            TensorDimensionAssignable<typename DstTraits::dimension,
                    typename SrcTraits::dimension>::value
        };

    public:
        enum {
            is_assignable_data_type =
            is_assignable_value_type && is_assignable_dimension
        };
        enum {
            is_assignable = is_assignable_space && is_assignable_value_type &&
                            is_assignable_dimension
        };

        using TrackType = flare::detail::SharedAllocationTracker;
        using DstType = TensorMapping<DstTraits, void>;
        using SrcType = TensorMapping<SrcTraits, void>;

        FLARE_INLINE_FUNCTION
        static bool assignable_layout_check(DstType &,
                                            const SrcType &src)  // Runtime check
        {
            size_t strides[9];
            bool assignable = true;
            src.stride(strides);
            size_t exp_stride = 1;
            if (std::is_same<typename DstTraits::array_layout,
                    flare::LayoutLeft>::value) {
                for (unsigned int i = 0; i < src.Rank; i++) {
                    if (i > 0) exp_stride *= src.extent(i - 1);
                    if (strides[i] != exp_stride) {
                        assignable = false;
                        break;
                    }
                }
            } else if (std::is_same<typename DstTraits::array_layout,
                    flare::LayoutRight>::value) {
                for (unsigned int i = 0; i < src.Rank; i++) {
                    if (i > 0) exp_stride *= src.extent(src.Rank - i);
                    if (strides[src.Rank - 1 - i] != exp_stride) {
                        assignable = false;
                        break;
                    }
                }
            }
            return assignable;
        }

        FLARE_INLINE_FUNCTION
        static void assign(DstType &dst, const SrcType &src,
                           const TrackType &src_track) {
            static_assert(is_assignable_space,
                          "Tensor assignment must have compatible spaces");

            static_assert(
                    is_assignable_value_type,
                    "Tensor assignment must have same value type or const = non-const");

            static_assert(is_assignable_dimension,
                          "Tensor assignment must have compatible dimensions");

            bool assignable_layout = assignable_layout_check(dst, src);  // Runtime
            // check
            if (!assignable_layout)
                flare::abort("Tensor assignment must have compatible layouts\n");

            using dst_offset_type = typename DstType::offset_type;

            if (size_t(DstTraits::dimension::rank_dynamic) <
                size_t(SrcTraits::dimension::rank_dynamic)) {
                using dst_dim = typename DstTraits::dimension;
                bool assignable = ((1 > DstTraits::dimension::rank_dynamic &&
                                    1 <= SrcTraits::dimension::rank_dynamic)
                                   ? dst_dim::ArgN0 == src.dimension_0()
                                   : true) &&
                                  ((2 > DstTraits::dimension::rank_dynamic &&
                                    2 <= SrcTraits::dimension::rank_dynamic)
                                   ? dst_dim::ArgN1 == src.dimension_1()
                                   : true) &&
                                  ((3 > DstTraits::dimension::rank_dynamic &&
                                    3 <= SrcTraits::dimension::rank_dynamic)
                                   ? dst_dim::ArgN2 == src.dimension_2()
                                   : true) &&
                                  ((4 > DstTraits::dimension::rank_dynamic &&
                                    4 <= SrcTraits::dimension::rank_dynamic)
                                   ? dst_dim::ArgN3 == src.dimension_3()
                                   : true) &&
                                  ((5 > DstTraits::dimension::rank_dynamic &&
                                    5 <= SrcTraits::dimension::rank_dynamic)
                                   ? dst_dim::ArgN4 == src.dimension_4()
                                   : true) &&
                                  ((6 > DstTraits::dimension::rank_dynamic &&
                                    6 <= SrcTraits::dimension::rank_dynamic)
                                   ? dst_dim::ArgN5 == src.dimension_5()
                                   : true) &&
                                  ((7 > DstTraits::dimension::rank_dynamic &&
                                    7 <= SrcTraits::dimension::rank_dynamic)
                                   ? dst_dim::ArgN6 == src.dimension_6()
                                   : true) &&
                                  ((8 > DstTraits::dimension::rank_dynamic &&
                                    8 <= SrcTraits::dimension::rank_dynamic)
                                   ? dst_dim::ArgN7 == src.dimension_7()
                                   : true);
                if (!assignable)
                    flare::abort(
                            "Tensor Assignment: trying to assign runtime dimension to non "
                            "matching compile time dimension.");
            }
            dst.m_impl_offset = dst_offset_type(src.m_impl_offset);
            dst.m_impl_handle = flare::detail::TensorDataHandle<DstTraits>::assign(
                    src.m_impl_handle, src_track);
        }
    };

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
// Subtensor mapping.
// Deduce destination tensor type from source tensor traits and subtensor arguments

    template<class, class ValueType, class Exts, class... Args>
    struct SubTensorDataTypeImpl;

/* base case */
    template<class ValueType>
    struct SubTensorDataTypeImpl<void, ValueType, flare::experimental::Extents<>> {
        using type = ValueType;
    };

/* for integral args, subtensor doesn't have that dimension */
    template<class ValueType, ptrdiff_t Ext, ptrdiff_t... Exts, class Integral,
            class... Args>
    struct SubTensorDataTypeImpl<
            std::enable_if_t<std::is_integral<std::decay_t<Integral>>::value>,
            ValueType, flare::experimental::Extents<Ext, Exts...>, Integral, Args...>
            : SubTensorDataTypeImpl<void, ValueType,
                    flare::experimental::Extents<Exts...>, Args...> {
    };

/* for ALL slice, subtensor has the same dimension */
    template<class ValueType, ptrdiff_t Ext, ptrdiff_t... Exts, class... Args>
    struct SubTensorDataTypeImpl<void, ValueType,
            flare::experimental::Extents<Ext, Exts...>,
            flare::ALL_t, Args...>
            : SubTensorDataTypeImpl<void, typename ApplyExtent<ValueType, Ext>::type,
                    flare::experimental::Extents<Exts...>, Args...> {
    };

/* for pair-style slice, subtensor has dynamic dimension, since pair doesn't give
 * static sizes */
/* Since we don't allow interleaving of dynamic and static extents, make all of
 * the dimensions to the left dynamic  */
    template<class ValueType, ptrdiff_t Ext, ptrdiff_t... Exts, class PairLike,
            class... Args>
    struct SubTensorDataTypeImpl<
            std::enable_if_t<is_pair_like<PairLike>::value>, ValueType,
            flare::experimental::Extents<Ext, Exts...>, PairLike, Args...>
            : SubTensorDataTypeImpl<
                    void, typename make_all_extents_into_pointers<ValueType>::type *,
                    flare::experimental::Extents<Exts...>, Args...> {
    };

    template<class ValueType, class Exts, class... Args>
    struct SubTensorDataType : SubTensorDataTypeImpl<void, ValueType, Exts, Args...> {
    };

//----------------------------------------------------------------------------

    template<class SrcTraits, class... Args>
    class TensorMapping<
            std::enable_if_t<(std::is_void<typename SrcTraits::specialize>::value &&
                              (std::is_same<typename SrcTraits::array_layout,
                                      flare::LayoutLeft>::value ||
                               std::is_same<typename SrcTraits::array_layout,
                                       flare::LayoutRight>::value ||
                               std::is_same<typename SrcTraits::array_layout,
                                       flare::LayoutStride>::value))>,
            SrcTraits, Args...> {
    private:
        static_assert(SrcTraits::rank == sizeof...(Args),
                      "Subtensor mapping requires one argument for each dimension of "
                      "source Tensor");

        enum {
            RZ = false,
            R0 = bool(is_integral_extent<0, Args...>::value),
            R1 = bool(is_integral_extent<1, Args...>::value),
            R2 = bool(is_integral_extent<2, Args...>::value),
            R3 = bool(is_integral_extent<3, Args...>::value),
            R4 = bool(is_integral_extent<4, Args...>::value),
            R5 = bool(is_integral_extent<5, Args...>::value),
            R6 = bool(is_integral_extent<6, Args...>::value),
            R7 = bool(is_integral_extent<7, Args...>::value)
        };

        enum {
            rank = unsigned(R0) + unsigned(R1) + unsigned(R2) + unsigned(R3) +
                   unsigned(R4) + unsigned(R5) + unsigned(R6) + unsigned(R7)
        };

        // Whether right-most rank is a range.
        enum {
            R0_rev =
            (0 == SrcTraits::rank
             ? RZ
             : (1 == SrcTraits::rank
                ? R0
                : (2 == SrcTraits::rank
                   ? R1
                   : (3 == SrcTraits::rank
                      ? R2
                      : (4 == SrcTraits::rank
                         ? R3
                         : (5 == SrcTraits::rank
                            ? R4
                            : (6 == SrcTraits::rank
                               ? R5
                               : (7 == SrcTraits::rank
                                  ? R6
                                  : R7))))))))
        };

        // Subtensor's layout
        using array_layout = std::conditional_t<
                (            /* Same array layout IF */
                        (rank == 0) /* output rank zero */
                        || SubtensorLegalArgsCompileTime<typename SrcTraits::array_layout,
                                typename SrcTraits::array_layout, rank,
                                SrcTraits::rank, 0, Args...>::value ||
                        // OutputRank 1 or 2, InputLayout Left, Interval 0
                        // because single stride one or second index has a stride.
                        (rank <= 2 && R0 &&
                         std::is_same<typename SrcTraits::array_layout,
                                 flare::LayoutLeft>::value)  // replace with input rank
                        ||
                        // OutputRank 1 or 2, InputLayout Right, Interval [InputRank-1]
                        // because single stride one or second index has a stride.
                        (rank <= 2 && R0_rev &&
                         std::is_same<typename SrcTraits::array_layout,
                                 flare::LayoutRight>::value)  // replace input rank
                ),
                typename SrcTraits::array_layout, flare::LayoutStride>;

        using value_type = typename SrcTraits::value_type;

        using data_type =
                typename SubTensorDataType<value_type,
                        typename flare::detail::ParseTensorExtents<
                                typename SrcTraits::data_type>::type,
                        Args...>::type;

    public:
        using traits_type = flare::TensorTraits<data_type, array_layout,
                typename SrcTraits::device_type,
                typename SrcTraits::memory_traits>;

        using type =
                flare::Tensor<data_type, array_layout, typename SrcTraits::device_type,
                        typename SrcTraits::memory_traits>;

        template<class MemoryTraits>
        struct apply {
            static_assert(flare::is_memory_traits<MemoryTraits>::value, "");

            using traits_type =
                    flare::TensorTraits<data_type, array_layout,
                            typename SrcTraits::device_type, MemoryTraits>;

            using type = flare::Tensor<data_type, array_layout,
                    typename SrcTraits::device_type, MemoryTraits>;
        };

        // The presumed type is 'TensorMapping< traits_type , void >'
        // However, a compatible TensorMapping is acceptable.
        template<class DstTraits>
        FLARE_INLINE_FUNCTION static void assign(
                TensorMapping<DstTraits, void> &dst,
                TensorMapping<SrcTraits, void> const &src, Args... args) {
            static_assert(TensorMapping<DstTraits, traits_type, void>::is_assignable,
                          "Subtensor destination type must be compatible with subtensor "
                          "derived type");

            using DstType = TensorMapping<DstTraits, void>;

            using dst_offset_type = typename DstType::offset_type;

            const SubTensorExtents<SrcTraits::rank, rank> extents(src.m_impl_offset.m_dim,
                                                                args...);

            dst.m_impl_offset = dst_offset_type(src.m_impl_offset, extents);

            dst.m_impl_handle = TensorDataHandle<DstTraits>::assign(
                    src.m_impl_handle,
                    src.m_impl_offset(extents.domain_offset(0), extents.domain_offset(1),
                                      extents.domain_offset(2), extents.domain_offset(3),
                                      extents.domain_offset(4), extents.domain_offset(5),
                                      extents.domain_offset(6), extents.domain_offset(7)));
        }
    };


    template<unsigned, class MapType>
    FLARE_INLINE_FUNCTION bool tensor_verify_operator_bounds(const MapType &) {
        return true;
    }

    template<unsigned R, class MapType, class iType, class... Args>
    FLARE_INLINE_FUNCTION bool tensor_verify_operator_bounds(const MapType &map,
                                                           const iType &i,
                                                           Args... args) {
        return (size_t(i) < map.extent(R)) &&
               tensor_verify_operator_bounds<R + 1>(map, args...);
    }

    template<unsigned, class MapType>
    inline void tensor_error_operator_bounds(char *, int, const MapType &) {}

    template<unsigned R, class MapType, class iType, class... Args>
    inline void tensor_error_operator_bounds(char *buf, int len, const MapType &map,
                                           const iType &i, Args... args) {
        const int n = snprintf(
                buf, len, " %ld < %ld %c", static_cast<unsigned long>(i),
                static_cast<unsigned long>(map.extent(R)), (sizeof...(Args) ? ',' : ')'));
        tensor_error_operator_bounds<R + 1>(buf + n, len - n, map, args...);
    }

/* Check #3: is the Tensor managed as determined by the MemoryTraits? */
    template<class MapType, bool is_managed = (MapType::is_managed != 0)>
    struct OperatorBoundsErrorOnDevice;

    template<class MapType>
    struct OperatorBoundsErrorOnDevice<MapType, false> {
        FLARE_INLINE_FUNCTION
        static void run(MapType const &) { flare::abort("Tensor bounds error"); }
    };

    template<class MapType>
    struct OperatorBoundsErrorOnDevice<MapType, true> {
        FLARE_INLINE_FUNCTION
        static void run(MapType const &map) {
            SharedAllocationHeader const *const header =
                    SharedAllocationHeader::get_header(
                            static_cast<void const *>(map.data()));
            char const *const label = header->label();
            enum {
                LEN = 128
            };
            char msg[LEN];
            char const *const first_part = "Tensor bounds error of tensor ";
            char *p = msg;
            char *const end = msg + LEN - 1;
            for (char const *p2 = first_part; (*p2 != '\0') && (p < end); ++p, ++p2) {
                *p = *p2;
            }
            for (char const *p2 = label; (*p2 != '\0') && (p < end); ++p, ++p2) {
                *p = *p2;
            }
            *p = '\0';
            flare::abort(msg);
        }
    };

/* Check #2: does the TensorMapping have the printable_label_typedef defined?
   See above that only the non-specialized standard-layout TensorMapping has
   this defined by default.
   The existence of this alias indicates the existence of MapType::is_managed
 */
    template<class T>
    using printable_label_typedef_t = typename T::printable_label_typedef;

    template<class Map>
    FLARE_FUNCTION
    std::enable_if_t<!is_detected<printable_label_typedef_t, Map>::value>
    operator_bounds_error_on_device(Map const &) {
        flare::abort("Tensor bounds error");
    }

    template<class Map>
    FLARE_FUNCTION
    std::enable_if_t<is_detected<printable_label_typedef_t, Map>::value>
    operator_bounds_error_on_device(Map const &map) {
        OperatorBoundsErrorOnDevice<Map>::run(map);
    }

    template<class MemorySpace, class TensorType, class MapType, class... Args>
    FLARE_INLINE_FUNCTION void tensor_verify_operator_bounds(
            flare::detail::TensorTracker<TensorType> const &tracker, const MapType &map,
            Args... args) {
        if (!tensor_verify_operator_bounds<0>(map, args...)) {
            FLARE_IF_ON_HOST(
            (enum { LEN = 1024 }; char buffer[LEN];
                    const std::string label =
                    tracker.m_tracker.template get_label<MemorySpace>();
                    int n = snprintf(buffer, LEN, "Tensor bounds error of tensor %s (",
                    label.c_str());
                    tensor_error_operator_bounds<0>(buffer + n, LEN - n, map, args...);
                    flare::detail::throw_runtime_exception(std::string(buffer));))

            FLARE_IF_ON_DEVICE((
                                       /* Check #1: is there a SharedAllocationRecord?
           (we won't use it, but if its not there then there isn't
            a corresponding SharedAllocationHeader containing a label).
           This check should cover the case of Tensors that don't
           have the Unmanaged trait but were initialized by pointer. */
                                       if (tracker.m_tracker.has_record()) {
                                       operator_bounds_error_on_device(map);
                               } else { flare::abort("Tensor bounds error"); }))
        }
    }

// primary template: memory space is accessible, do nothing.
    template<class MemorySpace, class AccessSpace,
            bool = SpaceAccessibility<AccessSpace, MemorySpace>::accessible>
    struct RuntimeCheckTensorMemoryAccessViolation {
        template<class Track, class Map>
        FLARE_FUNCTION RuntimeCheckTensorMemoryAccessViolation(char const *const,
                                                             Track const &,
                                                             Map const &) {}
    };

// explicit specialization: memory access violation will occur, call abort with
// the specified error message.
    template<class MemorySpace, class AccessSpace>
    struct RuntimeCheckTensorMemoryAccessViolation<MemorySpace, AccessSpace, false> {
        template<class Track, class Map>
        FLARE_FUNCTION RuntimeCheckTensorMemoryAccessViolation(char const *const msg,
                                                             Track const &track,
                                                             Map const &) {
            char err[256] = "";
            strncat(err, msg, 64);
            strcat(err, " (label=\"");

            FLARE_IF_ON_HOST(({
                auto const tracker = track.m_tracker;

                if (tracker.has_record()) {
                    strncat(err, tracker.template get_label<void>().c_str(), 128);
                } else {
                    strcat(err, "**UNMANAGED**");
                }
            }))

            FLARE_IF_ON_DEVICE(({
                strcat(err, "**UNAVAILABLE**");
                (void) track;
            }))

            strcat(err, "\")");

            flare::abort(err);
        }
    };

    template<class MemorySpace, class Track, class Map, class... Ignore>
    FLARE_FUNCTION void runtime_check_memory_access_violation(
            char const *const msg, Track const &track, Map const &map, Ignore...) {
        FLARE_IF_ON_HOST(
                ((void) RuntimeCheckTensorMemoryAccessViolation<MemorySpace,
                        DefaultHostExecutionSpace>(
                        msg, track, map);))
        FLARE_IF_ON_DEVICE(
                ((void) RuntimeCheckTensorMemoryAccessViolation<MemorySpace,
                        DefaultExecutionSpace>(
                        msg, track, map);))
    }

}  // namespace flare

#endif  // FLARE_CORE_TENSOR_TENSOR_MAPPING_H_
