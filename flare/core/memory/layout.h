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

/// \file layout.hpp
/// \brief Declaration of various \c MemoryLayout options.

#ifndef FLARE_CORE_MEMORY_LAYOUT_H_
#define FLARE_CORE_MEMORY_LAYOUT_H_

#include <cstddef>
#include <flare/core/common/traits.h>

namespace flare {

    enum {
        ARRAY_LAYOUT_MAX_RANK = 8
    };

    //----------------------------------------------------------------------------
    /// \struct LayoutLeft
    /// \brief Memory layout tag indicating left-to-right (Fortran scheme)
    ///   striding of multi-indices.
    ///
    /// This is an example of a \c MemoryLayout template parameter of
    /// View.  The memory layout describes how View maps from a
    /// multi-index (i0, i1, ..., ik) to a memory location.
    ///
    /// "Layout left" indicates a mapping where the leftmost index i0
    /// refers to contiguous access, and strides increase for dimensions
    /// going right from there (i1, i2, ...).  This layout imitates how
    /// Fortran stores multi-dimensional arrays.  For the special case of
    /// a two-dimensional array, "layout left" is also called "column
    /// major."
    struct LayoutLeft {
        //! Tag this class as a flare array layout
        using array_layout = LayoutLeft;

        size_t dimension[ARRAY_LAYOUT_MAX_RANK];

        enum : bool {
            is_extent_constructible = true
        };

        LayoutLeft(LayoutLeft const &) = default;

        LayoutLeft(LayoutLeft &&) = default;

        LayoutLeft &operator=(LayoutLeft const &) = default;

        LayoutLeft &operator=(LayoutLeft &&) = default;

        FLARE_INLINE_FUNCTION
        explicit constexpr LayoutLeft(size_t N0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
                                      size_t N1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
                                      size_t N2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
                                      size_t N3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
                                      size_t N4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
                                      size_t N5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
                                      size_t N6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
                                      size_t N7 = FLARE_IMPL_CTOR_DEFAULT_ARG)
                : dimension{N0, N1, N2, N3, N4, N5, N6, N7} {}

        friend bool operator==(const LayoutLeft &left, const LayoutLeft &right) {
            for (unsigned int rank = 0; rank < ARRAY_LAYOUT_MAX_RANK; ++rank)
                if (left.dimension[rank] != right.dimension[rank]) return false;
            return true;
        }

        friend bool operator!=(const LayoutLeft &left, const LayoutLeft &right) {
            return !(left == right);
        }
    };

    //----------------------------------------------------------------------------
    /// \struct LayoutRight
    /// \brief Memory layout tag indicating right-to-left (C or
    ///   lexigraphical scheme) striding of multi-indices.
    ///
    /// This is an example of a \c MemoryLayout template parameter of
    /// View.  The memory layout describes how View maps from a
    /// multi-index (i0, i1, ..., ik) to a memory location.
    ///
    /// "Right layout" indicates a mapping where the rightmost index ik
    /// refers to contiguous access, and strides increase for dimensions
    /// going left from there.  This layout imitates how C stores
    /// multi-dimensional arrays.  For the special case of a
    /// two-dimensional array, "layout right" is also called "row major."
    struct LayoutRight {
        //! Tag this class as a flare array layout
        using array_layout = LayoutRight;

        size_t dimension[ARRAY_LAYOUT_MAX_RANK];

        enum : bool {
            is_extent_constructible = true
        };

        LayoutRight(LayoutRight const &) = default;

        LayoutRight(LayoutRight &&) = default;

        LayoutRight &operator=(LayoutRight const &) = default;

        LayoutRight &operator=(LayoutRight &&) = default;

        FLARE_INLINE_FUNCTION
        explicit constexpr LayoutRight(size_t N0 = FLARE_IMPL_CTOR_DEFAULT_ARG,
                                       size_t N1 = FLARE_IMPL_CTOR_DEFAULT_ARG,
                                       size_t N2 = FLARE_IMPL_CTOR_DEFAULT_ARG,
                                       size_t N3 = FLARE_IMPL_CTOR_DEFAULT_ARG,
                                       size_t N4 = FLARE_IMPL_CTOR_DEFAULT_ARG,
                                       size_t N5 = FLARE_IMPL_CTOR_DEFAULT_ARG,
                                       size_t N6 = FLARE_IMPL_CTOR_DEFAULT_ARG,
                                       size_t N7 = FLARE_IMPL_CTOR_DEFAULT_ARG)
                : dimension{N0, N1, N2, N3, N4, N5, N6, N7} {}

        friend bool operator==(const LayoutRight &left, const LayoutRight &right) {
            for (unsigned int rank = 0; rank < ARRAY_LAYOUT_MAX_RANK; ++rank)
                if (left.dimension[rank] != right.dimension[rank]) return false;
            return true;
        }

        friend bool operator!=(const LayoutRight &left, const LayoutRight &right) {
            return !(left == right);
        }
    };

    //----------------------------------------------------------------------------
    /// \struct LayoutStride
    /// \brief  Memory layout tag indicated arbitrarily strided
    ///         multi-index mapping into contiguous memory.
    struct LayoutStride {
        //! Tag this class as a flare array layout
        using array_layout = LayoutStride;

        size_t dimension[ARRAY_LAYOUT_MAX_RANK];
        size_t stride[ARRAY_LAYOUT_MAX_RANK];

        enum : bool {
            is_extent_constructible = false
        };

        LayoutStride(LayoutStride const &) = default;

        LayoutStride(LayoutStride &&) = default;

        LayoutStride &operator=(LayoutStride const &) = default;

        LayoutStride &operator=(LayoutStride &&) = default;

        /** \brief  Compute strides from ordered dimensions.
         *
         *  Values of order uniquely form the set [0..rank)
         *  and specify ordering of the dimensions.
         *  Order = {0,1,2,...} is LayoutLeft
         *  Order = {...,2,1,0} is LayoutRight
         */
        template<typename iTypeOrder, typename iTypeDimen>
        FLARE_INLINE_FUNCTION static LayoutStride order_dimensions(
                int const rank, iTypeOrder const *const order,
                iTypeDimen const *const dimen) {
            LayoutStride tmp;
            // Verify valid rank order:
            int check_input = ARRAY_LAYOUT_MAX_RANK < rank ? 0 : int(1 << rank) - 1;
            for (int r = 0; r < ARRAY_LAYOUT_MAX_RANK; ++r) {
                tmp.dimension[r] = FLARE_IMPL_CTOR_DEFAULT_ARG;
                tmp.stride[r] = 0;
            }
            for (int r = 0; r < rank; ++r) {
                check_input &= ~int(1 << order[r]);
            }
            if (0 == check_input) {
                size_t n = 1;
                for (int r = 0; r < rank; ++r) {
                    tmp.stride[order[r]] = n;
                    n *= (dimen[order[r]]);
                    tmp.dimension[r] = dimen[r];
                }
            }
            return tmp;
        }

        FLARE_INLINE_FUNCTION
        explicit constexpr LayoutStride(
                size_t N0 = FLARE_IMPL_CTOR_DEFAULT_ARG, size_t S0 = 0,
                size_t N1 = FLARE_IMPL_CTOR_DEFAULT_ARG, size_t S1 = 0,
                size_t N2 = FLARE_IMPL_CTOR_DEFAULT_ARG, size_t S2 = 0,
                size_t N3 = FLARE_IMPL_CTOR_DEFAULT_ARG, size_t S3 = 0,
                size_t N4 = FLARE_IMPL_CTOR_DEFAULT_ARG, size_t S4 = 0,
                size_t N5 = FLARE_IMPL_CTOR_DEFAULT_ARG, size_t S5 = 0,
                size_t N6 = FLARE_IMPL_CTOR_DEFAULT_ARG, size_t S6 = 0,
                size_t N7 = FLARE_IMPL_CTOR_DEFAULT_ARG, size_t S7 = 0)
                : dimension{N0, N1, N2, N3, N4, N5, N6, N7}, stride{S0, S1, S2, S3,
                                                                    S4, S5, S6, S7} {}

        friend bool operator==(const LayoutStride &left, const LayoutStride &right) {
            for (unsigned int rank = 0; rank < ARRAY_LAYOUT_MAX_RANK; ++rank)
                if (left.dimension[rank] != right.dimension[rank] ||
                    left.stride[rank] != right.stride[rank])
                    return false;
            return true;
        }

        friend bool operator!=(const LayoutStride &left, const LayoutStride &right) {
            return !(left == right);
        }
    };

    // ===================================================================================

    //////////////////////////////////////////////////////////////////////////////////////

    enum class Iterate {
        Default,
        Left,  // Left indices stride fastest
        Right  // Right indices stride fastest
    };

    // To check for LayoutTiled
    // This is to hide extra compile-time 'identifier' info within the LayoutTiled
    // class by not relying on template specialization to include the ArgN*'s
    template<typename LayoutTiledCheck, class Enable = void>
    struct is_layouttiled : std::false_type {
    };

    template<typename LayoutTiledCheck>
    struct is_layouttiled<LayoutTiledCheck,
            std::enable_if_t<LayoutTiledCheck::is_array_layout_tiled>>
            : std::true_type {
    };

    namespace experimental {

        /// LayoutTiled
        // Must have Rank >= 2
        template<
                flare::Iterate OuterP, flare::Iterate InnerP, unsigned ArgN0,
                unsigned ArgN1, unsigned ArgN2 = 0, unsigned ArgN3 = 0, unsigned ArgN4 = 0,
                unsigned ArgN5 = 0, unsigned ArgN6 = 0, unsigned ArgN7 = 0,
                bool IsPowerOfTwo =
                (flare::detail::is_integral_power_of_two(ArgN0) &&
                 flare::detail::is_integral_power_of_two(ArgN1) &&
                 (flare::detail::is_integral_power_of_two(ArgN2) || (ArgN2 == 0)) &&
                 (flare::detail::is_integral_power_of_two(ArgN3) || (ArgN3 == 0)) &&
                 (flare::detail::is_integral_power_of_two(ArgN4) || (ArgN4 == 0)) &&
                 (flare::detail::is_integral_power_of_two(ArgN5) || (ArgN5 == 0)) &&
                 (flare::detail::is_integral_power_of_two(ArgN6) || (ArgN6 == 0)) &&
                 (flare::detail::is_integral_power_of_two(ArgN7) || (ArgN7 == 0)))>
        struct LayoutTiled {
            static_assert(IsPowerOfTwo,
                          "LayoutTiled must be given power-of-two tile dimensions");

            using array_layout = LayoutTiled<OuterP, InnerP, ArgN0, ArgN1, ArgN2, ArgN3,
                    ArgN4, ArgN5, ArgN6, ArgN7, IsPowerOfTwo>;
            static constexpr Iterate outer_pattern = OuterP;
            static constexpr Iterate inner_pattern = InnerP;

            enum {
                N0 = ArgN0
            };
            enum {
                N1 = ArgN1
            };
            enum {
                N2 = ArgN2
            };
            enum {
                N3 = ArgN3
            };
            enum {
                N4 = ArgN4
            };
            enum {
                N5 = ArgN5
            };
            enum {
                N6 = ArgN6
            };
            enum {
                N7 = ArgN7
            };

            size_t dimension[ARRAY_LAYOUT_MAX_RANK];

            enum : bool {
                is_extent_constructible = true
            };

            LayoutTiled(LayoutTiled const &) = default;

            LayoutTiled(LayoutTiled &&) = default;

            LayoutTiled &operator=(LayoutTiled const &) = default;

            LayoutTiled &operator=(LayoutTiled &&) = default;

            FLARE_INLINE_FUNCTION
            explicit constexpr LayoutTiled(size_t argN0 = 0, size_t argN1 = 0,
                                           size_t argN2 = 0, size_t argN3 = 0,
                                           size_t argN4 = 0, size_t argN5 = 0,
                                           size_t argN6 = 0, size_t argN7 = 0)
                    : dimension{argN0, argN1, argN2, argN3, argN4, argN5, argN6, argN7} {}

            friend bool operator==(const LayoutTiled &left, const LayoutTiled &right) {
                for (unsigned int rank = 0; rank < ARRAY_LAYOUT_MAX_RANK; ++rank)
                    if (left.dimension[rank] != right.dimension[rank]) return false;
                return true;
            }

            friend bool operator!=(const LayoutTiled &left, const LayoutTiled &right) {
                return !(left == right);
            }
        };

    }  // namespace experimental

    // For use with view_copy
    template<typename... Layout>
    struct layout_iterate_type_selector {
        static const flare::Iterate outer_iteration_pattern =
                flare::Iterate::Default;
        static const flare::Iterate inner_iteration_pattern =
                flare::Iterate::Default;
    };

    template<>
    struct layout_iterate_type_selector<flare::LayoutRight> {
        static const flare::Iterate outer_iteration_pattern = flare::Iterate::Right;
        static const flare::Iterate inner_iteration_pattern = flare::Iterate::Right;
    };

    template<>
    struct layout_iterate_type_selector<flare::LayoutLeft> {
        static const flare::Iterate outer_iteration_pattern = flare::Iterate::Left;
        static const flare::Iterate inner_iteration_pattern = flare::Iterate::Left;
    };

    template<>
    struct layout_iterate_type_selector<flare::LayoutStride> {
        static const flare::Iterate outer_iteration_pattern =
                flare::Iterate::Default;
        static const flare::Iterate inner_iteration_pattern =
                flare::Iterate::Default;
    };

    template<unsigned ArgN0, unsigned ArgN1, unsigned ArgN2, unsigned ArgN3,
            unsigned ArgN4, unsigned ArgN5, unsigned ArgN6, unsigned ArgN7>
    struct layout_iterate_type_selector<flare::experimental::LayoutTiled<
            flare::Iterate::Left, flare::Iterate::Left, ArgN0, ArgN1, ArgN2, ArgN3,
            ArgN4, ArgN5, ArgN6, ArgN7, true>> {
        static const flare::Iterate outer_iteration_pattern = flare::Iterate::Left;
        static const flare::Iterate inner_iteration_pattern = flare::Iterate::Left;
    };

    template<unsigned ArgN0, unsigned ArgN1, unsigned ArgN2, unsigned ArgN3,
            unsigned ArgN4, unsigned ArgN5, unsigned ArgN6, unsigned ArgN7>
    struct layout_iterate_type_selector<flare::experimental::LayoutTiled<
            flare::Iterate::Right, flare::Iterate::Left, ArgN0, ArgN1, ArgN2, ArgN3,
            ArgN4, ArgN5, ArgN6, ArgN7, true>> {
        static const flare::Iterate outer_iteration_pattern = flare::Iterate::Right;
        static const flare::Iterate inner_iteration_pattern = flare::Iterate::Left;
    };

    template<unsigned ArgN0, unsigned ArgN1, unsigned ArgN2, unsigned ArgN3,
            unsigned ArgN4, unsigned ArgN5, unsigned ArgN6, unsigned ArgN7>
    struct layout_iterate_type_selector<flare::experimental::LayoutTiled<
            flare::Iterate::Left, flare::Iterate::Right, ArgN0, ArgN1, ArgN2, ArgN3,
            ArgN4, ArgN5, ArgN6, ArgN7, true>> {
        static const flare::Iterate outer_iteration_pattern = flare::Iterate::Left;
        static const flare::Iterate inner_iteration_pattern = flare::Iterate::Right;
    };

    template<unsigned ArgN0, unsigned ArgN1, unsigned ArgN2, unsigned ArgN3,
            unsigned ArgN4, unsigned ArgN5, unsigned ArgN6, unsigned ArgN7>
    struct layout_iterate_type_selector<flare::experimental::LayoutTiled<
            flare::Iterate::Right, flare::Iterate::Right, ArgN0, ArgN1, ArgN2, ArgN3,
            ArgN4, ArgN5, ArgN6, ArgN7, true>> {
        static const flare::Iterate outer_iteration_pattern = flare::Iterate::Right;
        static const flare::Iterate inner_iteration_pattern = flare::Iterate::Right;
    };

}  // namespace flare

#endif  // FLARE_CORE_MEMORY_LAYOUT_H_
