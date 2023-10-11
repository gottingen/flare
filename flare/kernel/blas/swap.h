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
//
// Created by jeff on 23-10-8.
//

#ifndef FLARE_KERNEL_BLAS_SWAP_H_
#define FLARE_KERNEL_BLAS_SWAP_H_

#include <flare/kernel/blas/swap_impl.h>
#include <flare/core/layout_utility.h>

namespace flare::blas {

    /// \brief Swaps the entries of vectors x and y.
    ///
    /// \tparam execution_space an execution space to perform parallel work
    /// \tparam XVector Type of the first vector x; a 1-D flare::View.
    /// \tparam YVector Type of the first vector y; a 1-D flare::View.
    ///
    /// \param space [in] execution space passed to execution policies
    /// \param x [in/out] 1-D View.
    /// \param y [in/out] 1-D View.
    ///
    /// Swaps x and y. Note that this is akin to performing a deep_copy, swapping
    /// pointers inside view can only be performed if no aliasing, subviews, etc...
    /// exist, which cannot be asserted by this function.
    ///
    /// This function is non-blocking unless the underlying TPL requested
    /// at compile time is itself blocking
    template <class execution_space, class XVector, class YVector>
    void swap(execution_space const& space, XVector const& x, YVector const& y) {
        // Assert properties of XVector
        static_assert(flare::is_view<XVector>::value,
                      "flare::blas::swap: XVector must be a flare::View.");
        static_assert(XVector::rank == 1,
                      "flare::blas::swap: "
                      "Input vector x must have rank 1.");
        static_assert(std::is_same_v<typename XVector::value_type,
                      typename XVector::non_const_value_type>,
                      "flare::blas::swap: XVector must store non const values.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename XVector::memory_space>::accessible,
                "swap: execution_space cannot access data in XVector");

        // Assert properties of YVector, could probably use a function for this as
        // XVector and YVector are required to have identical properties...
        static_assert(flare::is_view<YVector>::value,
                      "flare::blas::swap: YVector must be a flare::View.");
        static_assert(YVector::rank == 1,
                      "flare::blas::swap: "
                      "Input vector y must have rank 1.");
        static_assert(std::is_same_v<typename YVector::value_type,
                      typename YVector::non_const_value_type>,
                      "flare::blas::swap: YVector must store non const values.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename YVector::memory_space>::accessible,
                "swap: execution_space cannot access data in YVector");

        using XVector_Internal = flare::View<
                typename XVector::non_const_value_type*,
                typename flare::detail::GetUnifiedLayout<XVector>::array_layout,
                flare::Device<execution_space, typename XVector::memory_space>,
                flare::MemoryTraits<flare::Unmanaged> >;
        using YVector_Internal = flare::View<
                typename YVector::non_const_value_type*,
                typename flare::detail::GetUnifiedLayout<YVector>::array_layout,
                flare::Device<execution_space, typename YVector::memory_space>,
                flare::MemoryTraits<flare::Unmanaged> >;

        XVector_Internal X(x);
        YVector_Internal Y(y);

        // Runtime check of the length of X and Y
        if (static_cast<int64_t>(X.extent(0)) != static_cast<int64_t>(Y.extent(0))) {
            throw std::runtime_error("X and Y must have equal extents!");
        }

        flare::Profiling::pushRegion("flare::blas::swap");
        // If X.extent(0) == 0, do nothing
        if (X.extent(0) != 0) {
            flare::blas::detail::Swap<execution_space, XVector_Internal, YVector_Internal>::swap(space,
                                                                                  X, Y);
        }
        flare::Profiling::popRegion();
    }

    /// \brief Swaps the entries of vectors x and y.
    ///
    /// \tparam XVector Type of the first vector x; a 1-D flare::View.
    /// \tparam YVector Type of the first vector y; a 1-D flare::View.
    ///
    /// \param x [in/out] 1-D View.
    /// \param y [in/out] 1-D View.
    ///
    /// This function is non-blocking unless the underlying TPL requested
    /// at compile time is itself blocking. Note that the kernel will be
    /// executed on the default stream of the execution_space associted with x.
    template <class XVector, class YVector>
    void swap(const XVector& x, const YVector& y) {
        const typename XVector::execution_space space =
                typename XVector::execution_space();
        swap(space, x, y);
    }

}  // namespace flare::blas


#endif  // FLARE_KERNEL_BLAS_SWAP_H_
