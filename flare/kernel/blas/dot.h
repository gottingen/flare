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


#ifndef FLARE_KERNEL_BLAS_DOT_H_
#define FLARE_KERNEL_BLAS_DOT_H_

#include <flare/kernel/blas/dot_impl.h>
#include <flare/core/layout_utility.h>

namespace flare::blas {

    /// \brief Return the dot product of the two vectors x and y.
    ///
    /// \tparam execution_space the flare execution space where the kernel
    ///         will be executed.
    /// \tparam XVector Type of the first vector x; a 1-D flare::View.
    /// \tparam YVector Type of the second vector y; a 1-D flare::View.
    ///
    /// \param space [in] an execution space instance that may specify
    ///                   in which stream/queue the kernel will be executed.
    /// \param x [in] Input 1-D View.
    /// \param y [in] Input 1-D View.
    ///
    /// \return The dot product result; a single value.
    template <class execution_space, class XVector, class YVector,
            typename std::enable_if<flare::is_execution_space_v<execution_space>,
            int>::type = 0>
    typename flare::detail::InnerProductSpaceTraits<
            typename XVector::non_const_value_type>::dot_type
    dot(const execution_space& space, const XVector& x, const YVector& y) {
        static_assert(flare::is_execution_space_v<execution_space>,
                      "flare::blas::dot: execution_space must be a valid flare "
                      "execution space.");
        static_assert(flare::is_view<XVector>::value,
                      "flare::blas::dot: XVector must be a flare::View.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename XVector::memory_space>::accessible,
                "flare::blas::dot: XVector must be accessible from execution_space");
        static_assert(flare::is_view<YVector>::value,
                      "flare::blas::dot: YVector must be a flare::View.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename YVector::memory_space>::accessible,
                "flare::blas::dot: YVector must be accessible from execution_space");
        static_assert((int)XVector::rank == (int)YVector::rank,
                      "flare::blas::dot: Vector ranks do not match.");
        static_assert(XVector::rank == 1,
                      "flare::blas::dot: "
                      "Both Vector inputs must have rank 1.");

        // Check compatibility of dimensions at run time.
        if (x.extent(0) != y.extent(0)) {
            std::ostringstream os;
            os << "flare::blas::dot: Dimensions do not match: "
               << ", x: " << x.extent(0) << " x 1"
               << ", y: " << y.extent(0) << " x 1";
            flare::detail::throw_runtime_exception(os.str());
        }

        using XVector_Internal = flare::View<
                typename XVector::const_value_type*,
                typename flare::detail::GetUnifiedLayout<XVector>::array_layout,
                typename XVector::device_type, flare::MemoryTraits<flare::Unmanaged>>;
        using YVector_Internal = flare::View<
                typename YVector::const_value_type*,
                typename flare::detail::GetUnifiedLayout<YVector>::array_layout,
                typename YVector::device_type, flare::MemoryTraits<flare::Unmanaged>>;

        using dot_type = typename flare::detail::InnerProductSpaceTraits<
                typename XVector::non_const_value_type>::dot_type;
        // result_type is usually just dot_type, except:
        //  if dot_type is float, result_type is double
        //  if dot_type is complex<float>, result_type is complex<double>
        // These special cases are to maintain accuracy.
        using result_type =
                typename flare::blas::detail::DotAccumulatingScalar<dot_type>::type;
        using RVector_Internal =
                flare::View<dot_type, default_layout, flare::HostSpace,
                flare::MemoryTraits<flare::Unmanaged>>;
        using RVector_Result =
                flare::View<result_type, default_layout, flare::HostSpace,
                flare::MemoryTraits<flare::Unmanaged>>;

        result_type result{};
        RVector_Result R   = RVector_Result(&result);
        XVector_Internal X = x;
        YVector_Internal Y = y;

        // Even though RVector is the template parameter, Dot::dot has an overload
        // that accepts RVector_Internal (with the special accumulator, if dot_type is
        // 32-bit precision). Impl::Dot needs to support both cases, and it's easier
        // to do this with overloading than by extending to deal with two
        // different scalar types.
        flare::blas::detail::DotSpecialAccumulator<execution_space, RVector_Internal,
                XVector_Internal, YVector_Internal>::dot(space, R,
                                                         X, Y);
        space.fence();
        // mfh 22 Jan 2020: We need the line below because
        // flare::complex<T> lacks a constructor that takes a
        // flare::complex<U> with U != T.
        return flare::detail::CastPossiblyComplex<dot_type, result_type>::cast(
                result);
    }

    /// \brief Return the dot product of the two vectors x and y.
    ///
    /// The kernel is executed in the default stream/queue associated
    /// with the execution space of XVector.
    ///
    /// \tparam XVector Type of the first vector x; a 1-D flare::View.
    /// \tparam YVector Type of the second vector y; a 1-D flare::View.
    ///
    /// \param x [in] Input 1-D View.
    /// \param y [in] Input 1-D View.
    ///
    /// \return The dot product result; a single value.
    template <class XVector, class YVector>
    typename flare::detail::InnerProductSpaceTraits<
            typename XVector::non_const_value_type>::dot_type
    dot(const XVector& x, const YVector& y) {
        return dot(typename XVector::execution_space{}, x, y);
    }

    /// \brief Compute the column-wise dot products of two multivectors.
    ///
    /// This function is non-blocking and thread-safe.
    ///
    /// \tparam execution_space the flare execution space where the kernel
    ///         will be executed.
    /// \tparam RV 0-D resp. 1-D output View
    /// \tparam XMV 1-D resp. 2-D input View
    /// \tparam YMV 1-D resp. 2-D input View
    ///
    /// \param space [in] an execution space instance that may specify
    ///                   in which stream/queue the kernel will be executed.
    /// \param R [out] Output 1-D or 0-D View to which to write results.
    /// \param X [in] Input 2-D or 1-D View.
    /// \param Y [in] Input 2-D or 1-D View.
    ///
    /// This function implements a few different use cases:
    /// <ul>
    /// <li> If X and Y are both 1-D, then this is a single dot product.
    ///   R must be 0-D (a View of a single value). </li>
    /// <li> If X and Y are both 2-D, then this function computes their
    ///   dot products columnwise.  R must be 1-D. </li>
    /// <li> If X is 2-D and Y is 1-D, then this function computes the dot
    ///   product of each column of X, with Y, in turn.  R must be
    ///   1-D. </li>
    /// <li> If X is 1-D and Y is 2-D, then this function computes the dot
    ///   product X with each column of Y, in turn.  R must be 1-D. </li>
    /// </ul>
    ///
    /// \note To implementers: We use enable_if here so that the compiler
    ///   doesn't confuse this version of dot() with the three-argument
    ///   version of dot() in blas.h.
    template <class execution_space, class RV, class XMV, class YMV>
    void dot(const execution_space& space, const RV& R, const XMV& X, const YMV& Y,
             typename std::enable_if<flare::is_view<RV>::value, int>::type = 0) {
        static_assert(flare::is_execution_space_v<execution_space>,
                      "flare::blas::dot: excution_space must be a valid flare "
                      "execution space.");
        static_assert(flare::is_view<RV>::value,
                      "flare::blas::dot: "
                      "R is not a flare::View.");
        static_assert(flare::is_view<XMV>::value,
                      "flare::blas::dot: "
                      "X is not a flare::View.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename XMV::memory_space>::accessible,
                "flare::blas::dot: XMV must be accessible from execution_space.");
        static_assert(flare::is_view<YMV>::value,
                      "flare::blas::dot: "
                      "Y is not a flare::View.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename YMV::memory_space>::accessible,
                "flare::blas::dot: XMV must be accessible from execution_space.");
        static_assert(std::is_same<typename RV::value_type,
                              typename RV::non_const_value_type>::value,
                      "flare::blas::dot: R is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");
        static_assert(RV::rank == 0 || RV::rank == 1,
                      "flare::blas::dot: R must have rank 0 or 1.");
        static_assert(XMV::rank == 1 || XMV::rank == 2,
                      "flare::blas::dot: X must have rank 1 or 2.");
        static_assert(YMV::rank == 1 || YMV::rank == 2,
                      "flare::blas::dot: Y must have rank 1 or 2.");
        static_assert((XMV::rank == 2 && YMV::rank == 2 && RV::rank == 1) ||
                      (XMV::rank == 1 && YMV::rank == 1 && RV::rank == 0) ||
                      (XMV::rank == 2 && YMV::rank == 1 && RV::rank == 1) ||
                      (XMV::rank == 1 && YMV::rank == 2 && RV::rank == 1),
                      "flare::blas::dot: Ranks of RV, XMV, and YMV don't match.  "
                      "See this function's documentation for the allowed "
                      "combinations of ranks.");

        // Check compatibility of dimensions at run time.

        // Regardless of ranks of X and Y, their numbers of rows must match.
        bool dimsMatch = true;
        if (X.extent(0) != Y.extent(0)) {
            dimsMatch = false;
        } else if (X.extent(1) != Y.extent(1) && X.extent(1) != 1 &&
                   Y.extent(1) != 1) {
            // Numbers of columns don't match, and neither X nor Y have one column.
            dimsMatch = false;
        }
        const auto maxNumCols = X.extent(1) > Y.extent(1) ? X.extent(1) : Y.extent(1);
        if (RV::rank == 1 && R.extent(0) != maxNumCols) {
            dimsMatch = false;
        }

        if (!dimsMatch) {
            std::ostringstream os;
            os << "flare::blas::dot: Dimensions of R, X, and Y do not match: ";
            if (RV::rank == 1) {
                os << "R: " << R.extent(0) << " x " << X.extent(1) << ", ";
            }
            os << "X: " << X.extent(0) << " x " << X.extent(1) << ", Y: " << Y.extent(0)
               << " x " << Y.extent(1);
            flare::detail::throw_runtime_exception(os.str());
        }

        // Create unmanaged versions of the input Views.
        using UnifiedXLayout =
                typename flare::detail::GetUnifiedLayout<XMV>::array_layout;
        using UnifiedRVLayout =
                typename flare::detail::GetUnifiedLayoutPreferring<
                        RV, UnifiedXLayout>::array_layout;

        typedef flare::View<typename std::conditional<
                RV::rank == 0, typename RV::non_const_value_type,
                typename RV::non_const_value_type*>::type,
                UnifiedRVLayout, typename RV::device_type,
                flare::MemoryTraits<flare::Unmanaged>>
                RV_Internal;
        typedef flare::View<
                typename std::conditional<XMV::rank == 1, typename XMV::const_value_type*,
                        typename XMV::const_value_type**>::type,
                UnifiedXLayout, typename XMV::device_type,
                flare::MemoryTraits<flare::Unmanaged>>
                XMV_Internal;
        typedef flare::View<
                typename std::conditional<YMV::rank == 1, typename YMV::const_value_type*,
                        typename YMV::const_value_type**>::type,
                typename flare::detail::GetUnifiedLayout<YMV>::array_layout,
                typename YMV::device_type, flare::MemoryTraits<flare::Unmanaged>>
                YMV_Internal;

        RV_Internal R_internal  = R;
        XMV_Internal X_internal = X;
        YMV_Internal Y_internal = Y;

        flare::blas::detail::Dot<execution_space, RV_Internal, XMV_Internal, YMV_Internal>::dot(
                space, R_internal, X_internal, Y_internal);
    }

    /// \brief Compute the column-wise dot products of two multivectors.
    ///
    /// This function is non-blocking and thread-safe.
    /// The kernel is executed in the default stream/queue associated
    /// with the execution space of XVM.
    ///
    /// \tparam RV 0-D resp. 1-D output View
    /// \tparam XMV 1-D resp. 2-D input View
    /// \tparam YMV 1-D resp. 2-D input View
    ///
    /// \param R [out] Output 1-D or 0-D View to which to write results.
    /// \param X [in] Input 2-D or 1-D View.
    /// \param Y [in] Input 2-D or 1-D View.
    ///
    /// This function implements a few different use cases:
    /// <ul>
    /// <li> If X and Y are both 1-D, then this is a single dot product.
    ///   R must be 0-D (a View of a single value). </li>
    /// <li> If X and Y are both 2-D, then this function computes their
    ///   dot products columnwise.  R must be 1-D. </li>
    /// <li> If X is 2-D and Y is 1-D, then this function computes the dot
    ///   product of each column of X, with Y, in turn.  R must be
    ///   1-D. </li>
    /// <li> If X is 1-D and Y is 2-D, then this function computes the dot
    ///   product X with each column of Y, in turn.  R must be 1-D. </li>
    /// </ul>
    ///
    /// \note To implementers: We use enable_if here so that the compiler
    ///   doesn't confuse this version of dot() with the three-argument
    ///   version of dot() in blas.h.
    template <class RV, class XMV, class YMV>
    void dot(const RV& R, const XMV& X, const YMV& Y,
             typename std::enable_if<flare::is_view<RV>::value, int>::type = 0) {
        dot(typename XMV::execution_space{}, R, X, Y);
    }
}  // namespace flare::blas
#endif  // FLARE_KERNEL_BLAS_DOT_H_
