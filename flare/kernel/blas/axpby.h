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
// Created by jeff on 23-10-7.
//

#ifndef FLARE_KERNEL_BLAS_AXPBY_H_
#define FLARE_KERNEL_BLAS_AXPBY_H_

#include <flare/kernel/blas/axpby_impl.h>
#include <flare/kernel/common/helper.h>

// axpby() accepts both scalar coefficients a and b, and vector
// coefficients (apply one for each column of the input multivectors).
// This traits class helps axpby() select the correct specialization
// of AV and BV (the type of a resp. b) for invoking the
// implementation.

namespace flare::blas {

    /// \brief Computes Y := a*X + b*Y
    ///
    /// This function is non-blocking and thread safe.
    ///
    /// \tparam execution_space a flare execution space where the kernel will run.
    /// \tparam AV 1-D or 2-D flare::View specialization.
    /// \tparam XMV 1-D or 2-D flare::View specialization.
    /// \tparam BV 1-D or 2-D flare::View specialization.
    /// \tparam YMV 1-D or 2-D flare::View specialization. It must have
    ///   the same rank as XMV.
    ///
    /// \param space [in] the execution space instance on which the kernel will run.
    /// \param a [in] view of type AV, scaling parameter for X.
    /// \param X [in] input view of type XMV.
    /// \param b [in] view of type BV, scaling parameter for Y.
    /// \param Y [in/out] view of type YMV in which the results will be stored.
    template <class execution_space, class AV, class XMV, class BV, class YMV>
    void axpby(const execution_space& space, const AV& a, const XMV& X, const BV& b,
               const YMV& Y) {
        static_assert(flare::is_execution_space_v<execution_space>,
                      "flare::blas::axpby: execution_space must be a valid flare "
                      "execution space.");
        static_assert(flare::is_view<XMV>::value,
                      "flare::blas::axpby: "
                      "X is not a flare::View.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename XMV::memory_space>::accessible,
                "flare::blas::axpby: XMV must be accessible from execution_space");
        static_assert(flare::is_view<YMV>::value,
                      "flare::blas::axpby: "
                      "Y is not a flare::View.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename YMV::memory_space>::accessible,
                "flare::blas::axpby: XMV must be accessible from execution_space");
        static_assert(std::is_same<typename YMV::value_type,
                              typename YMV::non_const_value_type>::value,
                      "flare::blas::axpby: Y is const.  It must be nonconst, "
                      "because it is an output argument "
                      "(we must be able to write to its entries).");
        static_assert(int(YMV::rank) == int(XMV::rank),
                      "flare::blas::axpby: "
                      "X and Y must have the same rank.");
        static_assert(YMV::rank == 1 || YMV::rank == 2,
                      "flare::blas::axpby: "
                      "XMV and YMV must either have rank 1 or rank 2.");

        // Check compatibility of dimensions at run time.
        if (X.extent(0) != Y.extent(0) || X.extent(1) != Y.extent(1)) {
            std::ostringstream os;
            os << "flare::blas::axpby: Dimensions of X and Y do not match: "
               << "X: " << X.extent(0) << " x " << X.extent(1) << ", Y: " << Y.extent(0)
               << " x " << Y.extent(1);
            flare::detail::throw_runtime_exception(os.str());
        }

        using UnifiedXLayout =
                typename flare::detail::GetUnifiedLayout<XMV>::array_layout;
        using UnifiedYLayout =
                typename flare::detail::GetUnifiedLayoutPreferring<
                        YMV, UnifiedXLayout>::array_layout;

        // Create unmanaged versions of the input Views.  XMV and YMV may be
        // rank 1 or rank 2.  AV and BV may be either rank-1 Views, or
        // scalar values.
        using XMV_Internal = flare::View<typename XMV::const_data_type,
                UnifiedXLayout, typename XMV::device_type,
                flare::MemoryTraits<flare::Unmanaged> >;
        using YMV_Internal = flare::View<typename YMV::non_const_data_type,
                UnifiedYLayout, typename YMV::device_type,
                flare::MemoryTraits<flare::Unmanaged> >;
        using AV_Internal =
                typename flare::detail::GetUnifiedScalarViewType<AV, XMV_Internal,
                        true>::type;
        using BV_Internal =
                typename flare::detail::GetUnifiedScalarViewType<BV, YMV_Internal,
                        true>::type;

        AV_Internal a_internal  = a;
        XMV_Internal X_internal = X;
        BV_Internal b_internal  = b;
        YMV_Internal Y_internal = Y;

        flare::blas::detail::Axpby<execution_space, AV_Internal, XMV_Internal, BV_Internal,
                YMV_Internal>::axpby(space, a_internal, X_internal, b_internal,
                                     Y_internal);
    }

    /// \brief Computes Y := a*X + b*Y
    ///
    /// This function is non-blocking and thread-safe
    /// The kernel is executed in the default stream/queue
    /// associated with the execution space of XMV.
    ///
    /// \tparam AV 1-D or 2-D flare::View specialization.
    /// \tparam XMV 1-D or 2-D flare::View specialization.
    /// \tparam BV 1-D or 2-D flare::View specialization.
    /// \tparam YMV 1-D or 2-D flare::View specialization. It must have
    ///   the same rank as XMV.
    ///
    /// \param a [in] view of type AV, scaling parameter for X.
    /// \param X [in] input view of type XMV.
    /// \param b [in] view of type BV, scaling parameter for Y.
    /// \param Y [in/out] view of type YMV in which the results will be stored.
    template <class AV, class XMV, class BV, class YMV>
    void axpby(const AV& a, const XMV& X, const BV& b, const YMV& Y) {
        axpby(typename XMV::execution_space{}, a, X, b, Y);
    }

    /// \brief Computes Y := a*X + Y
    ///
    /// This function is non-blocking and thread-safe
    ///
    /// \tparam execution_space a flare execution space where the kernel will run.
    /// \tparam AV 1-D or 2-D flare::View specialization.
    /// \tparam XMV 1-D or 2-D flare::View specialization.
    /// \tparam YMV 1-D or 2-D flare::View specialization. It must have
    ///   the same rank as XMV.
    ///
    /// \param space [in] the execution space instance on which the kernel will run.
    /// \param a [in] view of type AV, scaling parameter for X.
    /// \param X [in] input view of type XMV.
    /// \param Y [in/out] view of type YMV in which the results will be stored.
    template <class execution_space, class AV, class XMV, class YMV>
    void axpy(const execution_space& space, const AV& a, const XMV& X,
              const YMV& Y) {
        axpby(space, a, X,
              flare::ArithTraits<typename YMV::non_const_value_type>::one(), Y);
    }

    /// \brief Computes Y := a*X + Y
    ///
    /// This function is non-blocking and thread-safe
    /// The kernel is executed in the default stream/queue
    /// associated with the execution space of XMV.
    ///
    /// \tparam AV 1-D or 2-D flare::View specialization.
    /// \tparam XMV 1-D or 2-D flare::View specialization.
    /// \tparam YMV 1-D or 2-D flare::View specialization. It must have
    ///   the same rank as XMV.
    ///
    /// \param a [in] view of type AV, scaling parameter for X.
    /// \param X [in] input view of type XMV.
    /// \param Y [in/out] view of type YMV in which the results will be stored.
    template <class AV, class XMV, class YMV>
    void axpy(const AV& a, const XMV& X, const YMV& Y) {
        axpy(typename XMV::execution_space{}, a, X, Y);
    }

    ///
    /// Serial axpy on device
    ///
    template <class scalar_type, class XMV, class YMV>
    FLARE_FUNCTION void serial_axpy(const scalar_type alpha, const XMV X, YMV Y) {
#if (FLARE_DEBUG_LEVEL > 0)
        static_assert(flare::is_view<XMV>::value,
                "flare::blas::serial_axpy: XMV is not a flare::View");
  static_assert(flare::is_view<YMV>::value,
                "flare::blas::serial_axpy: YMV is not a flare::View");
  static_assert(XMV::rank == 1 || XMV::rank == 2,
                "flare::blas::serial_axpy: XMV must have rank 1 or 2.");
  static_assert(
      XMV::rank == YMV::rank,
      "flare::blas::serial_axpy: XMV and YMV must have the same rank.");

  if (X.extent(0) != Y.extent(0) || X.extent(1) != Y.extent(1)) {
    flare::abort("flare::blas::serial_axpy: X and Y dimensions do not match");
  }
#endif  // FLARE_DEBUG_LEVEL

        return flare::blas::detail::serial_axpy_mv(X.extent(0), X.extent(1), alpha, X.data(),
                                    Y.data(), X.stride_0(), X.stride_1(),
                                    Y.stride_0(), Y.stride_1());
    }

}  // namespace flare::blas
#endif  // FLARE_KERNEL_BLAS_AXPBY_H_
