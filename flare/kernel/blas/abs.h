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

#ifndef FLARE_KERNEL_BLAS_ABS_H_
#define FLARE_KERNEL_BLAS_ABS_H_

#include <flare/core.h>
#include <flare/kernel/common/helper.h>
#include <flare/kernel/blas/abs_impl.h>
namespace flare::blas {

    /// \brief R(i,j) = abs(X(i,j))
    ///
    /// Non-blocking function to replace each entry in R with the absolute value
    /// (magnitude) of the corresponding entry in X.
    ///
    /// \tparam execution_space a flare execution space to run the kernels on.
    /// \tparam RMV 1-D or 2-D flare::View specialization.
    /// \tparam XMV 1-D or 2-D flare::View specialization.  It must have
    ///   the same rank as RMV, and its entries must be assignable to
    ///   those of RMV.
    ///
    /// \param space [in] an execution_space instance where the kernel will run.
    /// \param R [out] view of type RMV that contains the absolute value X on
    /// output.
    /// \param X [in] view of type XMV.
    template <class execution_space, class RMV, class XMV>
    void abs(const execution_space& space, const RMV& R, const XMV& X) {
        static_assert(flare::is_execution_space_v<execution_space>,
                      "flare::blas::abs: execution_space must be a valid flare "
                      "execution space.");
        static_assert(flare::is_view<RMV>::value,
                      "flare::blas::abs: "
                      "R is not a flare::View.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename RMV::memory_space>::accessible,
                "flare::blas::abs: RMV must be accessible from execution space");
        static_assert(flare::is_view<XMV>::value,
                      "flare::blas::abs: "
                      "X is not a flare::View.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename XMV::memory_space>::accessible,
                "flare::blas::abs: XMV must be accessible from execution space");
        static_assert(std::is_same<typename RMV::value_type,
                              typename RMV::non_const_value_type>::value,
                      "flare::blas::abs: R is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");
        static_assert(int(RMV::rank) == int(XMV::rank),
                      "flare::blas::abs: "
                      "R and X must have the same rank.");
        static_assert(RMV::rank == 1 || RMV::rank == 2,
                      "flare::blas::abs: "
                      "RMV and XMV must either have rank 1 or rank 2.");

        // Check compatibility of dimensions at run time.
        if (X.extent(0) != R.extent(0) || X.extent(1) != R.extent(1)) {
            std::ostringstream os;
            os << "flare::blas::abs (MV): Dimensions of R and X do not match: "
               << "R: " << R.extent(0) << " x " << R.extent(1) << ", X: " << X.extent(0)
               << " x " << X.extent(1);
            flare::detail::throw_runtime_exception(os.str());
        }

        // Create unmanaged versions of the input Views.  RMV and XMV may be
        // rank 1 or rank 2.
        using RMV_Internal = flare::View<
                typename std::conditional<RMV::rank == 1,
                        typename RMV::non_const_value_type*,
                        typename RMV::non_const_value_type**>::type,
                typename flare::detail::GetUnifiedLayout<RMV>::array_layout,
                typename RMV::device_type, flare::MemoryTraits<flare::Unmanaged> >;
        using XMV_Internal = flare::View<
                typename std::conditional<XMV::rank == 1, typename XMV::const_value_type*,
                        typename XMV::const_value_type**>::type,
                typename flare::detail::GetUnifiedLayout<XMV>::array_layout,
                typename XMV::device_type, flare::MemoryTraits<flare::Unmanaged> >;

        RMV_Internal R_internal = R;
        XMV_Internal X_internal = X;

        flare::blas::detail::Abs<execution_space, RMV_Internal, XMV_Internal>::abs(space, R_internal,
                                                                    X_internal);
    }

    /// \brief R(i,j) = abs(X(i,j))
    ///
    /// Non-blocking function to replace each entry in R with the absolute value
    /// (magnitude) of the corresponding entry in X. The kernel is executed in the
    /// default stream/queue associated with the execution space of RMV.
    ///
    /// \tparam RMV 1-D or 2-D flare::View specialization.
    /// \tparam XMV 1-D or 2-D flare::View specialization.  It must have
    ///   the same rank as RMV, and its entries must be assignable to
    ///   those of RMV.
    ///
    /// \param R [out] view of type RMV that contains the absolute value X on
    /// output.
    /// \param X [in] view of type XMV.
    template <class RMV, class XMV>
    void abs(const RMV& R, const XMV& X) {
        abs(typename RMV::execution_space{}, R, X);
    }
}  // namespace flare::blas


#endif  // FLARE_KERNEL_BLAS_ABS_H_
