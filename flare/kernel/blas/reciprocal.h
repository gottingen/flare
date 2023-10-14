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

#ifndef FLARE_KERNEL_BLAS_RECIPROCAL_H_
#define FLARE_KERNEL_BLAS_RECIPROCAL_H_

#include <flare/kernel/blas/reciprocal_impl.h>

namespace flare::blas {

    /// \brief R(i,j) = reciprocal(X(i,j))
    /// Y(i) = 1/X(i)
    /// Replace each entry in R with the absolute value (magnitude), of the
    /// reciprocal of the corresponding entry in X.
    /// This function is non-blocking and thread-safe
    ///
    /// \tparam execution_space a flare execution space
    /// \tparam RMV 1-D or 2-D flare::Tensor specialization.
    /// \tparam XMV 1-D or 2-D flare::Tensor specialization.  It must have
    ///   the same rank as RMV, and its entries must be assignable to
    ///   those of RMV.
    ///
    /// \param space [in] an instance of execution space where the kernel will run
    /// \param R [out] a tensor of type RMV that contains the inverse of the values in
    /// X.
    /// \param X [in] a tensor of type XMV that contains the values to invert.
    template <class execution_space, class RMV, class XMV>
    void reciprocal(const execution_space& space, const RMV& R, const XMV& X) {
        static_assert(flare::is_execution_space_v<execution_space>,
                      "flare::blas::reciprocal: execution_space must be a valid "
                      "flare execition space.");
        static_assert(flare::is_tensor<RMV>::value,
                      "flare::blas::reciprocal: "
                      "R is not a flare::Tensor.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename RMV::memory_space>::accessible,
                "flare::blas::reciprocal: RMV must be accessible from execution_space");
        static_assert(flare::is_tensor<XMV>::value,
                      "flare::blas::reciprocal: "
                      "X is not a flare::Tensor.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename XMV::memory_space>::accessible,
                "flare::blas::reciprocal: XMV must be accessible from execution_space");
        static_assert(std::is_same<typename RMV::value_type,
                              typename RMV::non_const_value_type>::value,
                      "flare::blas::reciprocal: R is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");
        static_assert(int(RMV::rank) == int(XMV::rank),
                      "flare::blas::reciprocal: "
                      "R and X must have the same rank.");
        static_assert(RMV::rank == 1 || RMV::rank == 2,
                      "flare::blas::reciprocal: "
                      "RMV and XMV must either have rank 1 or rank 2.");

        // Check compatibility of dimensions at run time.
        if (X.extent(0) != R.extent(0) || X.extent(1) != R.extent(1)) {
            std::ostringstream os;
            os << "flare::blas::reciprocal (MV): Dimensions of R and X do not match: "
               << "R: " << R.extent(0) << " x " << R.extent(1) << ", X: " << X.extent(0)
               << " x " << X.extent(1);
            flare::detail::throw_runtime_exception(os.str());
        }

        // Create unmanaged versions of the input Tensors.  RMV and XMV may be
        // rank 1 or rank 2.
        typedef flare::Tensor<
                typename std::conditional<RMV::rank == 1,
                        typename RMV::non_const_value_type*,
                        typename RMV::non_const_value_type**>::type,
                typename flare::detail::GetUnifiedLayout<RMV>::array_layout,
                typename RMV::device_type, flare::MemoryTraits<flare::Unmanaged> >
                RMV_Internal;
        typedef flare::Tensor<
                typename std::conditional<XMV::rank == 1, typename XMV::const_value_type*,
                        typename XMV::const_value_type**>::type,
                typename flare::detail::GetUnifiedLayout<XMV>::array_layout,
                typename XMV::device_type, flare::MemoryTraits<flare::Unmanaged> >
                XMV_Internal;

        RMV_Internal R_internal = R;
        XMV_Internal X_internal = X;

        flare::blas::detail::Reciprocal<execution_space, RMV_Internal, XMV_Internal>::reciprocal(
                space, R_internal, X_internal);
    }

    /// \brief R(i,j) = reciprocal(X(i,j))
    ///
    /// Replace each entry in R with the absolute value (magnitude), of the
    /// reciprocal of the corresponding entry in X.
    /// This function is non-blocking and thread-safe
    /// The kernel is executed in the default stream/queue
    /// associated with the execution space of RMV.
    ///
    /// \tparam RMV 1-D or 2-D flare::Tensor specialization.
    /// \tparam XMV 1-D or 2-D flare::Tensor specialization.  It must have
    ///   the same rank as RMV, and its entries must be assignable to
    ///   those of RMV.
    template <class RMV, class XMV>
    void reciprocal(const RMV& R, const XMV& X) {
        reciprocal(typename RMV::execution_space{}, R, X);
    }
}
#endif  // FLARE_KERNEL_BLAS_RECIPROCAL_H_
