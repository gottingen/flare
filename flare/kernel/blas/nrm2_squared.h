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

#ifndef FLARE_KERNEL_BLAS_NRM2_SQUARED_H_
#define FLARE_KERNEL_BLAS_NRM2_SQUARED_H_

#include <flare/kernel/blas/nrm2_impl.h>

namespace flare::blas {

    /// \brief Return the nrm2 of the vector x.
    ///
    /// \tparam execution_space a flare execution space where the kernel will run.
    /// \tparam XVector Type of the first vector x; a 1-D flare::Tensor.
    ///
    /// \param space [in] the execution space instance, possibly containing a
    /// stream/queue where the kernel will be executed.
    /// \param x [in] Input 1-D Tensor.
    ///
    /// \return The nrm2 product result; a single value.
    template <
            class execution_space, class XVector,
            typename std::enable_if<flare::is_execution_space<execution_space>::value,
                    int>::type = 0>
    typename flare::detail::InnerProductSpaceTraits<
            typename XVector::non_const_value_type>::mag_type
    nrm2_squared(const execution_space& space, const XVector& x) {
        static_assert(flare::is_execution_space<execution_space>::value,
                      "flare::blas::nrm2_squared: execution_space must be a valid"
                      " flare execution space");
        static_assert(flare::is_tensor<XVector>::value,
                      "flare::blas::nrm2_squared: XVector must be a flare::Tensor.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename XVector::memory_space>::accessible,
                "flare::blas::nrm2_squared: XVector must be accessible"
                " from execution_space");
        static_assert(XVector::rank == 1,
                      "flare::blas::nrm2_squared: "
                      "Both Vector inputs must have rank 1.");

        typedef typename flare::detail::InnerProductSpaceTraits<
                typename XVector::non_const_value_type>::mag_type mag_type;

        typedef flare::Tensor<
                typename XVector::const_value_type*,
                typename flare::detail::GetUnifiedLayout<XVector>::array_layout,
                typename XVector::device_type, flare::MemoryTraits<flare::Unmanaged> >
                XVector_Internal;

        typedef flare::Tensor<mag_type, default_layout, flare::HostSpace,
        flare::MemoryTraits<flare::Unmanaged> >
                RVector_Internal;

        mag_type result;
        RVector_Internal R = RVector_Internal(&result);
        XVector_Internal X = x;

        flare::blas::detail::Nrm2<execution_space, RVector_Internal, XVector_Internal>::nrm2(
                space, R, X, false);
        space.fence();
        return result;
    }

    /// \brief Return the nrm2 of the vector x.
    ///
    /// The kernel is executed in thedefault stream/queue associated
    /// with the execution space of XVector.
    ///
    /// \tparam XVector Type of the first vector x; a 1-D flare::Tensor.
    ///
    /// \param x [in] Input 1-D Tensor.
    ///
    /// \return The nrm2 product result; a single value.
    template <class XVector>
    typename flare::detail::InnerProductSpaceTraits<
            typename XVector::non_const_value_type>::mag_type
    nrm2_squared(const XVector& x) {
        return nrm2_squared(typename XVector::execution_space{}, x);
    }

    /// \brief R(i,j) = nrm2(X(i,j))
    ///
    /// Replace each entry in R with the nrm2olute value (magnitude) of the
    /// corresponding entry in X.
    /// This function is non-blocking and thread-safe
    ///
    /// \tparam execution_space a flare execution space where the kernel will run.
    /// \tparam RMV 1-D or 2-D flare::Tensor specialization.
    /// \tparam XMV 1-D or 2-D flare::Tensor specialization.  It must have
    ///   the same rank as RMV, and its entries must be assignable to
    ///   those of RMV.
    ///
    /// \param space [in] the execution space instance, possibly containing a
    /// stream/queue where the kernel will be executed.
    /// \param R [in] Output Tensor (rank 0 or 1) that holds the result.
    /// \param X [in] Input Tensor (rank 1 or 2).
    template <class execution_space, class RV, class XMV>
    void nrm2_squared(
            const execution_space& space, const RV& R, const XMV& X,
            typename std::enable_if<flare::is_tensor<RV>::value, int>::type = 0) {
        static_assert(flare::is_execution_space<execution_space>::value,
                      "flare::blas::nrm2_squared: execution_space must be a valid"
                      " flare execution space");
        static_assert(flare::is_tensor<RV>::value,
                      "flare::blas::nrm2_squared: "
                      "R is not a flare::Tensor.");
        static_assert(flare::is_tensor<XMV>::value,
                      "flare::blas::nrm2_squared: "
                      "X is not a flare::Tensor.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename XMV::memory_space>::accessible,
                "flare::blas::nrm2_squared: XVector must be accessible"
                " from execution_space");
        static_assert(std::is_same<typename RV::value_type,
                              typename RV::non_const_value_type>::value,
                      "flare::blas::nrm2_squared: R is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");
        static_assert(((RV::rank == 0) && (XMV::rank == 1)) ||
                      ((RV::rank == 1) && (XMV::rank == 2)),
                      "flare::blas::nrm2_squared: "
                      "RV and XMV must either have rank 0 and 1 or rank 1 and 2.");
        typedef typename flare::detail::InnerProductSpaceTraits<
                typename XMV::non_const_value_type>::mag_type mag_type;
        static_assert(std::is_same<typename RV::value_type, mag_type>::value,
                      "flare::blas::nrm2: R must have the magnitude type of"
                      "the xvectors value_type it is an output argument "
                      "(we have to be able to write to its entries).");

        // Check compatibility of dimensions at run time.
        if (X.extent(1) != R.extent(0)) {
            std::ostringstream os;
            os << "flare::blas::nrm2 (MV): Dimensions of R and X do not match: "
               << "R: " << R.extent(0) << ", X: " << X.extent(0) << " x "
               << X.extent(1);
            flare::detail::throw_runtime_exception(os.str());
        }

        using UnifiedXLayout =
                typename flare::detail::GetUnifiedLayout<XMV>::array_layout;
        using UnifiedRVLayout =
                typename flare::detail::GetUnifiedLayoutPreferring<
                        RV, UnifiedXLayout>::array_layout;

        // Create unmanaged versions of the input Tensors.  RV and XMV may be
        // rank 1 or rank 2.
        typedef flare::Tensor<typename RV::non_const_data_type, UnifiedRVLayout,
                typename RV::device_type,
                flare::MemoryTraits<flare::Unmanaged> >
                RV_Internal;
        typedef flare::Tensor<typename XMV::const_data_type, UnifiedXLayout,
                typename XMV::device_type,
                flare::MemoryTraits<flare::Unmanaged> >
                XMV_Internal;

        RV_Internal R_internal  = R;
        XMV_Internal X_internal = X;

        flare::blas::detail::Nrm2<execution_space, RV_Internal, XMV_Internal>::nrm2(
                space, R_internal, X_internal, false);
    }

    /// \brief R(i,j) = nrm2(X(i,j))
    ///
    /// Replace each entry in R with the nrm2olute value (magnitude) of the
    /// corresponding entry in X.
    /// This function is non-blocking and thread-safe.
    /// The kernel is executed in the default stream/queue associated
    /// with the execution space of XVM.
    ///
    /// \tparam RMV 1-D or 2-D flare::Tensor specialization.
    /// \tparam XMV 1-D or 2-D flare::Tensor specialization.  It must have
    ///   the same rank as RMV, and its entries must be assignable to
    ///   those of RMV.
    template <class RV, class XMV>
    void nrm2_squared(
            const RV& R, const XMV& X,
            typename std::enable_if<flare::is_tensor<RV>::value, int>::type = 0) {
        nrm2_squared(typename XMV::execution_space{}, R, X);
    }
}  // namespace flare::blas

#endif  // FLARE_KERNEL_BLAS_NRM2_SQUARED_H_
