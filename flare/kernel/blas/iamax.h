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

#ifndef FLARE_KERNEL_BLAS_IAMAX_H_
#define FLARE_KERNEL_BLAS_IAMAX_H_

#include <flare/kernel/blas/iamax_impl.h>

namespace flare::blas {

    /// \brief Return the (smallest) index of the element of the maximum magnitude
    /// of the vector x.
    ///
    /// \tparam execution_space a flare execution space where the kernel will run.
    /// \tparam XVector Type of the first vector x; a 1-D flare::Tensor.
    ///
    /// \param space [in] execution space instance where the kernel will run.
    /// \param x [in] Input 1-D Tensor.
    ///
    /// \return The (smallest) index of the element of the maximum magnitude; a
    /// single value.
    ///         Note: Returned index is 1-based for compatibility with Fortran.
    template <class execution_space, class XVector,
            typename std::enable_if<flare::is_execution_space_v<execution_space>,
            int>::type = 0>
    typename XVector::size_type iamax(const execution_space& space,
                                      const XVector& x) {
        static_assert(flare::is_execution_space_v<execution_space>,
                      "flare::blas::iamax: execution_space must be a valid flare "
                      "execution space");
        static_assert(flare::is_tensor<XVector>::value,
                      "flare::blas::iamax: XVector must be a flare::Tensor.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename XVector::memory_space>::accessible,
                "flare::blas::iamax: XVector must be accessible from execution_space");
        static_assert(XVector::rank == 1,
                      "flare::blas::iamax: "
                      "Both Vector inputs must have rank 1.");

        typedef typename XVector::size_type index_type;

        typedef flare::Tensor<
                typename XVector::const_value_type*,
                typename flare::detail::GetUnifiedLayout<XVector>::array_layout,
                typename XVector::device_type, flare::MemoryTraits<flare::Unmanaged> >
                XVector_Internal;

        using layout_t = typename XVector_Internal::array_layout;

        typedef flare::Tensor<index_type, layout_t, flare::HostSpace,
        flare::MemoryTraits<flare::Unmanaged> >
                RVector_Internal;

        index_type result;
        RVector_Internal R = RVector_Internal(&result, layout_t());
        XVector_Internal X = x;

        flare::blas::detail::Iamax<execution_space, RVector_Internal, XVector_Internal>::iamax(space,
                                                                                R, X);
        space.fence();
        return result;
    }

    /// \brief Return the (smallest) index of the element of the maximum magnitude
    /// of the vector x.
    ///
    /// The kernel is executed in the default stream/queue associated
    /// with the execution space of XVector.
    ///
    /// \tparam XVector Type of the first vector x; a 1-D flare::Tensor.
    ///
    /// \param x [in] Input 1-D Tensor.
    ///
    /// \return The (smallest) index of the element of the maximum magnitude; a
    /// single value.
    ///         Note: Returned index is 1-based for compatibility with Fortran.
    template <class XVector>
    typename XVector::size_type iamax(const XVector& x) {
        return iamax(typename XVector::execution_space{}, x);
    }

    /// \brief R(j) = iamax(X(i,j))
    ///
    /// Replace each entry in R with the (smallest) index of the element of the
    /// maximum magnitude of the corresponding entry in X.
    /// This function is non-blocking and thread-safe.
    ///
    /// \tparam RMV 0-D or 1-D flare::Tensor specialization.
    /// \tparam XMV 1-D or 2-D flare::Tensor specialization.
    ///
    /// \param space [in] execution space instance where the kernel will run.
    /// \param R [out] Output Tensor (rank 0 or 1) containing the results.
    /// \param X [in] Input Tensor (rank 1 or 2).
    ///
    /// Note for TPL cuBLAS: When TPL cuBLAS iamax is used and returns result to a
    /// tensor, RMV must be 0-D tensor and XMV must be 1-D tensor.
    template <class execution_space, class RV, class XMV>
    void iamax(const execution_space& space, const RV& R, const XMV& X,
               typename std::enable_if<flare::is_tensor<RV>::value, int>::type = 0) {
        static_assert(flare::is_execution_space_v<execution_space>,
                      "flare::blas::iamax: execution_space must be a valid flare "
                      "execution space.");
        static_assert(flare::is_tensor<RV>::value,
                      "flare::blas::iamax: "
                      "R is not a flare::Tensor.");
        static_assert(flare::is_tensor<XMV>::value,
                      "flare::blas::iamax: "
                      "X is not a flare::Tensor.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename XMV::memory_space>::accessible,
                "flare::blas::iamax: XMV must be accessible from execution_space.");
        static_assert(std::is_same<typename RV::value_type,
                              typename RV::non_const_value_type>::value,
                      "flare::blas::iamax: R is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");
        static_assert(((RV::rank == 0) && (XMV::rank == 1)) ||
                      ((RV::rank == 1) && (XMV::rank == 2)),
                      "flare::blas::iamax: "
                      "RV and XMV must either have rank 0 and 1 or rank 1 and 2.");

        typedef typename XMV::size_type index_type;
        static_assert(std::is_same<typename RV::value_type, index_type>::value,
                      "flare::blas::iamax: R must have the type of"
                      "the Xvectors size_type it is an output argument "
                      "(we have to be able to write to its entries).");

        // Check compatibility of dimensions at run time.
        if (X.extent(1) != R.extent(0)) {
            std::ostringstream os;
            os << "flare::blas::iamax (MV): Dimensions of R and X do not match: "
               << "R: " << R.extent(0) << ", X: " << X.extent(0) << " x "
               << X.extent(1);
            flare::detail::throw_runtime_exception(os.str());
        }

        using UnifiedXLayout =
                typename flare::detail::GetUnifiedLayout<XMV>::array_layout;
        using UnifiedRVLayout =
                typename flare::detail::GetUnifiedLayoutPreferring<
                        RV, UnifiedXLayout>::array_layout;

        // Create unmanaged versions of the input Tensors.  RV may be rank 0 or rank 1.
        // XMV may be rank 1 or rank 2.
        typedef flare::Tensor<typename std::conditional<
                RV::rank == 0, typename RV::non_const_value_type,
                typename RV::non_const_value_type*>::type,
                UnifiedRVLayout,
                typename std::conditional<
                        std::is_same<typename RV::device_type::memory_space,
                                flare::HostSpace>::value,
                        flare::HostSpace, typename RV::device_type>::type,
                flare::MemoryTraits<flare::Unmanaged> >
                RV_Internal;
        typedef flare::Tensor<
                typename std::conditional<XMV::rank == 1, typename XMV::const_value_type*,
                        typename XMV::const_value_type**>::type,
                UnifiedXLayout, typename XMV::device_type,
                flare::MemoryTraits<flare::Unmanaged> >
                XMV_Internal;

        RV_Internal R_internal  = R;
        XMV_Internal X_internal = X;

        flare::blas::detail::Iamax<execution_space, RV_Internal, XMV_Internal>::iamax(space, R_internal, X_internal);
    }

    /// \brief R(j) = iamax(X(i,j))
    ///
    /// Replace each entry in R with the (smallest) index of the element of the
    /// maximum magnitude of the corresponding entry in X.
    /// This function is non-blocking and thread-safe.
    /// The kernel is executed in the default stream/queue associated
    /// with the execution space of XVector.
    ///
    /// \tparam RMV 0-D or 1-D flare::Tensor specialization.
    /// \tparam XMV 1-D or 2-D flare::Tensor specialization.
    ///
    /// Note for TPL cuBLAS: When TPL cuBLAS iamax is used and returns result to a
    /// tensor, RMV must be 0-D tensor and XMV must be 1-D tensor.
    template <class RV, class XMV>
    void iamax(const RV& R, const XMV& X,
               typename std::enable_if<flare::is_tensor<RV>::value, int>::type = 0) {
        iamax(typename XMV::execution_space{}, R, X);
    }

}  // namespace flare::blas

#endif  // FLARE_KERNEL_BLAS_IAMAX_H_
