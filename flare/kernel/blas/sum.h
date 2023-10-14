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

#ifndef FLARE_KERNEL_BLAS_SUM_H_
#define FLARE_KERNEL_BLAS_SUM_H_

#include <flare/kernel/blas/sum_impl.h>

namespace flare::blas {

    /// \brief Return the sum of the vector x.
    ///
    /// \tparam execution_space a flare execution space where the kernel will run.
    /// \tparam XVector Type of the first vector x; a 1-D flare::Tensor.
    ///
    /// \param space [in] execution space instance where the kernel will run.
    /// \param x [in] Input 1-D Tensor.
    ///
    /// \return The sum product result; a single value.
    template <class execution_space, class XVector,
            typename std::enable_if<flare::is_execution_space_v<execution_space>,
            int>::type = 0>
    typename XVector::non_const_value_type sum(const execution_space& space,
                                               const XVector& x) {
        static_assert(flare::is_execution_space_v<execution_space>,
                      "flare::blas::sum: execution_space must be a valid flare "
                      "execution space");
        static_assert(flare::is_tensor<XVector>::value,
                      "flare::blas::sum: XVector must be a flare::Tensor.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename XVector::memory_space>::accessible,
                "flare::blas::sum: XVector must be accessible from execution_space.");
        static_assert(XVector::rank == 1,
                      "flare::blas::sum: "
                      "Both Vector inputs must have rank 1.");

        using XVector_Internal = flare::Tensor<
                typename XVector::const_value_type*,
                typename flare::detail::GetUnifiedLayout<XVector>::array_layout,
                typename XVector::device_type, flare::MemoryTraits<flare::Unmanaged> >;

        using layout_t = typename XVector_Internal::array_layout;

        using RVector_Internal =
                flare::Tensor<typename XVector::non_const_value_type, layout_t,
                        flare::HostSpace, flare::MemoryTraits<flare::Unmanaged> >;

        typename XVector::non_const_value_type result;
        RVector_Internal R = RVector_Internal(&result, layout_t());
        XVector_Internal X = x;

        flare::blas::detail::Sum<execution_space, RVector_Internal, XVector_Internal>::sum(space, R,
                                                                            X);
        space.fence();
        return result;
    }

    /// \brief Return the sum of the vector x.
    ///
    /// The kernel is executed in the default stream/queue associated
    /// with the execution space of XVector.
    ///
    /// \tparam XVector Type of the first vector x; a 1-D flare::Tensor.
    ///
    /// \param x [in] Input 1-D Tensor.
    ///
    /// \return The sum product result; a single value.
    template <class XVector>
    typename XVector::non_const_value_type sum(const XVector& x) {
        return sum(typename XVector::execution_space{}, x);
    }

    /// \brief R(j) = sum(X(i,j))
    ///
    /// Replace each entry in R with the sumolute value (magnitude) of the
    /// corresponding entry in X.
    /// This function is non-blocking and thread-safe.
    ///
    /// \tparam execution_space a flare execution space where the kernel will run.
    /// \tparam RMV 1-D or 2-D flare::Tensor specialization.
    /// \tparam XMV 1-D or 2-D flare::Tensor specialization.  It must have
    ///   the same rank as RMV, and its entries must be assignable to
    ///   those of RMV.
    ///
    /// \param space [in] execution space instance where the kernel will run.
    /// \param R [out] Output Tensor (rank 0 or 1) containing the results.
    /// \param X [in] Input Tensor (rank 1 or 2).
    template <class execution_space, class RV, class XMV>
    void sum(const execution_space& space, const RV& R, const XMV& X,
             typename std::enable_if<flare::is_tensor<RV>::value, int>::type = 0) {
        static_assert(flare::is_execution_space_v<execution_space>,
                      "flare::blas::sum: execution_space must be a valid flare "
                      "execution space.");
        static_assert(flare::is_tensor<RV>::value,
                      "flare::blas::sum: "
                      "R is not a flare::Tensor.");
        static_assert(flare::is_tensor<XMV>::value,
                      "flare::blas::sum: "
                      "X is not a flare::Tensor.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename XMV::memory_space>::accessible,
                "flare::blas::sum: XMV must be accessible from execution_space.");
        static_assert(std::is_same<typename RV::value_type,
                              typename RV::non_const_value_type>::value,
                      "flare::blas::sum: R is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");
        static_assert(((RV::rank == 0) && (XMV::rank == 1)) ||
                      ((RV::rank == 1) && (XMV::rank == 2)),
                      "flare::blas::sum: "
                      "RV and XMV must either have rank 0 and 1 or rank 1 and 2.");

        // Check compatibility of dimensions at run time.
        if (X.extent(1) != R.extent(0)) {
            std::ostringstream os;
            os << "flare::blas::sum (MV): Dimensions of R and X do not match: "
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

        flare::blas::detail::Sum<execution_space, RV_Internal, XMV_Internal>::sum(space, R_internal,
                                                                   X_internal);
    }

    /// \brief R(j) = sum(X(i,j))
    ///
    /// Replace each entry in R with the sumolute value (magnitude) of the
    /// corresponding entry in X.
    /// This function is non-blocking and thread-safe.
    /// The kernel is executed in the default stream/queue associated
    /// with the execution space of XVM.
    ///
    /// \tparam RMV 1-D or 2-D flare::Tensor specialization.
    /// \tparam XMV 1-D or 2-D flare::Tensor specialization.  It must have
    ///   the same rank as RMV, and its entries must be assignable to
    ///   those of RMV.
    ///
    /// \param R [out] Output Tensor (rank 0 or 1) containing the results.
    /// \param X [in] Input Tensor (rank 1 or 2).
    template <class RV, class XMV>
    void sum(const RV& R, const XMV& X,
             typename std::enable_if<flare::is_tensor<RV>::value, int>::type = 0) {
        sum(typename XMV::execution_space{}, R, X);
    }

}  // namespace flare::blas

#endif  // FLARE_KERNEL_BLAS_SUM_H_
