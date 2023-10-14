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

#ifndef FLARE_KERNEL_BLAS_NRMINF_H_
#define FLARE_KERNEL_BLAS_NRMINF_H_

#include <flare/kernel/blas/nrminf_impl.h>

namespace flare::blas {


    /// \brief Return the nrminf := Max(abs(X(i))) of the vector x.
    ///
    /// \tparam execution_space The execution space in which the kernel will run.
    /// \tparam XVector Type of the first vector x; a 1-D flare::Tensor.
    ///
    /// \param space [in] an execution space instance that can specify computing
    ///                   resources to be used, for instance a stream or queue.
    /// \param x [in] Input 1-D Tensor.
    ///
    /// \return The nrminf product result; a single value.
    template <
            class execution_space, class XVector,
            typename std::enable_if<flare::is_execution_space<execution_space>::value,
                    int>::type = 0>
    typename flare::detail::InnerProductSpaceTraits<
            typename XVector::non_const_value_type>::mag_type
    nrminf(const execution_space& space, const XVector& x) {
        static_assert(flare::is_tensor<XVector>::value,
                      "flare::blas::nrminf: XVector must be a flare::Tensor.");
        static_assert(XVector::rank == 1,
                      "flare::blas::nrminf: "
                      "Both Vector inputs must have rank 1.");
        typedef typename flare::detail::InnerProductSpaceTraits<
                typename XVector::non_const_value_type>::mag_type mag_type;

        typedef flare::Tensor<
                typename XVector::const_value_type*,
                typename flare::detail::GetUnifiedLayout<XVector>::array_layout,
                typename XVector::device_type, flare::MemoryTraits<flare::Unmanaged> >
                XVector_Internal;

        using layout_t = typename XVector_Internal::array_layout;

        typedef flare::Tensor<mag_type, layout_t, flare::HostSpace,
        flare::MemoryTraits<flare::Unmanaged> >
                RVector_Internal;

        mag_type result;
        RVector_Internal R = RVector_Internal(&result, layout_t());
        XVector_Internal X = x;

        flare::blas::detail::NrmInf<execution_space, RVector_Internal, XVector_Internal>::nrminf(
                space, R, X);
        space.fence();
        return result;
    }

    /// \brief Return the nrminf of the vector x.
    ///
    /// \tparam XVector Type of the first vector x; a 1-D flare::Tensor.
    ///
    /// \param x [in] Input 1-D Tensor.
    ///
    /// \return The nrminf product result; a single value.
    template <class XVector>
    typename flare::detail::InnerProductSpaceTraits<
            typename XVector::non_const_value_type>::mag_type
    nrminf(const XVector& x) {
        return nrminf(typename XVector::execution_space{}, x);
    }

    /// \brief R(j) = nrminf(X(i,j))
    ///
    /// Replace each entry in R with the nrminfolute value (magnitude) of the
    /// corresponding entry in X.
    ///
    /// \tparam execution_space, the execution space in which the kernel will run.
    /// \tparam RMV 1-D or 2-D flare::Tensor specialization.
    /// \tparam XMV 1-D or 2-D flare::Tensor specialization.  It must have
    ///   the same rank as RMV, and its entries must be assignable to
    ///   those of RMV.
    template <class execution_space, class RV, class XMV>
    void nrminf(
            const execution_space& space, const RV& R, const XMV& X,
            typename std::enable_if<flare::is_tensor<RV>::value, int>::type = 0) {
        static_assert(flare::is_execution_space<execution_space>::value,
                      "flare::blas::nrminf: space is not an execution space instance");
        static_assert(flare::is_tensor<RV>::value,
                      "flare::blas::nrminf: "
                      "R is not a flare::Tensor.");
        static_assert(flare::is_tensor<XMV>::value,
                      "flare::blas::nrminf: "
                      "X is not a flare::Tensor.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename XMV::memory_space>::accessible,
                "flare::blas::nrminf: X is not accessible from execution_space");
        static_assert(std::is_same<typename RV::value_type,
                              typename RV::non_const_value_type>::value,
                      "flare::blas::nrminf: R is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");
        static_assert(((RV::rank == 0) && (XMV::rank == 1)) ||
                      ((RV::rank == 1) && (XMV::rank == 2)),
                      "flare::blas::nrminf: "
                      "RV and XMV must either have rank 0 and 1 or rank 1 and 2.");

        typedef typename flare::detail::InnerProductSpaceTraits<
                typename XMV::non_const_value_type>::mag_type mag_type;
        static_assert(std::is_same<typename RV::value_type, mag_type>::value,
                      "flare::blas::nrminf: R must have the magnitude type of"
                      "the xvectors value_type it is an output argument "
                      "(we have to be able to write to its entries).");

        // Check compatibility of dimensions at run time.
        if (X.extent(1) != R.extent(0)) {
            std::ostringstream os;
            os << "flare::blas::nrminf (MV): Dimensions of R and X do not match: "
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
        typedef flare::Tensor<typename std::conditional<
                RV::rank == 0, typename RV::non_const_value_type,
                typename RV::non_const_value_type*>::type,
                UnifiedRVLayout, typename RV::device_type,
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

        flare::blas::detail::NrmInf<execution_space, RV_Internal, XMV_Internal>::nrminf(
                space, R_internal, X_internal);
    }

    /// \brief R(j) = nrminf(X(i,j))
    ///
    /// Replace each entry in R with the nrminfolute value (magnitude) of the
    /// corresponding entry in X.
    ///
    /// \tparam RMV 1-D or 2-D flare::Tensor specialization.
    /// \tparam XMV 1-D or 2-D flare::Tensor specialization.  It must have
    ///   the same rank as RMV, and its entries must be assignable to
    ///   those of RMV.
    template <class RV, class XMV>
    void nrminf(
            const RV& R, const XMV& X,
            typename std::enable_if<flare::is_tensor<RV>::value, int>::type = 0) {
        nrminf(typename XMV::execution_space{}, R, X);
    }


}  // namespace flare::blas

#endif  // FLARE_KERNEL_BLAS_NRMINF_H_
