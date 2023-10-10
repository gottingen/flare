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

#ifndef FLARE_KERNEL_BLAS_NRM1_H_
#define FLARE_KERNEL_BLAS_NRM1_H_

#include <flare/kernel/blas/nrm1_impl.h>

namespace flare::blas {


    /// \brief Return the nrm1 of the vector x.
    ///
    /// \tparam execution_space a flare execution space where the kernel will run.
    /// \tparam XVector Type of the first vector x; a 1-D flare::View.
    ///
    /// \param space [in] the execution space instance, possibly containing a
    /// stream/queue where the kernel will be executed.
    /// \param x [in] Input 1-D View.
    ///
    /// \return The nrm1 product result; a single value.
    template<
            class execution_space, class XVector,
            typename std::enable_if<flare::is_execution_space<execution_space>::value,
                    int>::type = 0>
    typename flare::detail::InnerProductSpaceTraits<
            typename XVector::non_const_value_type>::mag_type
    nrm1(const execution_space &space, const XVector &x) {
        static_assert(
                flare::is_execution_space<execution_space>::value,
                "flare::blas::nrm1: execution_space must be a flare::execution_space.");
        static_assert(flare::is_view<XVector>::value,
                      "flare::blas::nrm1: XVector must be a flare::View.");
        static_assert(XVector::rank == 1,
                      "flare::blas::nrm1: "
                      "Both Vector inputs must have rank 1.");
        using mag_type = typename flare::detail::InnerProductSpaceTraits<
                typename XVector::non_const_value_type>::mag_type;

        using XVector_Internal = flare::View<
                typename XVector::const_value_type *,
                typename flare::detail::GetUnifiedLayout<XVector>::array_layout,
                typename XVector::device_type, flare::MemoryTraits<flare::Unmanaged> >;

        using RVector_Internal =
                flare::View<mag_type, default_layout, flare::HostSpace,
                        flare::MemoryTraits<flare::Unmanaged> >;

        mag_type result;
        RVector_Internal R = RVector_Internal(&result);
        XVector_Internal X = x;

        flare::blas::detail::Nrm1<execution_space, RVector_Internal, XVector_Internal>::nrm1(space,
                                                                                             R, X);
        space.fence();
        return result;
    }

    /// \brief Return the nrm1 of the vector x.
    ///
    /// \tparam XVector Type of the first vector x; a 1-D flare::View.
    ///
    /// \param x [in] Input 1-D View.
    ///
    /// \return The nrm1 product result; a single value.
    template<class XVector>
    typename flare::detail::InnerProductSpaceTraits<
            typename XVector::non_const_value_type>::mag_type
    nrm1(const XVector &x) {
        return nrm1(typename XVector::execution_space{}, x);
    }

    /// \brief R(j) = nrm1(X(i,j))
    ///
    /// Replace each entry in R with the nrm1olute value (magnitude) of the
    /// corresponding entry in X.
    /// This function is non-blocking and thread-safe
    ///
    /// \tparam execution_space a flare execution space where the kernel will run.
    /// \tparam RMV 1-D or 2-D flare::View specialization.
    /// \tparam XMV 1-D or 2-D flare::View specialization.  It must have
    ///   the same rank as RMV, and its entries must be assignable to
    ///   those of RMV.
    ///
    /// \param space [in] the execution space instance, possibly containing a
    /// stream/queue where the kernel will be executed.
    /// \param R [out] Output 1-D View containing the result
    /// \param X [in] Input 1-D View.
    template<class execution_space, class RV, class XMV>
    void nrm1(const execution_space &space, const RV &R, const XMV &X,
              typename std::enable_if<flare::is_view<RV>::value, int>::type = 0) {
        static_assert(flare::is_view<RV>::value,
                      "flare::blas::nrm1: "
                      "R is not a flare::View.");
        static_assert(flare::is_view<XMV>::value,
                      "flare::blas::nrm1: "
                      "X is not a flare::View.");
        static_assert(std::is_same<typename RV::value_type,
                              typename RV::non_const_value_type>::value,
                      "flare::blas::nrm1: R is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");
        static_assert(((RV::rank == 0) && (XMV::rank == 1)) ||
                      ((RV::rank == 1) && (XMV::rank == 2)),
                      "flare::blas::nrm1: "
                      "RV and XMV must either have rank 0 and 1 or rank 1 and 2.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename XMV::memory_space>::accessible,
                "flare::blas::nrm1: execution_space cannot access data in XMV");

        typedef typename flare::detail::InnerProductSpaceTraits<
                typename XMV::non_const_value_type>::mag_type mag_type;
        static_assert(std::is_same<typename RV::value_type, mag_type>::value,
                      "flare::blas::nrm1: R must have the magnitude type of"
                      "the xvectors value_type it is an output argument "
                      "(we have to be able to write to its entries).");

        // Check compatibility of dimensions at run time.
        if (X.extent(1) != R.extent(0)) {
            std::ostringstream os;
            os << "flare::blas::nrm1 (MV): Dimensions of R and X do not match: "
               << "R: " << R.extent(0) << ", X: " << X.extent(0) << " x "
               << X.extent(1);
            flare::detail::throw_runtime_exception(os.str());
        }

        using UnifiedXLayout =
                typename flare::detail::GetUnifiedLayout<XMV>::array_layout;
        using UnifiedRVLayout =
                typename flare::detail::GetUnifiedLayoutPreferring<
                        RV, UnifiedXLayout>::array_layout;

        // Create unmanaged versions of the input Views.  RV and XMV may be
        // rank 1 or rank 2.
        typedef flare::View<typename std::conditional<
                RV::rank == 0, typename RV::non_const_value_type,
                typename RV::non_const_value_type *>::type,
                UnifiedRVLayout, typename RV::device_type,
                flare::MemoryTraits<flare::Unmanaged> >
                RV_Internal;
        typedef flare::View<
                typename std::conditional<XMV::rank == 1, typename XMV::const_value_type *,
                        typename XMV::const_value_type **>::type,
                UnifiedXLayout, typename XMV::device_type,
                flare::MemoryTraits<flare::Unmanaged> >
                XMV_Internal;

        RV_Internal R_internal = R;
        XMV_Internal X_internal = X;

        flare::blas::detail::Nrm1<execution_space, RV_Internal, XMV_Internal>::nrm1(
                space, R_internal, X_internal);
    }

    /// \brief R(j) = nrm1(X(i,j))
    ///
    /// Replace each entry in R with the nrm1olute value (magnitude) of the
    /// corresponding entry in X.
    /// This function is non-blocking and thread-safe. The kernel is executed in the
    /// default stream/queue associated with the execution space of XMV.
    ///
    /// \tparam RMV 1-D or 2-D flare::View specialization.
    /// \tparam XMV 1-D or 2-D flare::View specialization.  It must have
    ///   the same rank as RMV, and its entries must be assignable to
    ///   those of RMV.
    ///
    /// \param R [out] Output 1-D View containing the result
    /// \param X [in] Input 1-D View.
    template<class RV, class XMV>
    void nrm1(const RV &R, const XMV &X,
              typename std::enable_if<flare::is_view<RV>::value, int>::type = 0) {
        nrm1(typename XMV::execution_space{}, R, X);
    }

    /// \brief Return the nrm1 of the vector x via asum (the actual blas name).
    template<class XVector>
    typename flare::detail::InnerProductSpaceTraits<typename XVector::non_const_value_type>::mag_type
    asum(const XVector &x) {
        return nrm1(x);
    }
}  // namespace flare::blas

#endif  // FLARE_KERNEL_BLAS_NRM1_H_
