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

#ifndef FLARE_KERNEL_BLAS_SCAL_H_
#define FLARE_KERNEL_BLAS_SCAL_H_

#include <flare/kernel/blas/scal_impl.h>

namespace flare::blas {


    /// \brief Computes R := alpha*X
    ///
    /// This function is non-blocking and thread-safe
    ///
    /// \tparam execution_space a flare execution space where the kernel will run.
    /// \tparam RMV 1-D or 2-D flare::View specialization.
    /// \tparam XMV 1-D or 2-D flare::View specialization. It must have
    ///   the same rank as RMV.
    /// \tparam AV 1-D or 2-D flare::View specialization.
    ///
    /// \param space [in] the execution space instance on which the kernel will run.
    /// \param R [in/out] view of type RMV in which the results will be stored.
    /// \param a [in] view of type AV, scaling parameter for X.
    /// \param X [in] input view of type XMV.
    template <class execution_space, class RMV, class AV, class XMV>
    void scal(const execution_space& space, const RMV& R, const AV& a,
              const XMV& X) {
        static_assert(flare::is_execution_space_v<execution_space>,
                      "flare::blas::scal: execution_space must be a valid flare "
                      "execution space");
        static_assert(flare::is_view<RMV>::value,
                      "flare::blas::scal: "
                      "R is not a flare::View.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename RMV::memory_space>::accessible,
                "flare::blas::scal: RMV must be accessible from execution_space.");
        static_assert(flare::is_view<XMV>::value,
                      "flare::blas::scal: "
                      "X is not a flare::View.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename XMV::memory_space>::accessible,
                "flare::blas::scal: XMV must be accessible from execution_space");
        static_assert(
                flare::SpaceAccessibility<typename RMV::memory_space,
                        typename XMV::memory_space>::assignable,
                "flare::blas::scal: XMV must be assignable to RMV");
        static_assert(std::is_same<typename RMV::value_type,
                              typename RMV::non_const_value_type>::value,
                      "flare::blas::scal: R is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");
        static_assert((int)RMV::rank == (int)XMV::rank,
                      "flare::blas::scal: "
                      "R and X must have the same rank.");
        static_assert(RMV::rank == 1 || RMV::rank == 2,
                      "flare::blas::scal: "
                      "RMV and XMV must either have rank 1 or rank 2.");

        // Check compatibility of dimensions at run time.
        if (X.extent(0) != R.extent(0) || X.extent(1) != R.extent(1)) {
            std::ostringstream os;
            os << "flare::blas::scal: Dimensions of R and X do not match: "
               << "R: " << R.extent(0) << " x " << R.extent(1) << ", X: " << X.extent(0)
               << " x " << X.extent(1);
            flare::detail::throw_runtime_exception(os.str());
        }

        using UnifiedRLayout =
                typename flare::detail::GetUnifiedLayout<RMV>::array_layout;
        using UnifiedXLayout =
                typename flare::detail::GetUnifiedLayoutPreferring<
                        XMV, UnifiedRLayout>::array_layout;

        // Create unmanaged versions of the input Views.  RMV and XMV may be
        // rank 1 or rank 2.  AV may be either a rank-1 View, or a scalar
        // value.
        using RMV_Internal = flare::View<typename RMV::non_const_data_type,
                UnifiedRLayout, typename RMV::device_type,
                flare::MemoryTraits<flare::Unmanaged> >;
        using XMV_Internal = flare::View<typename XMV::const_data_type,
                UnifiedXLayout, typename XMV::device_type,
                flare::MemoryTraits<flare::Unmanaged> >;
        using AV_Internal =
                typename flare::detail::GetUnifiedScalarViewType<AV, XMV_Internal,
                        true>::type;

        RMV_Internal R_internal = R;
        AV_Internal a_internal  = a;
        XMV_Internal X_internal = X;

        flare::blas::detail::Scal<execution_space, RMV_Internal, AV_Internal, XMV_Internal>::scal(
                space, R_internal, a_internal, X_internal);
    }

    /// \brief Computes R := alpha*X
    ///
    /// This function is non-blocking and thread-safe
    /// The kernel is executed in the default stream/queue
    /// associated with the execution space of YMV.
    ///
    /// \tparam RMV 1-D or 2-D flare::View specialization.
    /// \tparam XMV 1-D or 2-D flare::View specialization. It must have
    ///   the same rank as RMV.
    /// \tparam AV 1-D or 2-D flare::View specialization.
    ///
    /// \param R [in/out] view of type RMV in which the results will be stored.
    /// \param a [in] view of type AV, scaling parameter for X.
    /// \param X [in] input view of type XMV.
    template <class RMV, class AV, class XMV>
    void scal(const RMV& R, const AV& a, const XMV& X) {
        scal(typename RMV::execution_space{}, R, a, X);
    }

    ///
    /// Serial Scale
    ///

    struct SerialScale {
        template <typename ScalarType, typename AViewType>
        FLARE_INLINE_FUNCTION static int invoke(const ScalarType alpha,
                                                 const AViewType& A) {
            return flare::blas::detail::SerialScaleInternal::invoke(
                    A.extent(0), A.extent(1), alpha, A.data(), A.stride_0(), A.stride_1());
        }
    };

    ///
    /// Team Scale
    ///

    template <typename MemberType>
    struct TeamScale {
        template <typename ScalarType, typename AViewType>
        FLARE_INLINE_FUNCTION static int invoke(const MemberType& member,
                                                 const ScalarType alpha,
                                                 const AViewType& A) {
            return flare::blas::detail::TeamScaleInternal::invoke(member, A.extent(0), A.extent(1),
                                                   alpha, A.data(), A.stride_0(),
                                                   A.stride_1());
        }
    };

    ///
    /// TeamVector Scale
    ///

    template <typename MemberType>
    struct TeamVectorScale {
        template <typename ScalarType, typename AViewType>
        FLARE_INLINE_FUNCTION static int invoke(const MemberType& member,
                                                 const ScalarType alpha,
                                                 const AViewType& A) {
            return flare::blas::detail::TeamVectorScaleInternal::invoke(member, A.extent(0),
                                                         A.extent(1), alpha, A.data(),
                                                         A.stride_0(), A.stride_1());
        }
    };
    template <class TeamType, class RVector, class XVector>
    void FLARE_INLINE_FUNCTION
    team_scal(const TeamType& team, const RVector& r,
         const typename XVector::non_const_value_type& a, const XVector& x) {
        return flare::blas::detail::TeamScal<TeamType, RVector, XVector>::team_scal(team, r, a, x);
    }

}  // namespace flare::blas

#endif  // FLARE_KERNEL_BLAS_SCAL_H_
