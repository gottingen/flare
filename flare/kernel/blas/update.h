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


#ifndef FLARE_KERNEL_BLAS_UPDATE_H_
#define FLARE_KERNEL_BLAS_UPDATE_H_

#include <flare/kernel/blas/update_impl.h>
#include <flare/core/layout_utility.h>

namespace flare::blas {

    /// \brief Compute Z := alpha*X + beta*Y + gamma*Z.
    ///
    /// This function is non-blocking and thread-safe
    ///
    /// \tparam execution_space a flare execution space where the kernel will run.
    /// \tparam XMV 1-D or 2-D flare::View specialization.
    /// \tparam YMV 1-D or 2-D flare::View specialization.  It must have
    ///   the same rank as XMV.
    /// \tparam ZMV 1-D or 2-D flare::View specialization.  It must have
    ///   the same rank as XMV and YMV, and it must make sense to add up
    ///   the entries of XMV and YMV and assign them to the entries of
    ///   ZMV.
    ///
    /// \param space [in] the execution space instance on which the kernel will run.
    /// \param alpha [in] scaling parameter for X
    /// \param X [in] input view of type XMV
    /// \param beta [in] scaling parameter for Y
    /// \param Y [in] input view of type YMV
    /// \param gamma [in] scaling parameter for Z
    /// \param Z [in/out] view of type ZMV in which the results will be stored.
    template <class execution_space, class XMV, class YMV, class ZMV>
    void update(const execution_space& space,
                const typename XMV::non_const_value_type& alpha, const XMV& X,
                const typename YMV::non_const_value_type& beta, const YMV& Y,
                const typename ZMV::non_const_value_type& gamma, const ZMV& Z) {
        static_assert(flare::is_execution_space_v<execution_space>,
                      "flare::blas::update: execution_space must be a valid flare "
                      "execution space.");
        static_assert(flare::is_view<XMV>::value,
                      "flare::blas::update: "
                      "X is not a flare::View.");
        static_assert(flare::is_view<YMV>::value,
                      "flare::blas::update: "
                      "Y is not a flare::View.");
        static_assert(flare::is_view<ZMV>::value,
                      "flare::blas::update: "
                      "Z is not a flare::View.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename XMV::memory_space>::accessible,
                "flare::blas::update: XMV must be accessible from execution_space.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename YMV::memory_space>::accessible,
                "flare::blas::update: YMV must be accessible from execution_space.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename ZMV::memory_space>::accessible,
                "flare::blas::update: ZMV must be accessible from execution_space.");
        static_assert(std::is_same<typename ZMV::value_type,
                              typename ZMV::non_const_value_type>::value,
                      "flare::blas::update: Z is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");
        static_assert(int(ZMV::rank) == int(XMV::rank),
                      "flare::blas::update: "
                      "X and Z must have the same rank.");
        static_assert(int(ZMV::rank) == int(YMV::rank),
                      "flare::blas::update: "
                      "Y and Z must have the same rank.");
        static_assert(ZMV::rank == 1 || ZMV::rank == 2,
                      "flare::blas::update: "
                      "XMV, YMV, and ZMV must either have rank 1 or rank 2.");

        // Check compatibility of dimensions at run time.
        if (X.extent(0) != Y.extent(0) || X.extent(1) != Y.extent(1) ||
            X.extent(0) != Z.extent(0) || X.extent(1) != Z.extent(1)) {
            std::ostringstream os;
            os << "flare::blas::update (MV): Dimensions of X, Y, and Z do not match: "
               << "Z: " << Z.extent(0) << " x " << Z.extent(1) << ", X: " << X.extent(0)
               << " x " << X.extent(1) << ", Y: " << Y.extent(0) << " x "
               << Y.extent(1);
            flare::detail::throw_runtime_exception(os.str());
        }

        // Create unmanaged versions of the input Views.  XMV, YMV, and ZMV
        // may be rank 1 or rank 2, but they must all have the same rank.

        using XMV_Internal = flare::View<
                typename std::conditional<XMV::rank == 1, typename XMV::const_value_type*,
                        typename XMV::const_value_type**>::type,
                typename flare::detail::GetUnifiedLayout<XMV>::array_layout,
                typename XMV::device_type, flare::MemoryTraits<flare::Unmanaged> >;

        using YMV_Internal = flare::View<
                typename std::conditional<YMV::rank == 1, typename YMV::const_value_type*,
                        typename YMV::const_value_type**>::type,
                typename flare::detail::GetUnifiedLayout<YMV>::array_layout,
                typename YMV::device_type, flare::MemoryTraits<flare::Unmanaged> >;

        using ZMV_Internal = flare::View<
                typename std::conditional<ZMV::rank == 1,
                        typename ZMV::non_const_value_type*,
                        typename ZMV::non_const_value_type**>::type,
                typename flare::detail::GetUnifiedLayout<ZMV>::array_layout,
                typename ZMV::device_type, flare::MemoryTraits<flare::Unmanaged> >;

        XMV_Internal X_internal = X;
        YMV_Internal Y_internal = Y;
        ZMV_Internal Z_internal = Z;

        flare::blas::detail::Update<execution_space, XMV_Internal, YMV_Internal,
                ZMV_Internal>::update(space, alpha, X_internal, beta, Y_internal,
                                      gamma, Z_internal);
    }

    /// \brief Compute Z := alpha*X + beta*Y + gamma*Z.
    ///
    /// This function is non-blocking and thread-safe
    /// The kernel is executed in the default stream/queue
    /// associated with the execution space of ZMV.
    ///
    /// \tparam XMV 1-D or 2-D flare::View specialization.
    /// \tparam YMV 1-D or 2-D flare::View specialization.  It must have
    ///   the same rank as XMV.
    /// \tparam ZMV 1-D or 2-D flare::View specialization.  It must have
    ///   the same rank as XMV and YMV, and it must make sense to add up
    ///   the entries of XMV and YMV and assign them to the entries of
    ///   ZMV.
    ///
    /// \param alpha [in] scaling parameter for X
    /// \param X [in] input view of type XMV
    /// \param beta [in] scaling parameter for Y
    /// \param Y [in] input view of type YMV
    /// \param gamma [in] scaling parameter for Z
    /// \param Z [in/out] view of type ZMV in which the results will be stored.
    template <class XMV, class YMV, class ZMV>
    void update(const typename XMV::non_const_value_type& alpha, const XMV& X,
                const typename YMV::non_const_value_type& beta, const YMV& Y,
                const typename ZMV::non_const_value_type& gamma, const ZMV& Z) {
        update(typename ZMV::execution_space{}, alpha, X, beta, Y, gamma, Z);
    }

}  // namespace flare::blas

#endif  // FLARE_KERNEL_BLAS_UPDATE_H_
