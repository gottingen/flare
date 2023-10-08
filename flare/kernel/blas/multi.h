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


#ifndef FLARE_KERNEL_BLAS_MULTI_H_
#define FLARE_KERNEL_BLAS_MULTI_H_
#include <flare/kernel/blas/multi_impl.h>
#include <flare/kernel/common/helper.h>

namespace flare::blas {

    /// \brief Element wise multiplication of two vectors:
    ///        Y[i] = gamma * Y[i] + alpha * A[i] * X[i]
    ///
    /// This function is non-blocking and thread-safe
    ///
    /// \tparam execution_type a flare execution space type.
    /// \tparam YMV Type of the first vector Y; a 1-D or 2-D flare::View.
    /// \tparam AV  Type of the second vector A; a 1-D flare::View.
    /// \tparam XMV Type of the third vector X; a 1-D or 2-D flare::View.
    ///
    /// \param space [in] An instance of execution_space on which the kernel
    ///                   will run (it may specify an execution stream/queue).
    /// \param gamma [in] The scalar to apply to Y.
    /// \param Y [in/out] The Y vector.
    /// \param alpha [in] The scalar to apply to A.
    /// \param A [in]     The vector to apply to X.
    /// \param X [in]     The X vector.
    template <class execution_space, class YMV, class AV, class XMV>
    void mult(const execution_space& space, typename YMV::const_value_type& gamma,
              const YMV& Y, typename AV::const_value_type& alpha, const AV& A,
              const XMV& X) {
        static_assert(flare::is_execution_space_v<execution_space>,
                      "flare::blas::mult: execution_space must be a valid flare "
                      "execution space.");
        static_assert(flare::is_view<YMV>::value,
                      "flare::blas::mult: "
                      "Y is not a flare::View.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename YMV::memory_space>::accessible,
                "flare::blas::mult: YMV must be accessible from execution_space.");
        static_assert(flare::is_view<AV>::value,
                      "flare::blas::mult: "
                      "A is not a flare::View.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename AV::memory_space>::accessible,
                "flare::blas::mult: AV must be accessible from execution_space.");
        static_assert(flare::is_view<XMV>::value,
                      "flare::blas::mult: "
                      "X is not a flare::View.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename XMV::memory_space>::accessible,
                "flare::blas::mult: AV must be accessible from execution_space.");
        static_assert(std::is_same<typename YMV::value_type, typename YMV::non_const_value_type>::value,
                      "flare::blas::mult: Y is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");
        static_assert(
                (XMV::rank == 1 && YMV::rank == 1) || (XMV::rank == 2 && YMV::rank == 2),
                "flare::blas::mult: Y and X must be either both rank 1, "
                "or both rank 2.");
        static_assert(AV::rank == 1, "flare::blas::mult: A must have rank 1.");

        // Check compatibility of dimensions at run time.
        if (Y.extent(0) != A.extent(0) || Y.extent(0) != X.extent(0) ||
            Y.extent(1) != X.extent(1)) {
            std::ostringstream os;
            os << "flare::blas::mult: Dimensions do not match: "
               << "Y: " << Y.extent(0) << " x " << Y.extent(1) << ", A: " << A.extent(0)
               << " x " << A.extent(0) << ", X: " << X.extent(0) << " x "
               << X.extent(1);
            flare::detail::throw_runtime_exception(os.str());
        }

        using YUnifiedLayout =
                typename flare::detail::GetUnifiedLayout<YMV>::array_layout;
        using AUnifiedLayout =
                typename flare::detail::GetUnifiedLayoutPreferring<
                        AV, YUnifiedLayout>::array_layout;
        using XUnifiedLayout =
                typename flare::detail::GetUnifiedLayoutPreferring<
                        XMV, YUnifiedLayout>::array_layout;

        // Create unmanaged versions of the input Views.
        typedef flare::View<typename YMV::non_const_data_type, YUnifiedLayout,
                typename YMV::device_type,
                flare::MemoryTraits<flare::Unmanaged> >
                YMV_Internal;
        typedef flare::View<typename AV::const_value_type*, AUnifiedLayout,
                typename AV::device_type,
                flare::MemoryTraits<flare::Unmanaged> >
                AV_Internal;
        typedef flare::View<typename XMV::const_data_type, XUnifiedLayout,
                typename XMV::device_type,
                flare::MemoryTraits<flare::Unmanaged> >
                XMV_Internal;

        YMV_Internal Y_internal = Y;
        AV_Internal A_internal  = A;
        XMV_Internal X_internal = X;

        flare::blas::detail::Mult<execution_space, YMV_Internal, AV_Internal, XMV_Internal>::mult(
                space, gamma, Y_internal, alpha, A_internal, X_internal);
    }

    /// \brief Element wise multiplication of two vectors:
    ///        Y[i] = gamma * Y[i] + alpha * A[i] * X[i]
    ///
    /// This function is non-blocking and thread-safe
    /// The kernel is executed in the default stream/queue
    /// associated with the execution space of YMV.
    ///
    /// \tparam YMV Type of the first vector Y; a 1-D or 2-D flare::View.
    /// \tparam AV  Type of the second vector A; a 1-D flare::View.
    /// \tparam XMV Type of the third vector X; a 1-D or 2-D flare::View.
    ///
    /// \param gamma [in] The scalar to apply to Y.
    /// \param Y [in/out] The Y vector.
    /// \param alpha [in] The scalar to apply to A.
    /// \param A [in]     The vector to apply to X.
    /// \para`m X [in]     The X vector.
    template <class YMV, class AV, class XMV>
    void mult(typename YMV::const_value_type& gamma, const YMV& Y,
              typename AV::const_value_type& alpha, const AV& A, const XMV& X) {
        mult(typename YMV::execution_space{}, gamma, Y, alpha, A, X);
    }

}  // namespace flare::blas

#endif  // FLARE_KERNEL_BLAS_MULTI_H_
