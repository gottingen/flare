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


#ifndef FLARE_ANN_NORM_L2_H_
#define FLARE_ANN_NORM_L2_H_

#include <flare/ann/norm_l2_impl.h>
#include <flare/ann/normalize_impl.h>

namespace flare::ann {

    /// \brief Return the norm_l2 of the vector x.
    ///
    /// \tparam execution_space a flare execution space where the kernel will run.
    /// \tparam XVector Type of the first vector x; a 1-D flare::Tensor.
    ///
    /// \param space [in] the execution space instance, possibly containing a
    /// stream/queue where the kernel will be executed.
    /// \param x [in] Input 1-D Tensor.
    ///
    /// \return The norm_l1 product result; a single value.
    template<
            class execution_space, class XVector,
            typename std::enable_if<flare::is_execution_space<execution_space>::value,
                    int>::type = 0>
    typename simd_traits<XVector, execution_space>::mag_type
    norm_l2(const execution_space &space, const XVector &x, bool batch = true) {
        static_assert(
                flare::is_execution_space<execution_space>::value,
                "flare::ann::norm_l1: execution_space must be a flare::execution_space.");
        static_assert(flare::is_tensor<XVector>::value,
                      "flare::ann::norm_l1: XVector must be a flare::Tensor.");
        static_assert(XVector::rank == 1,
                      "flare::ann::norm_l1: "
                      "Both Vector inputs must have rank 1.");
        using mag_type = typename simd_traits<XVector, execution_space>::mag_type;

        using XVector_Internal = flare::Tensor<
                typename XVector::const_value_type *,
                typename flare::detail::GetUnifiedLayout<XVector>::array_layout,
                typename XVector::device_type, flare::MemoryTraits<flare::Unmanaged> >;

        using RVector_Internal =
                flare::Tensor<mag_type, default_layout, flare::HostSpace,
                        flare::MemoryTraits<flare::Unmanaged> >;

        mag_type result;
        RVector_Internal R = RVector_Internal(&result);
        XVector_Internal X = x;
        if (simd_traits<XVector, execution_space>::is_batch_available && batch) {
            flare::ann::detail::NormL2<execution_space, RVector_Internal, XVector_Internal>::batch_norm(space, R, X);
        } else {
            flare::ann::detail::NormL2<execution_space, RVector_Internal, XVector_Internal>::norm(space, R, X);
        }
        space.fence();
        return result;
    }

    /// \brief Return the norm_l1 of the vector x.
    ///
    /// \tparam XVector Type of the first vector x; a 1-D flare::Tensor.
    ///
    /// \param x [in] Input 1-D Tensor.
    ///
    /// \return The norm_l1 product result; a single value.
    template<class XVector>
    typename simd_traits<XVector>::mag_type norm_l2(const XVector &x, bool batch = true) {
        return norm_l2(typename XVector::execution_space{}, x,  batch);
    }

    /// \brief R(i,j) = normalize(X(i,j))
    ///
    /// Non-blocking function to replace each entry in R with the absolute value
    /// (magnitude) of the corresponding entry in X.
    ///
    /// \tparam execution_space a flare execution space to run the kernels on.
    /// \tparam RMV 1-D or 2-D flare::Tensor specialization.
    /// \tparam XMV 1-D or 2-D flare::Tensor specialization.  It must have
    ///   the same rank as RMV, and its entries must be assignable to
    ///   those of RMV.
    ///
    /// \param space [in] an execution_space instance where the kernel will run.
    /// \param R [out] tensor of type RMV that contains the absolute value X on
    /// output.
    /// \param X [in] tensor of type XMV.
    template <class execution_space, class RMV, class XMV>
    void normalize_l2(const execution_space& space, const RMV& R, const XMV& X, typename RMV::non_const_value_type norm, bool batch = true) {
        static_assert(flare::is_execution_space_v<execution_space>,
                      "flare::blas::abs: execution_space must be a valid flare "
                      "execution space.");
        static_assert(flare::is_tensor<RMV>::value,
                      "flare::blas::abs: "
                      "R is not a flare::Tensor.");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename RMV::memory_space>::accessible,
                "flare::blas::abs: RMV must be accessible from execution space");
        static_assert(flare::is_tensor<XMV>::value,
                      "flare::blas::abs: "
                      "X is not a flare::Tensor.");
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

        // Create unmanaged versions of the input Tensors.  RMV and XMV may be
        // rank 1 or rank 2.
        using RMV_Internal = flare::Tensor<
                typename std::conditional<RMV::rank == 1,
                        typename RMV::non_const_value_type*,
                        typename RMV::non_const_value_type**>::type,
                typename flare::detail::GetUnifiedLayout<RMV>::array_layout,
                typename RMV::device_type, flare::MemoryTraits<flare::Unmanaged> >;
        using XMV_Internal = flare::Tensor<
                typename std::conditional<XMV::rank == 1, typename XMV::const_value_type*,
                        typename XMV::const_value_type**>::type,
                typename flare::detail::GetUnifiedLayout<XMV>::array_layout,
                typename XMV::device_type, flare::MemoryTraits<flare::Unmanaged> >;

        RMV_Internal R_internal = R;
        XMV_Internal X_internal = X;

        if (simd_traits<XMV, execution_space>::is_batch_available && batch) {
            flare::ann::detail::NormalizeDiv<execution_space, RMV_Internal, XMV_Internal>::normalize(space, R_internal,
                                                                                                    X_internal, norm);
        } else {
            flare::ann::detail::NormalizeDiv<execution_space, RMV_Internal, XMV_Internal>::normalize(space, R_internal,
                                                                                                    X_internal, norm);
        }
    }

    /// \brief R(i,j) = normalize_l2(X(i,j))
    ///
    /// Non-blocking function to replace each entry in R with the absolute value
    /// (magnitude) of the corresponding entry in X. The kernel is executed in the
    /// default stream/queue associated with the execution space of RMV.
    ///
    /// \tparam RMV 1-D or 2-D flare::Tensor specialization.
    /// \tparam XMV 1-D or 2-D flare::Tensor specialization.  It must have
    ///   the same rank as RMV, and its entries must be assignable to
    ///   those of RMV.
    ///
    /// \param R [out] tensor of type RMV that contains the absolute value X on
    /// output.
    /// \param X [in] tensor of type XMV.
    template <class RMV, class XMV>
    void normalize_l2(const RMV& R, const XMV& X, typename RMV::non_const_value_type norm, bool batch = true) {
        normalize_l2(typename RMV::execution_space{}, R, X, norm,batch);
    }

    /// \brief R(i,j) = normalize_l2(X(i,j))
    ///
    /// Non-blocking function to replace each entry in R with the absolute value
    /// (magnitude) of the corresponding entry in X. The kernel is executed in the
    /// default stream/queue associated with the execution space of RMV.
    ///
    /// \tparam RMV 1-D or 2-D flare::Tensor specialization.
    /// \tparam XMV 1-D or 2-D flare::Tensor specialization.  It must have
    ///   the same rank as RMV, and its entries must be assignable to
    ///   those of RMV.
    ///
    /// \param R [out] tensor of type RMV that contains the absolute value X on
    /// output.
    /// \param X [in] tensor of type XMV.
    template <class RMV, class XMV>
    void normalize_l2(const RMV& R, const XMV& X, bool batch = true) {
        auto norm = norm_l2(X, batch);
        normalize_l2(typename RMV::execution_space{}, R, X, norm, batch);
    }
}  // namespace flare::ann

#endif  // FLARE_ANN_NORM_L2_H_
