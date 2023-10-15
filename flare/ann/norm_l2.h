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
}  // namespace flare::ann

#endif  // FLARE_ANN_NORM_L2_H_
