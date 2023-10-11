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

#ifndef FLARE_ANN_DISTANCE_L1_H_
#define FLARE_ANN_DISTANCE_L1_H_

#include <flare/ann/distance_l1_impl.h>

namespace flare::ann {

    /// \brief Return the L1 distance of the two vectors x and y.
    ///
    /// \tparam execution_space the flare execution space where the kernel
    ///         will be executed.
    /// \tparam XVector Type of the first vector x; a 1-D flare::View.
    ///
    /// \param space [in] an execution space instance that may specify
    ///                   in which stream/queue the kernel will be executed.
    /// \param x [in] Input 1-D View.
    /// \param y [in] Input 1-D View.
    ///
    /// \return The distance l1 result; a single value.

    template<class XVector, class execution_space,
            typename std::enable_if<flare::is_execution_space_v<execution_space>,
                    int>::type = 0>
    typename simd_traits<XVector, execution_space>::mag_type
    distance_l1(const execution_space &space, const XVector &x, const XVector &y, bool batch = true) {
        static_assert(
                flare::is_execution_space<execution_space>::value,
                "flare::ann::distance_l1: execution_space must be a flare::execution_space.");
        static_assert(flare::is_view<XVector>::value,
                      "flare::ann::distance_l1: XVector must be a flare::View.");
        static_assert(XVector::rank == 1,
                      "flare::ann::distance_l1: "
                      "Both Vector inputs must have rank 1.");
        using mag_type = typename simd_traits<XVector, execution_space>::mag_type;

        using XVector_Internal = flare::View<
                typename XVector::const_value_type *,
                typename flare::detail::GetUnifiedLayout<XVector>::array_layout,
                typename XVector::device_type, flare::MemoryTraits<flare::Unmanaged> >;

        using RVector_Internal =
                flare::View<mag_type, default_layout, flare::HostSpace,
                        flare::MemoryTraits<flare::Unmanaged> >;
        mag_type result;
        RVector_Internal R = RVector_Internal(&result);
        if (simd_traits<XVector, execution_space>::is_batch_available && batch) {
            flare::ann::detail::DistanceL1<execution_space, RVector_Internal, XVector_Internal>::batch_distance(space, R, x,
                                                                                                          y);
        } else {
            flare::ann::detail::DistanceL1<execution_space, RVector_Internal, XVector_Internal>::distance(space, R, x,
                                                                                                          y);
        }
        space.fence();
        return result;
    }

    template<class XVector>
    typename simd_traits<XVector>::mag_type distance_l1(const XVector &x, const XVector &y, bool batch = true) {
        return distance_l1(typename XVector::execution_space{}, x, y, batch);
    }

}  // namespace flare::ann

#endif  // FLARE_ANN_DISTANCE_L1_H_
