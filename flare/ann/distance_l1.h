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

namespace flare::ann {

    /// \brief Return the L1 distance of the two vectors x and y.
    ///
    /// \tparam execution_space the flare execution space where the kernel
    ///         will be executed.
    /// \tparam XVector Type of the first vector x; a 1-D flare::View.
    /// \tparam YVector Type of the second vector y; a 1-D flare::View.
    ///
    /// \param space [in] an execution space instance that may specify
    ///                   in which stream/queue the kernel will be executed.
    /// \param x [in] Input 1-D View.
    /// \param y [in] Input 1-D View.
    ///
    /// \return The dot product result; a single value.

    template <class execution_space, class XVector, class YVector,
            typename std::enable_if<flare::is_execution_space_v<execution_space>,
                    int>::type = 0>
    typename flare::detail::InnerProductSpaceTraits<
            typename XVector::non_const_value_type>::dot_type
    distance_l1(const execution_space& space, const XVector& x, const YVector& y);

}  // namespace flare::ann

#endif  // FLARE_ANN_DISTANCE_L1_H_
