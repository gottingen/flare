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


#ifndef FLARE_KERNEL_BLAS_FILL_H_
#define FLARE_KERNEL_BLAS_FILL_H_

#include <flare/core.h>

namespace flare::blas {

    /// \brief Fill the multivector or single vector X with the given value.
    ///
    /// This function is non-blocking and thread-safe
    ///
    /// \tparam execution_space a flare execution space
    /// \tparam XMV 1-D or 2-D output View
    ///
    /// \param space [in] A flare instance of execution_space on which the
    ///                   kernel will run.
    /// \param X [out] Output View (1-D or 2-D).
    /// \param val [in] Value with which to fill the entries of X.
    template <class execution_space, class XMV>
    void fill(const execution_space& space, const XMV& X,
              const typename XMV::non_const_value_type& val) {
        flare::Profiling::pushRegion("flare::blas::fill<execution_space, XMV>");
        flare::deep_copy(space, X, val);
        flare::Profiling::popRegion();
    }

    /// \brief Fill the multivector or single vector X with the given value.
    ///
    /// This function is non-blocking and thread-safe
    /// The kernel is executed in the default stream/queue
    /// associated with the execution space of XMV.
    ///
    /// \tparam XMV 1-D or 2-D output View
    ///
    /// \param X [out] Output View (1-D or 2-D).
    /// \param val [in] Value with which to fill the entries of X.
    template <class XMV>
    void fill(const XMV& X, const typename XMV::non_const_value_type& val) {
        flare::Profiling::pushRegion("flare::blas::fill");
        flare::deep_copy(X, val);
        flare::Profiling::popRegion();
    }

}  // namespace flare::blas

#endif  // FLARE_KERNEL_BLAS_FILL_H_
