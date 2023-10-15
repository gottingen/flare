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


#ifndef FLARE_ANN_NORMALIZE_IMPL_H_
#define FLARE_ANN_NORMALIZE_IMPL_H_
#include <flare/core.h>
#include <flare/core/arith_traits.h>
#include <flare/simd/simd.h>
#include <flare/core/layout_utility.h>
#include <flare/kernel/dense/norm_l1.h>
#include <flare/kernel/dense/scalar_division_functor.h>
#include <flare/kernel/dense/batch_scalar_division_functor.h>

namespace flare::ann::detail {

    /// \brief base normalize of the vector x.
    /// X(i) = X(i) / norm norm:
    /// \tparam execution_space a flare execution space where the kernel will run.
    /// \tparam XVector Type of the first vector x; a 1-D flare::Tensor.
    ///
    /// \param space [in] the execution space instance, possibly containing a
    /// stream/queue where the kernel will be executed.
    /// \param X [in] Input 1-D Tensor.
    /// \param R [out] output 1-D Tensor, can be same  with X.
    ///
    /// \return void

    template <class execution_space, class RMV, class XMV>
    struct NormalizeDiv{
        using size_type = typename XMV::size_type;

        static void normalize(const execution_space& space, const RMV& R, const XMV& X, typename RMV::non_const_value_type norm) {
            static_assert(flare::is_tensor<RMV>::value,
                          "flare::ann::NormalizeDiv::normalize<1-D>:"
                          "RMV is not a flare::Tensor.");
            static_assert(flare::is_tensor<XMV>::value,
                          "flare::ann::NormalizeDiv::normalize<1-D>:"
                          "XMV is not a flare::Tensor.");
            static_assert(RMV::rank == 1,
                          "flare::ann::NormalizeDiv::normalize<1-D>:"
                          "RMV is not rank 1.");
            static_assert(XMV::rank == 1,
                          "flare::ann::NormalizeDiv::normalize<1-D>:"
                          "XMV is not rank 1.");
            flare::Profiling::pushRegion("flare::ann::normalize");
            const size_type numRows = X.extent(0);

            if (numRows < static_cast<size_type>(INT_MAX)) {
                typedef int index_type;
                flare::kernel::dense::RightDivScalarInvoke<execution_space, RMV, XMV, index_type>(space, R, X, norm);
            } else {
                typedef std::int64_t index_type;
                flare::kernel::dense::RightDivScalarInvoke<execution_space, RMV, XMV, index_type>(space, R, X, norm);
            }
            flare::Profiling::popRegion();
        }

        static void batch_normalize(const execution_space& space, const RMV& R, const XMV& X, typename RMV::non_const_value_type norm) {
            static_assert(flare::is_tensor<RMV>::value,
                          "flare::ann::NormalizeDiv::batch_normalize<1-D>:"
                          "RMV is not a flare::Tensor.");
            static_assert(flare::is_tensor<XMV>::value,
                          "flare::ann::NormalizeDiv::batch_normalize<1-D>:"
                          "XMV is not a flare::Tensor.");
            static_assert(RMV::rank == 1,
                          "flare::ann::NormalizeDiv::batch_normalize<1-D>:"
                          "RMV is not rank 1.");
            static_assert(XMV::rank == 1,
                          "flare::ann::NormalizeDiv::batch_normalize<1-D>:"
                          "XMV is not rank 1.");
            flare::Profiling::pushRegion("flare::ann::batch_normalize");
            const size_type numRows = X.extent(0);

            if (numRows < static_cast<size_type>(INT_MAX)) {
                typedef int index_type;
                FLARE_IF_ON_DEVICE((flare::kernel::dense::RightDivScalarInvoke<execution_space, RMV, XMV, index_type>(space, R, X, norm);))
                FLARE_IF_ON_HOST((flare::kernel::dense::BatchRightDivScalarInvoke<execution_space, RMV, XMV, index_type>(space, R, X, norm);))
            } else {
                typedef std::int64_t index_type;
                FLARE_IF_ON_DEVICE((flare::kernel::dense::RightDivScalarInvoke<execution_space, RMV, XMV, index_type>(space, R, X, norm);))
                FLARE_IF_ON_HOST((flare::kernel::dense::BatchRightDivScalarInvoke<execution_space, RMV, XMV, index_type>(space, R, X, norm);))
            }
            flare::Profiling::popRegion();
        }
    };
}  // namespace flare::ann::detail
#endif  // FLARE_ANN_NORMALIZE_IMPL_H_
