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


#ifndef FLARE_KERNEL_DENSE_DISTANCE_HAMMING_H_
#define FLARE_KERNEL_DENSE_DISTANCE_HAMMING_H_

#include <flare/core.h>
#include <flare/kernel/common/inner_product_space_traits.h>
#include <flare/kernel/blas/utility.h>
#include <flare/core/arith_traits.h>
#include <flare/core/layout_utility.h>

namespace flare::kernel::dense {

    //
    // distance l2
    //

    /// \brief 1-distance functor for single vectors.
    ///
    /// \tparam RV 0-D output Tensor
    /// \tparam XV 1-D input Tensor
    /// \tparam SizeType Index type.  Use int (32 bits) if possible.

    template<typename RV, typename XV, class SizeType = typename XV::size_type>
    struct DistanceHammingFunctor {
        using size_type = SizeType;
        using xvalue_type = typename XV::non_const_value_type;
        using XAT = flare::ArithTraits<xvalue_type>;

        typename XV::const_type m_x;
        typename XV::const_type m_y;

        DistanceHammingFunctor(const XV &x, const XV &y) : m_x(x), m_y(y) {
            static_assert(flare::is_tensor<RV>::value,
                          "flare::ann::detail::DistanceHammingFunctor: "
                          "R is not a flare::Tensor.");
            static_assert(flare::is_tensor<XV>::value,
                          "flare::ann::detail::DistanceHammingFunctor: "
                          "X is not a flare::Tensor.");

            static_assert(std::is_same<typename RV::value_type,
                                  typename RV::non_const_value_type>::value,
                          "flare::ann::detail::DistanceHammingFunctor: R is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");

            static_assert(RV::rank == 0 && XV::rank == 1,
                          "flare::blas::detail::DistanceHammingFunctor: "
                          "RV must have rank 0 and XV must have rank 1.");
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i, double &sum) const {
            sum += flare::experimental::popcount_builtin(m_x(i) ^ m_y(i));
        }
        FLARE_INLINE_FUNCTION void init(double& update) const {
            update = 0.0;
        }

        FLARE_INLINE_FUNCTION void join(double& update,
                                        const double& source) const {
            update = source;
        }

        FLARE_INLINE_FUNCTION void final(double& update) const {
            (void)update;
        }

    };

    /// \brief Compute the distance jaccard of the single vector (1-D
    ///   Tensor) X, and store the result in the 0-D Tensor r.
    template<typename execution_space, typename RV, typename XV, class SizeType>
    void DistanceHammingInvoke(const execution_space &space, const RV &r, const XV &X, const XV &Y) {
        const SizeType numRows = static_cast<SizeType>(X.extent(0));
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        typedef DistanceHammingFunctor<RV, XV, SizeType> functor_type;
        functor_type op(X, Y);
        flare::parallel_reduce("flare::ann::distance_hamming", policy, op, r);
    }


}  // namespace flare::kernel::dense

#endif  // FLARE_KERNEL_DENSE_DISTANCE_HAMMING_H_
