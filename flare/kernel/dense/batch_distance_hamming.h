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


#ifndef FLARE_KERNEL_DENSE_BATCH_DISTANCE_HAMMING_H_
#define FLARE_KERNEL_DENSE_BATCH_DISTANCE_HAMMING_H_

#include <flare/core.h>
#include <flare/core/arith_traits.h>
#include <flare/core/layout_utility.h>
#include <flare/kernel/dense/batch_pop_count.h>
#include <flare/core/bit_manipulation.h>

namespace flare::kernel::dense {

    /// \brief 1-norm functor for single vectors.
    ///
    /// \tparam RV 0-D output Tensor
    /// \tparam XV 1-D input Tensor
    /// \tparam SizeType Index type.  Use int (32 bits) if possible.
    template<typename DT, typename RV, typename XV, class SizeType = typename XV::size_type>
    struct BatchDistanceHammingFunctor {
        using size_type = SizeType;
        using xvalue_type = typename DT::mag_type;
        using batch_type = typename DT::batch_type;

        typename XV::const_type m_x;
        typename XV::const_type m_y;
        double m_init_value;

        BatchDistanceHammingFunctor(const XV &x, const XV &y, double init_value = 0.0)
                : m_x(x), m_y(y), m_init_value(init_value) {
            static_assert(flare::is_tensor<RV>::value,
                          "flare::ann::detail::DistanceL1Functor: "
                          "R is not a flare::Tensor.");
            static_assert(flare::is_tensor<XV>::value,
                          "flare::ann::detail::DistanceL1Functor: "
                          "X is not a flare::Tensor.");

            static_assert(std::is_same<typename RV::value_type,
                                  typename RV::non_const_value_type>::value,
                          "flare::ann::detail::DistanceL1Functor: R is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");

            static_assert(RV::rank == 0 && XV::rank == 1,
                          "flare::blas::detail::DistanceL1Functor: "
                          "RV must have rank 0 and XV must have rank 1.");
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i, double &sum) const {
            auto inx = DT::batch_size * i;
            auto x = batch_type::load_aligned(m_x.data() + inx);
            auto y = batch_type::load_aligned(m_y.data() + inx);
            sum += detail::batch_pop_count<xvalue_type, DT::batch_size>::pop_count(x ^ y);
        }

        FLARE_INLINE_FUNCTION void init(double &update) const {
            update = m_init_value;
        }

        FLARE_INLINE_FUNCTION void join(double &update,
                                        const double &source) const {
            update += source;
        }

        FLARE_INLINE_FUNCTION void final(double &update) const {
            (void)update;
        }

    };


    /// \brief Compute the distance l1 of the single vector (1-D
    ///   Tensor) X, and store the result in the 0-D Tensor r.
    template<typename execution_space, typename RV, typename XV, typename SizeType>
    void DistanceHammingBatchInvoke(const execution_space &space, const RV &r, const XV &X, const XV &Y) {
        using DT = simd_traits<XV, execution_space>;
        const SizeType numRows = static_cast<SizeType>(X.extent(0));
        SizeType numBatch = numRows / DT::batch_size;
        const SizeType nMod = numRows % DT::batch_size;
        flare::RangePolicy<execution_space, SizeType> batch_policy(space, 0, numBatch);
        double tail_sum{0.0};
        if (nMod != 0) {
            for (SizeType i = numRows - nMod; i < numRows; ++i) {
                tail_sum += flare::experimental::popcount_builtin(X(i) ^ Y(i));
            }
        }
        typedef BatchDistanceHammingFunctor<DT, RV, XV, SizeType> batch_functor_type;
        batch_functor_type batch_op(X, Y, tail_sum);
        flare::parallel_reduce("flare::ann::batch_distance_l2", batch_policy, batch_op, r);
        // do it for local
    }

}  // namespace flare::kernel::dense

#endif  // FLARE_KERNEL_DENSE_BATCH_DISTANCE_HAMMING_H_
