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


#ifndef FLARE_KERNEL_DENSE_BATCH_NORM_L1_H_
#define FLARE_KERNEL_DENSE_BATCH_NORM_L1_H_

#include <flare/core.h>
#include <flare/core/arith_traits.h>
#include <flare/core/layout_utility.h>

namespace flare::kernel::dense {

    //
    // norm_l1
    //

    /// \brief 1-norm functor for single vectors.
    ///
    /// \tparam RV 0-D output Tensor
    /// \tparam XV 1-D input Tensor
    /// \tparam SizeType Index type.  Use int (32 bits) if possible.
    template <typename DT, class RV, class XV, class SizeType = typename XV::size_type>
    struct BatchNormL1Functor {
        using size_type   = SizeType;
        using xvalue_type = typename XV::non_const_value_type;
        using XAT         = flare::ArithTraits<xvalue_type>;
        using value_type  = typename XAT::mag_type;
        using MAT         = flare::ArithTraits<value_type>;
        using batch_type = typename DT::batch_type;

        typename XV::const_type m_x;
        value_type m_init_value;

        BatchNormL1Functor(const XV& x, value_type init_value = MAT::zero()) : m_x(x), m_init_value(init_value) {
            static_assert(flare::is_tensor<RV>::value,
                          "flare::kernel::dense::BatchNormL1Functor: "
                          "R is not a flare::Tensor.");
            static_assert(flare::is_tensor<XV>::value,
                          "flare::kernel::dense::BatchNormL1Functor: "
                          "X is not a flare::Tensor.");
            static_assert(std::is_same<typename RV::value_type,
                                  typename RV::non_const_value_type>::value,
                          "flare::kernel::dense::BatchNormL1Functor: R is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            static_assert(RV::rank == 0 && XV::rank == 1,
                          "flare::kernel::dense::BatchNormL1Functor: "
                          "RV must have rank 0 and XV must have rank 1.");
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type& i, value_type& sum) const {
            auto inx = DT::batch_size * i;
            auto a = batch_type::load_aligned(m_x.data() + inx);
            sum += flare::simd::reduce_add(flare::simd::abs(a));
        }

        FLARE_INLINE_FUNCTION void init(value_type& update) const {
            update = m_init_value;
        }

        FLARE_INLINE_FUNCTION void join(value_type& update,
                                        const value_type& source) const {
            update += source;
        }

        FLARE_INLINE_FUNCTION void final(value_type& update) const {
            (void)update;
        }

    };

    /// \brief Compute the 2-norm (or its square) of the single vector (1-D
    ///   Tensor) X, and store the result in the 0-D Tensor r.
    template <class execution_space, class RV, class XV, class SizeType>
    void BatchNormL1Invoke(const execution_space& space, const RV& r, const XV& X) {
        using DT = simd_traits<XV, execution_space>;
        const SizeType numRows = static_cast<SizeType>(X.extent(0));
        SizeType numBatch = numRows / DT::batch_size;
        const SizeType nMod = numRows % DT::batch_size;
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numBatch);

        typename RV::non_const_value_type sum = 0.0;
        if (nMod != 0) {
            for (SizeType i = numRows - nMod; i < numRows; ++i) {
                sum += flare::ArithTraits<typename RV::non_const_value_type>::abs(X(i));
            }
        }

        typedef BatchNormL1Functor<DT, RV, XV, SizeType> functor_type;
        functor_type op(X, sum);
        flare::parallel_reduce("flare::kernel::dense::norm_l1::S0", policy, op, r);
        // do it for local
    }



}  // namespace flare::kernel::dense

#endif  // FLARE_KERNEL_DENSE_BATCH_NORM_L1_H_
