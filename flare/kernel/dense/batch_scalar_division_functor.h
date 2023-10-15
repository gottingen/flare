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

#ifndef FLARE_KERNEL_DENSE_BATCH_SCALAR_DIVISION_FUNCTOR_H_
#define FLARE_KERNEL_DENSE_BATCH_SCALAR_DIVISION_FUNCTOR_H_

#include <flare/core.h>
#include <flare/core/arith_traits.h>
#include <flare/core/layout_utility.h>

namespace flare::kernel::dense {

    // Single-vector, entry-wise div value  magnitude: R(i) = X(i) / y .
    template<typename DT, class RV, class XV, class SizeType = typename RV::size_type>
    struct BatchRightDivScalarFunctor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename XV::non_const_value_type> ATS;
        using value_type = typename ATS::mag_type;
        using batch_type = typename DT::batch_type;

        RV R_;
        XV X_;
        value_type y_;

        BatchRightDivScalarFunctor(const RV &R, const XV &X, const value_type y) : R_(R), X_(X), y_(y) {
            static_assert(flare::is_tensor<RV>::value,
                          "flare::blas::detail::"
                          "BatchRightDivScalarFunctor: RV is not a flare::Tensor.");
            static_assert(flare::is_tensor<XV>::value,
                          "flare::blas::detail::"
                          "BatchRightDivScalarFunctor: XV is not a flare::Tensor.");
            static_assert(RV::rank == 1,
                          "flare::blas::detail::"
                          "V_Abs_Functor: RV is not rank 1");
            static_assert(XV::rank == 1,
                          "flare::blas::detail::"
                          "BatchRightDivScalarFunctor: XV is not rank 1");
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i) const {
            auto inx = DT::batch_size * i;
            auto x = batch_type::load_aligned(X_.data() + inx);
            batch_type::store_aligned(R_.data() + inx, x/y_);
        }
    };

    // Single-vector, entry-wise, in-place absolute value / magnitude: R(i) =
    // abs(R(i)).
    template<typename DT, class RV, class SizeType = typename RV::size_type>
    struct BatchRightDivScalarSelfFunctor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename RV::non_const_value_type> ATS;
        using value_type = typename ATS::mag_type;
        using batch_type = typename DT::batch_type;
        RV R_;
        value_type y_;

        BatchRightDivScalarSelfFunctor(const RV &R, value_type y) : R_(R), y_(y) {
            static_assert(flare::is_tensor<RV>::value,
                          "flare::blas::detail::"
                          "batchRightDivScalarSelfFunctor: RV is not a flare::Tensor.");
            static_assert(RV::rank == 1,
                          "flare::blas::detail::"
                          "batchRightDivScalarSelfFunctor: RV is not rank 1");
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i) const {
            auto inx = DT::batch_size * i;
            auto x = batch_type::load_aligned(R_.data() + inx)/y_;
            batch_type::store_aligned(R_.data() + inx, x);
        }
    };

    template<class execution_space, class RV, class XV, class SizeType>
    void BatchRightDivScalarInvoke(const execution_space &space, const RV &R, const XV &X, typename RV::non_const_value_type norm) {
        static_assert(flare::is_tensor<RV>::value,
                      "flare::kernel::dense::"
                      "RightDivScalarInvoke: RV is not a flare::Tensor.");
        static_assert(flare::is_tensor<XV>::value,
                      "flare::kernel::dense::"
                      "RightDivScalarInvoke: XV is not a flare::Tensor.");
        static_assert(RV::rank == 1,
                      "flare::kernel::dense::"
                      "RightDivScalarInvoke: RV is not rank 1");
        static_assert(XV::rank == 1,
                      "flare::kernel::dense::"
                      "RightDivScalarInvoke: XV is not rank 1");

        using DT = simd_traits<XV, execution_space>;
        const SizeType numRows = X.extent(0);
        SizeType numBatch = numRows / DT::batch_size;
        const SizeType nMod = numRows % DT::batch_size;
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numBatch);

        if (nMod != 0) {
            for (SizeType i = numRows - nMod; i < numRows; ++i) {
                R(i) = X(i)/norm;
            }
        }
        if ((void *) (R.data()) ==
            (void *) (X.data())) {  // if R and X are the same (alias one another)
            BatchRightDivScalarSelfFunctor<RV, SizeType> op(R, norm);
            flare::parallel_for("flare::kernel::dense::scalar_div", policy, op);
        } else {
            BatchRightDivScalarFunctor<RV, XV, SizeType> op(R, X, norm);
            flare::parallel_for("flare::kernel::dense::scalar_div", policy, op);
        }
    }


}  // namespace flare::kernel::dense

#endif  // FLARE_KERNEL_DENSE_BATCH_SCALAR_DIVISION_FUNCTOR_H_
