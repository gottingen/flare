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

#ifndef FLARE_ANN_DISTANCE_L1_IMPL_H_
#define FLARE_ANN_DISTANCE_L1_IMPL_H_

#include <flare/core.h>
#include <flare/core/arith_traits.h>
#include <flare/simd/simd.h>
#include <flare/core/layout_utility.h>

namespace flare::ann::detail {

    //
    // distance l1
    //

    /// \brief 1-distance functor for single vectors.
    ///
    /// \tparam RV 0-D output View
    /// \tparam XV 1-D input View
    /// \tparam SizeType Index type.  Use int (32 bits) if possible.
    template<typename RV, typename XV, class SizeType = typename XV::size_type>
    struct DistanceL1Functor {
        using size_type = SizeType;
        using xvalue_type = typename XV::non_const_value_type;
        using XAT = flare::ArithTraits<xvalue_type>;
        using value_type = typename XAT::mag_type;
        using MAT = flare::ArithTraits<value_type>;

        typename XV::const_type m_x;
        typename XV::const_type m_y;

        DistanceL1Functor(const XV &x, const XV &y) : m_x(x), m_y(y) {
            static_assert(flare::is_view<RV>::value,
                          "flare::ann::detail::DistanceL1Functor: "
                          "R is not a flare::View.");
            static_assert(flare::is_view<XV>::value,
                          "flare::ann::detail::DistanceL1Functor: "
                          "X is not a flare::View.");

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
        void operator()(const size_type &i, value_type &sum) const {
            sum += flare::abs(m_x(i) - m_y(i));
        }
    };

    /// \brief 1-norm functor for single vectors.
    ///
    /// \tparam RV 0-D output View
    /// \tparam XV 1-D input View
    /// \tparam SizeType Index type.  Use int (32 bits) if possible.
    template<typename DT, typename RV, typename XV, class SizeType = typename XV::size_type>
    struct BatchDistanceL1Functor {
        using size_type = SizeType;
        using xvalue_type = typename DT::mag_type;
        using batch_type = typename DT::batch_type;
        using XAT = flare::ArithTraits<xvalue_type>;
        using value_type = typename XAT::mag_type;

        typename XV::const_type m_x;
        typename XV::const_type m_y;
        SizeType m_len;

        BatchDistanceL1Functor(const XV &x, const XV &y) : m_x(x), m_y(y), m_len(m_x.extent(0)) {
            static_assert(flare::is_view<RV>::value,
                          "flare::ann::detail::DistanceL1Functor: "
                          "R is not a flare::View.");
            static_assert(flare::is_view<XV>::value,
                          "flare::ann::detail::DistanceL1Functor: "
                          "X is not a flare::View.");

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
        void operator()(const size_type &i, value_type &sum) const {
            auto inx = DT::batch_size * i;
            auto a = batch_type::load_aligned(m_x.data() + inx);
            auto b = batch_type::load_aligned(m_y.data() + inx);
            sum += flare::simd::reduce_add(flare::simd::abs(a - b));
        }
    };

    /// \brief Compute the distance l1 of the single vector (1-D
    ///   View) X, and store the result in the 0-D View r.
    template<typename execution_space, typename RV, typename XV, class SizeType>
    void DistanceL1Invoke(const execution_space &space, const RV &r, const XV &X, const XV &Y) {
        const SizeType numRows = static_cast<SizeType>(X.extent(0));
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        typedef DistanceL1Functor<RV, XV, SizeType> functor_type;
        functor_type op(X, Y);
        flare::parallel_reduce("flare::ann::distance_l1", policy, op, r);
    }

    /// \brief Compute the distance l1 of the single vector (1-D
    ///   View) X, and store the result in the 0-D View r.
    template<typename execution_space, typename RV, typename XV, typename SizeType>
    void DistanceL1BatchInvoke(const execution_space &space, const RV &r, const XV &X, const XV &Y) {
        using DT = simd_traits<XV, execution_space>;
        const SizeType numRows = static_cast<SizeType>(X.extent(0));
        SizeType numBatch = numRows / DT::batch_size;
        const SizeType nMod = numRows % DT::batch_size;
        flare::RangePolicy<execution_space, SizeType> batch_policy(space, 0, numBatch);
        typedef BatchDistanceL1Functor<DT, RV, XV, SizeType> batch_functor_type;
        batch_functor_type batch_op(X, Y);
        flare::parallel_reduce("flare::ann::batch_distance_l1", batch_policy, batch_op, r);
        // do it for local
        if (nMod != 0) {
            typename RV::non_const_value_type sum = 0.0;
            for (SizeType i = numRows - nMod; i < numRows; ++i) {
                //sum += flare::ArithTraits<typename RV::non_const_value_type>::abs(X(i) - Y(i));
                sum += flare::abs(X(i) - Y(i));
            }
            *r.data() += sum;
        }
    }

    template<typename execution_space, typename RV, typename XV>
    struct DistanceL1 {
        using size_type = typename XV::size_type;

        static void distance(const execution_space &space, const RV &R, const XV &X, const XV &Y) {
            static_assert(flare::is_view<RV>::value,
                          "flare::ann::detail::"
                          "DistanceL1<1-D>: RV is not a flare::View.");
            static_assert(flare::is_view<XV>::value,
                          "flare::ann::detail::"
                          "DistanceL1<1-D>: XV is not a flare::View.");
            static_assert(RV::rank == 0,
                          "flare::ann::detail::DistanceL1<1-D>: "
                          "RV is not rank 0.");
            static_assert(XV::rank == 1,
                          "flare::ann::detail::DistanceL1<1-D>: "
                          "XV is not rank 1.");
            flare::Profiling::pushRegion("flare::ann::DistanceL1");
            const size_type numRows = X.extent(0);

            if (numRows < static_cast<size_type>(INT_MAX)) {
                DistanceL1Invoke<execution_space, RV, XV, int>(space, R, X, Y);
            } else {
                using index_type = std::int64_t;
                DistanceL1Invoke<execution_space, RV, XV, index_type>(space, R, X, Y);
            }
            flare::Profiling::popRegion();
        }

        static void batch_distance(const execution_space &space, const RV &R, const XV &X, const XV &Y) {
            static_assert(flare::is_view<RV>::value,
                          "flare::ann::detail::"
                          "BatchDistanceL1<1-D>: RV is not a flare::View.");
            static_assert(flare::is_view<XV>::value,
                          "flare::ann::detail::"
                          "DistanceL1<1-D>: XV is not a flare::View.");
            static_assert(RV::rank == 0,
                          "flare::ann::detail::BatchDistanceL1<1-D>: "
                          "RV is not rank 0.");
            static_assert(XV::rank == 1,
                          "flare::ann::detail::BatchDistanceL1<1-D>: "
                          "XV is not rank 1.");
            flare::Profiling::pushRegion("flare::ann::BatchDistanceL1");
            const size_type numRows = X.extent(0);

            if (numRows < static_cast<size_type>(INT_MAX)) {
                DistanceL1BatchInvoke<execution_space, RV, XV, int>(space, R, X, Y);
            } else {
                using index_type = std::int64_t;
                DistanceL1BatchInvoke<execution_space, RV, XV, index_type>(space, R, X, Y);
            }
            flare::Profiling::popRegion();
        }
    };

}  // namespace flare::ann::detail

#endif  // FLARE_ANN_DISTANCE_L1_IMPL_H_
