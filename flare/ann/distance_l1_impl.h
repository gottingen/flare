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
#include <flare/kernel/blas/utility.h>
#include <flare/kernel/common/helper.h>
#include <flare/core/arith_traits.h>
#include <flare/kernel/common/inner_product_space_traits.h>
#include <flare/simd/simd.h>

namespace flare::ann::detail {

    //
    // distance l1
    //

    /// \brief 1-norm functor for single vectors.
    ///
    /// \tparam RV 0-D output View
    /// \tparam XV 1-D input View
    /// \tparam SizeType Index type.  Use int (32 bits) if possible.
    template <typename RV, typename XV, class SizeType = typename XV::size_type>
    struct DistanceL1Functor {
        using size_type   = SizeType;
        using xvalue_type = typename XV::non_const_value_type;
        using XAT         = flare::ArithTraits<xvalue_type>;
        using value_type  = typename XAT::mag_type;
        using MAT         = flare::ArithTraits<value_type>;

        typename XV::const_type m_x;
        typename XV::const_type m_y;

        DistanceL1Functor(const XV& x, const XV& y) : m_x(x), m_y(y) {
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
        void operator()(const size_type& i, value_type& sum) const {
            xvalue_type val = m_x(i) - m_y(i);
            sum += MAT::abs(XAT::real(val)) + MAT::abs(XAT::imag(val));
        }
    };

    /// \brief 1-norm functor for single vectors.
    ///
    /// \tparam RV 0-D output View
    /// \tparam XV 1-D input View
    /// \tparam SizeType Index type.  Use int (32 bits) if possible.
    template <typename RV, typename XV, class SizeType = typename XV::size_type>
    struct DistanceL1BatchFunctor {
        using size_type   = SizeType;
        using xvalue_type = typename XV::non_const_value_type;
        using XAT         = flare::ArithTraits<xvalue_type>;
        using value_type  = typename XAT::mag_type;
        using MAT         = flare::ArithTraits<value_type>;

        typename XV::const_type m_x;
        typename XV::const_type m_y;

        DistanceL1BatchFunctor(const XV& x, const XV& y) : m_x(x), m_y(y) {
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
        void operator()(const size_type& i, value_type& sum) const {
            auto inx = flare::simd::batch<typename XV::value_type>::size * i;
            using vv_type = typename XV::non_const_value_type;
            auto a = flare::simd::batch<vv_type, flare::simd::default_arch>::load_aligned(m_x.data() + inx);
            auto b = flare::simd::batch<vv_type, flare::simd::default_arch>::load_aligned(m_y.data() + inx);
            sum += flare::simd::reduce_add(flare::simd::abs(a - b));
        }
    };

    /// \brief Compute the distance l1 of the single vector (1-D
    ///   View) X, and store the result in the 0-D View r.
    template <typename execution_space, typename RV, typename XV, class SizeType>
    void DistanceL1Invoke(const execution_space& space, const RV& r, const XV& X, const XV& Y) {
        const SizeType numRows = static_cast<SizeType>(X.extent(0));
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        typedef DistanceL1Functor<RV, XV, SizeType> functor_type;
        functor_type op(X, Y);
        flare::parallel_reduce("flare::ann::distance_l1", policy, op, r);
    }

    /// \brief Compute the distance l1 of the single vector (1-D
    ///   View) X, and store the result in the 0-D View r.
    template <typename execution_space, typename RV, typename XV, typename SizeType>
    void DistanceL1BatchInvoke(const execution_space& space, const RV& r, const XV& X, const XV& Y) {
        const SizeType numRows = static_cast<SizeType>(X.extent(0));
        using vv_type = typename XV::non_const_value_type;
        const SizeType numBatch = numRows/flare::simd::batch<vv_type, flare::simd::default_arch>::size;
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numBatch);

        typedef DistanceL1BatchFunctor<RV, XV, SizeType> functor_type;
        functor_type op(X, Y);
        flare::parallel_reduce("flare::ann::distance_l1", policy, op, r);
    }

    template <typename execution_space, typename RV, typename XV>
    struct DistanceL1 {
        using size_type = typename XV::size_type;

        static void distance(const execution_space& space, const RV& R, const XV& X, const XV& Y) {
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
    };

    template <typename execution_space, typename RV, typename XV>
    struct BatchDistanceL1 {
        using size_type = typename XV::size_type;

        static void distance(const execution_space& space, const RV& R, const XV& X, const XV& Y) {
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
