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
//
// Created by jeff on 23-10-8.
//

#ifndef FLARE_KERNEL_BLAS_NRM2_IMPL_H_
#define FLARE_KERNEL_BLAS_NRM2_IMPL_H_

#include <flare/core.h>
#include <flare/kernel/common/inner_product_space_traits.h>
#include <flare/kernel/blas/utility.h>
#include <flare/core/arith_traits.h>
#include <flare/core/layout_utility.h>

namespace flare::blas::detail {

    //
    // nrm2_squared
    //

    /// \brief 2-norm (squared) functor for single vectors.
    ///
    /// \tparam RV 0-D output Tensor
    /// \tparam XV 1-D input Tensor
    /// \tparam SizeType Index type.  Use int (32 bits) if possible.
    template <class RV, class XV, class SizeType = typename XV::size_type>
    struct V_Nrm2_Functor {
        typedef SizeType size_type;
        typedef typename XV::non_const_value_type xvalue_type;
        typedef flare::detail::InnerProductSpaceTraits<xvalue_type> IPT;
        typedef flare::ArithTraits<typename IPT::mag_type> AT;
        typedef typename IPT::mag_type value_type;

        typename XV::const_type m_x;
        bool m_take_sqrt;

        V_Nrm2_Functor(const XV& x, bool take_sqrt) : m_x(x), m_take_sqrt(take_sqrt) {
            static_assert(flare::is_tensor<RV>::value,
                          "flare::blas::Impl::V_Nrm2_Functor: "
                          "R is not a flare::Tensor.");
            static_assert(flare::is_tensor<XV>::value,
                          "flare::blas::Impl::V_Nrm2_Functor: "
                          "X is not a flare::Tensor.");
            static_assert(std::is_same<typename RV::value_type,
                                  typename RV::non_const_value_type>::value,
                          "flare::blas::Impl::V_Nrm2_Functor: R is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            static_assert(RV::rank == 0 && XV::rank == 1,
                          "flare::blas::Impl::V_Nrm2_Functor: "
                          "RV must have rank 0 and XV must have rank 1.");
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type& i, value_type& sum) const {
            const typename IPT::mag_type tmp = IPT::norm(m_x(i));
            sum += tmp * tmp;
        }

        FLARE_INLINE_FUNCTION void init(value_type& update) const {
            update = AT::zero();
        }

        FLARE_INLINE_FUNCTION void join(value_type& update,
                                         const value_type& source) const {
            update += source;
        }

        FLARE_INLINE_FUNCTION void final(value_type& update) const {
            if (m_take_sqrt)
                update =
                        flare::ArithTraits<typename RV::non_const_value_type>::sqrt(update);
        }
    };

/// \brief Column-wise 2-norm functor for multivectors; works for
///   any layout, but best performance with LayoutLeft.
///
/// \tparam RV 1-D output Tensor
/// \tparam XMV 2-D input Tensor
/// \tparam SizeType Index type.  Use int (32 bits) if possible.
    template <class ExecSpace, class RV, class XV, class size_type>
    struct Nrm2_MV_Functor {
        typedef typename RV::non_const_value_type rvalue_type;
        typedef typename XV::non_const_value_type xvalue_type;
        typedef flare::detail::InnerProductSpaceTraits<xvalue_type> IPT;
        typedef flare::ArithTraits<typename IPT::mag_type> AT;
        typedef typename IPT::mag_type value_type;

        using TeamMem = typename flare::TeamPolicy<ExecSpace>::member_type;

        RV r;
        XV x;

        size_type
                teamsPerVec;  // number of teams collectively performing a dot product

        Nrm2_MV_Functor(const RV& r_, const XV& x_, int teamsPerVec_)
                : r(r_), x(x_), teamsPerVec(teamsPerVec_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const TeamMem& t) const {
            size_type globalRank = t.league_rank();
            size_type localRank  = globalRank % teamsPerVec;
            size_type i          = globalRank / teamsPerVec;

            value_type localResult = AT::zero();
            size_type begin        = localRank * (x.extent(0) / teamsPerVec);
            size_type end          = (localRank + 1) * (x.extent(0) / teamsPerVec);
            if (localRank == teamsPerVec - 1) end = x.extent(0);

            flare::parallel_reduce(
                    flare::TeamThreadRange(t, begin, end),
                    [&](size_type k, value_type& update) {
                        value_type tmp = IPT::norm(x(k, i));
                        update += tmp * tmp;
                    },
                    localResult);

            flare::single(flare::PerTeam(t), [&]() {
                flare::atomic_add(&r(i), rvalue_type(localResult));
            });
        }
    };

/// \brief Compute the 2-norm (or its square) of the single vector (1-D
///   Tensor) X, and store the result in the 0-D Tensor r.
    template <class execution_space, class RV, class XV, class SizeType>
    void V_Nrm2_Invoke(const execution_space& space, const RV& r, const XV& X,
                       const bool& take_sqrt) {
        const SizeType numRows = static_cast<SizeType>(X.extent(0));
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        using functor_type = V_Nrm2_Functor<RV, XV, SizeType>;
        functor_type op(X, take_sqrt);
        flare::parallel_reduce("flare::blas::Nrm2::S0", policy, op, r);
    }

/// \brief Compute the 2-norms (or their square) of the columns of the
///   multivector (2-D Tensor) X, and store result(s) in the 1-D Tensor r.
// Main version: the result tensor is accessible from execution space, so it can
// be computed in-place
    template <class execution_space, class RV, class XV, class size_type>
    void MV_Nrm2_Invoke(
            const execution_space& space, const RV& r, const XV& x, bool take_sqrt,
            typename std::enable_if<flare::SpaceAccessibility<
                    execution_space, typename RV::memory_space>::accessible>::type* =
            nullptr) {
        if (r.extent(0) != x.extent(1)) {
            std::ostringstream oss;
            oss << "flare::blas::nrm2 (rank-2): result vector has wrong length ("
                << r.extent(0) << ", but x has " << x.extent(1) << " columns)";
            throw std::runtime_error(oss.str());
        }
        // Zero out the result vector
        flare::deep_copy(
                space, r, flare::ArithTraits<typename RV::non_const_value_type>::zero());
        size_type teamsPerVec;
        flare::blas::detail::multipleReductionWorkDistribution<execution_space,
                size_type>(
                x.extent(0), x.extent(1), teamsPerVec);
        size_type numTeams = x.extent(1) * teamsPerVec;
        flare::TeamPolicy<execution_space> pol(space, numTeams, flare::AUTO);
        flare::parallel_for(
                "flare::blas::Nrm2::S1", pol,
                Nrm2_MV_Functor<execution_space, RV, XV, size_type>(r, x, teamsPerVec));
        if (take_sqrt) {
            flare::parallel_for(
                    "flare::blas::Nrm2::Sqrt",
                    flare::RangePolicy<execution_space>(space, 0, r.extent(0)),
                    TakeSqrtFunctor<RV>(r));
        }
    }

// Version for when a temporary result tensor is needed (implemented in terms of
// the other version)
    template <class execution_space, class RV, class XV, class size_type>
    void MV_Nrm2_Invoke(
            const execution_space& space, const RV& r, const XV& x, bool take_sqrt,
            typename std::enable_if<!flare::SpaceAccessibility<
                    execution_space, typename RV::memory_space>::accessible>::type* =
            nullptr) {
        flare::Tensor<typename RV::non_const_value_type*, typename XV::memory_space>
                tempResult(
                flare::tensor_alloc(flare::WithoutInitializing, "Nrm2 temp result"),
                r.extent(0));
        MV_Nrm2_Invoke<execution_space, decltype(tempResult), XV, size_type>(
                space, tempResult, x, take_sqrt);
        flare::deep_copy(space, r, tempResult);
        space.fence();
    }


    // Unification layer
    template <class execution_space, class RMV, class XMV, int rank = XMV::rank>
    struct Nrm2 {
        static void nrm2(const execution_space& space, const RMV& R, const XMV& X,
                         const bool& take_sqrt);
    };

    //! Full specialization of Nrm2 for single vectors (1-D Tensors).
    template <class execution_space, class RMV, class XMV>
    struct Nrm2<execution_space, RMV, XMV, 1> {
        typedef typename XMV::size_type size_type;

        static void nrm2(const execution_space& space, const RMV& R, const XMV& X,
                         const bool& take_sqrt) {
            static_assert(flare::is_tensor<RMV>::value,
                          "flare::blas::Impl::"
                          "Nrm2<1-D>: RMV is not a flare::Tensor.");
            static_assert(flare::is_tensor<XMV>::value,
                          "flare::blas::Impl::"
                          "Nrm2<1-D>: XMV is not a flare::Tensor.");
            static_assert(RMV::rank == 0,
                          "flare::blas::Impl::Nrm2<1-D>: "
                          "RMV is not rank 0.");
            static_assert(XMV::rank == 1,
                          "flare::blas::Impl::Nrm2<1-D>: "
                          "XMV is not rank 1.");
            flare::Profiling::pushRegion("flare::blas::nrm2");
            const size_type numRows = X.extent(0);

            if (numRows < static_cast<size_type>(INT_MAX)) {
                V_Nrm2_Invoke<execution_space, RMV, XMV, int>(space, R, X, take_sqrt);
            } else {
                typedef std::int64_t index_type;
                V_Nrm2_Invoke<execution_space, RMV, XMV, index_type>(space, R, X,
                                                                     take_sqrt);
            }
            flare::Profiling::popRegion();
        }
    };

    template <class execution_space, class RV, class XMV>
    struct Nrm2<execution_space, RV, XMV, 2> {
        typedef typename XMV::size_type size_type;

        static void nrm2(const execution_space& space, const RV& R, const XMV& X,
                         const bool& take_sqrt) {
            static_assert(flare::is_tensor<RV>::value,
                          "flare::blas::Impl::"
                          "Nrm2<2-D>: RV is not a flare::Tensor.");
            static_assert(flare::is_tensor<XMV>::value,
                          "flare::blas::Impl::"
                          "Nrm2<2-D>: XMV is not a flare::Tensor.");
            static_assert(RV::rank == 1,
                          "flare::blas::Impl::Nrm2<2-D>: "
                          "RV is not rank 1.");
            static_assert(XMV::rank == 2,
                          "flare::blas::Impl::Nrm2<2-D>: "
                          "XMV is not rank 2.");
            flare::Profiling::pushRegion("flare::blas::nrm2");

            const size_type numRows = X.extent(0);
            const size_type numCols = X.extent(1);
            if (numCols == flare::ArithTraits<size_type>::one()) {
                auto R0 = flare::subtensor(R, 0);
                auto X0 = flare::subtensor(X, flare::ALL(), 0);
                if (numRows < static_cast<size_type>(INT_MAX)) {
                    V_Nrm2_Invoke<execution_space, decltype(R0), decltype(X0), int>(
                            space, R0, X0, take_sqrt);
                } else {
                    typedef std::int64_t index_type;
                    V_Nrm2_Invoke<execution_space, decltype(R0), decltype(X0), index_type>(
                            space, R0, X0, take_sqrt);
                }
            } else {
                if (numRows < static_cast<size_type>(INT_MAX) &&
                    numRows * numCols < static_cast<size_type>(INT_MAX)) {
                    MV_Nrm2_Invoke<execution_space, RV, XMV, int>(space, R, X, take_sqrt);
                } else {
                    typedef std::int64_t index_type;
                    MV_Nrm2_Invoke<execution_space, RV, XMV, index_type>(space, R, X,
                                                                         take_sqrt);
                }
            }
            flare::Profiling::popRegion();
        }
    };

}  // namespace flare::blas::detail

//
// Macro for definition of full specialization of
// flare::blas::Impl::Nrm2 for rank == 2.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define FLARE_BLAS_NRM2_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE)  \
  template struct Nrm2<                                                        \
      EXEC_SPACE,                                                              \
      flare::Tensor<                                                            \
          typename flare::detail::InnerProductSpaceTraits<SCALAR>::mag_type, \
          LAYOUT, flare::HostSpace,                                           \
          flare::MemoryTraits<flare::Unmanaged> >,                           \
      flare::Tensor<const SCALAR*, LAYOUT,                                      \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                      \
                   flare::MemoryTraits<flare::Unmanaged> >,                  \
      1>;
//
// Macro for definition of full specialization of
// flare::blas::Impl::Nrm2 for rank == 2.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define FLARE_BLAS_NRM2_MV_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, \
                                          MEM_SPACE)                  \
  template struct Nrm2<                                               \
      EXEC_SPACE,                                                     \
      flare::Tensor<typename flare::detail::InnerProductSpaceTraits< \
                       SCALAR>::mag_type*,                            \
                   LAYOUT,                                            \
                   flare::Device<flare::DefaultHostExecutionSpace,  \
                                  flare::HostSpace>,                 \
                   flare::MemoryTraits<flare::Unmanaged> >,         \
      flare::Tensor<const SCALAR**, LAYOUT,                            \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,             \
                   flare::MemoryTraits<flare::Unmanaged> >,         \
      2>;
#endif  // FLARE_KERNEL_BLAS_NRM2_IMPL_H_
