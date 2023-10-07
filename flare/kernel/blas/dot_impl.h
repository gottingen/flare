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

#ifndef FLARE_KERNEL_BLAS_IMPL_H_
#define FLARE_KERNEL_BLAS_IMPL_H_

#include <flare/kernel/common/inner_product_space_traits.h>
#include <flare/kernel/blas/utility.h>
#include <flare/core.h>

namespace flare::blas::detail {

    /// \brief Functor that implements the single-vector, two-argument
    ///   version of flare::blas::dot (dot product of two vectors).
    ///
    /// \tparam XVector Type of the first vector x; 1-D View
    /// \tparam YVector Type of the second vector y; 1-D View
    /// \tparam SizeType Type of the row index used in the dot product.
    ///   For best performance, use int instead of size_t here.
    template<class execution_space, class AV, class XVector, class YVector,
            typename SizeType>
    struct DotFunctor {
        typedef SizeType size_type;
        typedef typename AV::non_const_value_type avalue_type;
        typedef flare::detail::InnerProductSpaceTraits<avalue_type> IPT;
        typedef typename IPT::dot_type value_type;

        XVector m_x;
        YVector m_y;

        DotFunctor(const XVector &x, const YVector &y) : m_x(x), m_y(y) {}

        void run(const char *label, const execution_space &space, AV result) {
            flare::RangePolicy<execution_space, size_type> policy(space, 0,
                                                                  m_x.extent(0));
            flare::parallel_reduce(label, policy, *this, result);
        }

        // Prefer const size_type& to const size_type or size_type,
        // since the compiler has an easier time inlining the former.
        FLARE_FORCEINLINE_FUNCTION void operator()(const size_type &i,
                                                   value_type &sum) const {
            flare::detail::updateDot(sum, m_x(i), m_y(i));  // sum += m_x(i) * m_y(i)
        }

        FLARE_INLINE_FUNCTION void init(value_type &update) const {
            update = flare::ArithTraits<value_type>::zero();
        }

        FLARE_INLINE_FUNCTION void join(value_type &update,
                                        const value_type &source) const {
            update += source;
        }
    };


    template<class ExecSpace, class RV, class XV, class YV, class size_type>
    struct Dot_MV_Functor {
        using Scalar = typename RV::non_const_value_type;
        using IPT = flare::detail::InnerProductSpaceTraits<
                typename XV::non_const_value_type>;
        using dot_type = typename IPT::dot_type;
        using KAT = flare::ArithTraits<dot_type>;

        using TeamMem = typename flare::TeamPolicy<ExecSpace>::member_type;

        RV r;
        XV x;
        YV y;

        size_type
                teamsPerDot;  // number of teams collectively performing a dot product

        Dot_MV_Functor(const RV &r_, const XV &x_, const YV &y_, int teamsPerDot_)
                : r(r_), x(x_), y(y_), teamsPerDot(teamsPerDot_) {}

        FLARE_INLINE_FUNCTION void operator()(const TeamMem &t) const {
            size_type globalRank = t.league_rank();
            size_type localRank = globalRank % teamsPerDot;
            size_type i = globalRank / teamsPerDot;
            size_type xcol = x.extent(1) == 1 ? 0 : i;
            size_type ycol = y.extent(1) == 1 ? 0 : i;

            dot_type localResult = KAT::zero();
            size_type begin = localRank * (x.extent(0) / teamsPerDot);
            size_type end = (localRank + 1) * (x.extent(0) / teamsPerDot);
            if (localRank == teamsPerDot - 1) end = x.extent(0);
            flare::parallel_reduce(
                    flare::TeamThreadRange(t, begin, end),
                    [&](size_type k, dot_type &update) {
                        flare::detail::updateDot(update, x.access(k, xcol),
                                                 y.access(k, ycol));
                    },
                    localResult);

            flare::single(flare::PerTeam(t),
                          [&]() { flare::atomic_add(&r(i), Scalar(localResult)); });
        }
    };

    // Main version: the result view is accessible from execution space, so it can
    // be computed in-place
    template<class execution_space, class RV, class XV, class YV, class size_type>
    void MV_Dot_Invoke(
            const execution_space &space, const RV &r, const XV &x, const YV &y,
            typename std::enable_if<flare::SpaceAccessibility<
                    execution_space, typename RV::memory_space>::accessible>::type * =
            nullptr) {
        size_type numDots = std::max(x.extent(1), y.extent(1));
        if (x.extent(0) != y.extent(0)) {
            std::ostringstream oss;
            oss << "flare::blas::dot (rank-2): x and y have different lengths ("
                << x.extent(0) << " and " << y.extent(0) << ")";
            throw std::runtime_error(oss.str());
        }
        if ((x.extent(1) != size_t(1) && x.extent(1) != size_t(numDots)) ||
            (y.extent(1) != size_t(1) && y.extent(1) != size_t(numDots))) {
            std::ostringstream oss;
            oss << "flare::blas::dot (rank-2): x and y have incompatible numbers of "
                   "columns ("
                << x.extent(1) << " and " << y.extent(1) << ")";
            throw std::runtime_error(oss.str());
        }
        if (r.extent(0) != size_t(numDots)) {
            std::ostringstream oss;
            oss << "flare::blas::dot (rank-2): result vector has wrong length ("
                << r.extent(0) << ", but " << numDots
                << " dot products will be computed)";
            throw std::runtime_error(oss.str());
        }
        // Zero out the result vector
        flare::deep_copy(
                space, r, flare::ArithTraits<typename RV::non_const_value_type>::zero());
        size_type teamsPerDot;
        flare::blas::detail::multipleReductionWorkDistribution<execution_space,
                size_type>(
                x.extent(0), numDots, teamsPerDot);
        size_type numTeams = numDots * teamsPerDot;
        flare::TeamPolicy<execution_space> pol(space, numTeams, flare::AUTO);
        flare::parallel_for("Dot_MV", pol,
                            Dot_MV_Functor<execution_space, RV, XV, YV, size_type>(
                                    r, x, y, teamsPerDot));
    }

    // Version for when a temporary result view is needed (implemented in terms of
    // the other version)
    template<class execution_space, class RV, class XV, class YV, class size_type>
    void MV_Dot_Invoke(
            const execution_space &space, const RV &r, const XV &x, const YV &y,
            typename std::enable_if<!flare::SpaceAccessibility<
                    execution_space, typename RV::memory_space>::accessible>::type * =
            nullptr) {
        flare::View<typename RV::non_const_value_type *, typename XV::memory_space>
                tempResult(
                flare::view_alloc(flare::WithoutInitializing, "Dot_MV temp result"),
                r.extent(0));
        MV_Dot_Invoke<execution_space, decltype(tempResult), XV, YV, size_type>(
                space, tempResult, x, y);
        flare::deep_copy(space, r, tempResult);
        space.fence();
    }

    // Some platforms, such as Mac Clang, seem to get poor accuracy with
    // float and complex<float>.  Work around some Trilinos test
    // failures by using a higher-precision type for intermediate dot
    // product sums.
    //
    // Note that this is not the same thing as
    // InnerProductSpaceTraits<scalar>::dot_type
    template<typename scalar_t>
    struct DotAccumulatingScalar {
        using type = scalar_t;
    };

    template<>
    struct DotAccumulatingScalar<float> {
        using type = double;
    };

    template<>
    struct DotAccumulatingScalar<flare::complex<float>> {
        using type = flare::complex<double>;
    };

    template<typename scalar_t>
    struct HasSpecialAccumulator {
        enum : bool {
            value = !std::is_same<scalar_t,
                    typename DotAccumulatingScalar<scalar_t>::type>::value
        };
    };

    // Unification layer
    template<class execution_space, class RV, class XV, class YV, int XV_Rank = XV::rank, int YV_Rank = YV::rank>
    struct Dot {
        static_assert(flare::is_view<XV>::value,
                      "flare::blas::Impl::"
                      "Dot<2-D>: XV is not a flare::View.");
        static_assert(flare::is_view<YV>::value,
                      "flare::blas::Impl::"
                      "Dot<2-D>: YV is not a flare::View.");
        static_assert(RV::rank == 1,
                      "flare::blas::Impl::Dot<2-D>: "
                      "RV is not rank 1.");

        typedef typename YV::size_type size_type;

        // Helper to get the first column of a rank-1 or rank-2 view.
        // This makes it easier to add a path for single-column dot.
        template<typename V>
        static auto getFirstColumn(
                const V &v, typename std::enable_if<V::rank == 2>::type * = nullptr) {
            return flare::subview(v, flare::ALL(), 0);
        }

        template<typename V>
        static V getFirstColumn(
                const V &v, typename std::enable_if<V::rank == 1>::type * = nullptr) {
            return v;
        }

        static void dot(const execution_space &space, const RV &R, const XV &X,
                        const YV &Y) {
            flare::Profiling::pushRegion("flare::blas::dot");

            const size_type numRows = X.extent(0);
            const size_type numDots = std::max(X.extent(1), Y.extent(1));
            if (numDots == flare::ArithTraits<size_type>::one()) {
                auto R0 = flare::subview(R, 0);
                auto X0 = getFirstColumn(X);
                auto Y0 = getFirstColumn(Y);
                if (numRows < static_cast<size_type>(INT_MAX)) {
                    typedef int index_type;
                    DotFunctor<execution_space, decltype(R0), decltype(X0), decltype(Y0),
                            index_type>
                            f(X0, Y0);
                    f.run("flare::blas::dot<1D>", space, R0);
                } else {
                    typedef int64_t index_type;
                    DotFunctor<execution_space, decltype(R0), decltype(X0), decltype(Y0),
                            index_type>
                            f(X0, Y0);
                    f.run("flare::blas::dot<1D>", space, R0);
                }
            } else {
                if (numRows < static_cast<size_type>(INT_MAX) &&
                    numRows * numDots < static_cast<size_type>(INT_MAX)) {
                    typedef int index_type;
                    MV_Dot_Invoke<execution_space, RV, XV, YV, index_type>(space, R, X, Y);
                } else {
                    typedef std::int64_t index_type;
                    MV_Dot_Invoke<execution_space, RV, XV, YV, index_type>(space, R, X, Y);
                }
            }
            flare::Profiling::popRegion();
        }
    };


    template<class execution_space, class RV, class XV, class YV>
    struct DotSpecialAccumulator {
        static_assert(flare::is_view<XV>::value,
                      "flare::blas::Impl::"
                      "DotSpecialAccumulator: XV is not a flare::View.");
        static_assert(flare::is_view<YV>::value,
                      "flare::blas::Impl::"
                      "DotSpecialAccumulator: YV is not a flare::View.");
        static_assert(static_cast<int>(XV::rank) == static_cast<int>(YV::rank),
                      "flare::blas::Impl::"
                      "DotSpecialAccumulator: X and Y have different ranks.");
        static_assert(XV::rank == 1,
                      "flare::blas::Impl::"
                      "DotSpecialAccumulator: X and Y are not rank-1 Views.");
        static_assert(flare::is_view<RV>::value,
                      "flare::blas::Impl::"
                      "DotSpecialAccumulator: RV is not a flare::View.");
        static_assert(std::is_same<typename XV::non_const_value_type,
                              typename YV::non_const_value_type>::value,
                      "flare::blas::Impl::DotSpecialAccumulator: X and Y have "
                      "different scalar types.");
        static_assert(std::is_same<typename RV::value_type,
                              typename RV::non_const_value_type>::value,
                      "flare::blas::Dot<1D>: R is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");

        using size_type = typename YV::size_type;
        using dot_type = typename flare::detail::InnerProductSpaceTraits<
                typename XV::non_const_value_type>::dot_type;
        using accum_type = typename DotAccumulatingScalar<dot_type>::type;
        // This is the same View type as RV, but using the special accumulator as the
        // value type
        using RV_Result = flare::View<accum_type, typename RV::array_layout,
                typename RV::device_type,
                flare::MemoryTraits<flare::Unmanaged>>;

        static void dot(const execution_space &space, const RV_Result &R, const XV &X,
                        const YV &Y) {
            flare::Profiling::pushRegion("flare::blas::dot");
            const size_type numElems = X.extent(0);

            if (numElems < static_cast<size_type>(INT_MAX)) {
                typedef int index_type;
                DotFunctor<execution_space, RV_Result, XV, YV, index_type> f(X, Y);
                f.run("flare::blas::dot<1D>", space, R);
            } else {
                typedef int64_t index_type;
                DotFunctor<execution_space, RV_Result, XV, YV, index_type> f(X, Y);
                f.run("flare::blas::dot<1D>", space, R);
            }
            flare::Profiling::popRegion();
        }
    };

    //! Full specialization of Dot for single vectors (1-D Views).
    //  The rank-1 case is currently the only one that may use a different
    //  accumulator type than <tt>InnerProductSpaceTraits::dot_type</tt>.
    template<class execution_space, class RV, class XV, class YV>
    struct Dot<execution_space, RV, XV, YV, 1, 1> {
        // Check some things about the template parameters at compile time to get nice
        // error messages, before using them under the assumption they are valid.
        static_assert(flare::is_view<XV>::value,
                      "flare::blas::Impl::"
                      "Dot<1-D>: XV is not a flare::View.");
        static_assert(flare::is_view<YV>::value,
                      "flare::blas::Impl::"
                      "Dot<1-D>: YV is not a flare::View.");
        static_assert(flare::is_view<RV>::value,
                      "flare::blas::Impl::"
                      "Dot<1-D>: RV is not a flare::View.");
        static_assert(RV::rank == 0,
                      "flare::blas::Impl::Dot<1-D>: "
                      "RV is not rank 0.");
        static_assert(XV::rank == 1,
                      "flare::blas::Impl::Dot<1-D>: "
                      "XV is not rank 1.");
        static_assert(YV::rank == 1,
                      "flare::blas::Impl::Dot<1-D>: "
                      "YV is not rank 1.");
        static_assert(std::is_same<typename RV::value_type,
                              typename RV::non_const_value_type>::value,
                      "flare::blas::Dot<1D>: R is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");

        typedef typename YV::size_type size_type;
        typedef typename RV::non_const_value_type dot_type;
        typedef typename DotAccumulatingScalar<dot_type>::type special_result_type;

        // This is the same View type as RV, but using the special accumulator as the
        // value type
        typedef flare::View<special_result_type, typename RV::array_layout,
                typename RV::device_type,
                flare::MemoryTraits<flare::Unmanaged>>
                RV_Result;

        static void dot(const execution_space &space, const RV &R, const XV &X,
                        const YV &Y) {
            flare::Profiling::pushRegion("flare::blas::dot");
            const size_type numElems = X.extent(0);

            if (numElems < static_cast<size_type>(INT_MAX)) {
                typedef int index_type;
                DotFunctor<execution_space, RV, XV, YV, index_type> f(X, Y);
                f.run("flare::blas::dot<1D>", space, R);
            } else {
                typedef int64_t index_type;
                DotFunctor<execution_space, RV, XV, YV, index_type> f(X, Y);
                f.run("flare::blas::dot<1D>", space, R);
            }
            flare::Profiling::popRegion();
        }
    };

}  // namespace flare::blas::detail

#define FLARE_BLAS_DOT_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  template struct Dot<EXEC_SPACE,                                            \
                      flare::View<SCALAR, LAYOUT, flare::HostSpace,        \
                                   flare::MemoryTraits<flare::Unmanaged>>, \
                      flare::View<const SCALAR*, LAYOUT,                    \
                                   flare::Device<EXEC_SPACE, MEM_SPACE>,    \
                                   flare::MemoryTraits<flare::Unmanaged>>, \
                      flare::View<const SCALAR*, LAYOUT,                    \
                                   flare::Device<EXEC_SPACE, MEM_SPACE>,    \
                                   flare::MemoryTraits<flare::Unmanaged>>, \
                      1, 1>;                                                 \
  template struct Dot<                                                       \
      EXEC_SPACE,                                                            \
      flare::View<SCALAR, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>,    \
                   flare::MemoryTraits<flare::Unmanaged>>,                 \
      flare::View<const SCALAR*, LAYOUT,                                    \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                    \
                   flare::MemoryTraits<flare::Unmanaged>>,                 \
      flare::View<const SCALAR*, LAYOUT,                                    \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                    \
                   flare::MemoryTraits<flare::Unmanaged>>,                 \
      1, 1>;                                                                 \
  template struct DotSpecialAccumulator<                                     \
      EXEC_SPACE,                                                            \
      flare::View<SCALAR, LAYOUT, flare::HostSpace,                        \
                   flare::MemoryTraits<flare::Unmanaged>>,                 \
      flare::View<const SCALAR*, LAYOUT,                                    \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                    \
                   flare::MemoryTraits<flare::Unmanaged>>,                 \
      flare::View<const SCALAR*, LAYOUT,                                    \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                    \
                   flare::MemoryTraits<flare::Unmanaged>>>;                 \
  template struct DotSpecialAccumulator<                                     \
      EXEC_SPACE,                                                            \
      flare::View<SCALAR, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>,    \
                   flare::MemoryTraits<flare::Unmanaged>>,                 \
      flare::View<const SCALAR*, LAYOUT,                                    \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                    \
                   flare::MemoryTraits<flare::Unmanaged>>,                 \
      flare::View<const SCALAR*, LAYOUT,                                    \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                    \
                   flare::MemoryTraits<flare::Unmanaged>>>;

#define FLARE_BLAS_DOT_MV_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, \
                                         MEM_SPACE)                  \
  template struct Dot<                                               \
      EXEC_SPACE,                                                    \
      flare::View<SCALAR*, LAYOUT,                                  \
                   flare::Device<flare::DefaultHostExecutionSpace, \
                                  flare::HostSpace>,                \
                   flare::MemoryTraits<flare::Unmanaged>>,         \
      flare::View<const SCALAR**, LAYOUT,                           \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,            \
                   flare::MemoryTraits<flare::Unmanaged>>,         \
      flare::View<const SCALAR**, LAYOUT,                           \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,            \
                   flare::MemoryTraits<flare::Unmanaged>>,         \
      2, 2>;                                            \
  template struct Dot<                                               \
      EXEC_SPACE,                                                    \
      flare::View<SCALAR*, LAYOUT,                                  \
                   flare::Device<flare::DefaultHostExecutionSpace, \
                                  flare::HostSpace>,                \
                   flare::MemoryTraits<flare::Unmanaged>>,         \
      flare::View<const SCALAR**, LAYOUT,                           \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,            \
                   flare::MemoryTraits<flare::Unmanaged>>,         \
      flare::View<const SCALAR*, LAYOUT,                            \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,            \
                   flare::MemoryTraits<flare::Unmanaged>>,         \
      2, 1>;                                            \
  template struct Dot<                                               \
      EXEC_SPACE,                                                    \
      flare::View<SCALAR*, LAYOUT,                                  \
                   flare::Device<flare::DefaultHostExecutionSpace, \
                                  flare::HostSpace>,                \
                   flare::MemoryTraits<flare::Unmanaged>>,         \
      flare::View<const SCALAR*, LAYOUT,                            \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,            \
                   flare::MemoryTraits<flare::Unmanaged>>,         \
      flare::View<const SCALAR**, LAYOUT,                           \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,            \
                   flare::MemoryTraits<flare::Unmanaged>>,         \
      1, 2>;


#endif  // FLARE_KERNEL_BLAS_IMPL_H_
