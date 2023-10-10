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
// Created by jeff on 23-10-7.
//

#ifndef FLARE_KERNEL_BLAS_ABS_IMPL_H_
#define FLARE_KERNEL_BLAS_ABS_IMPL_H_

#include <flare/core.h>
#include <flare/core/arith_traits.h>

namespace flare::blas::detail {

    //
    // abs
    //

    // Entry-wise absolute value / magnitude: R(i,j) = abs(X(i,j)).
    template<class RMV, class XMV, class SizeType = typename RMV::size_type>
    struct MV_Abs_Functor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename XMV::non_const_value_type> ATS;

        const size_type numCols;
        RMV R_;
        XMV X_;

        MV_Abs_Functor(const RMV &R, const XMV &X)
                : numCols(X.extent(1)), R_(R), X_(X) {
            static_assert(flare::is_view<RMV>::value,
                          "flare::blas::detail::MV_Abs_Functor: RMV is not a flare::View.");
            static_assert(flare::is_view<XMV>::value,
                          "flare::blas::detail::"
                          "MV_Abs_Functor: XMV is not a flare::View.");
            static_assert(RMV::rank == 2,
                          "flare::blas::detail::"
                          "MV_Abs_Functor: RMV is not rank 2");
            static_assert(XMV::rank == 2,
                          "flare::blas::detail::"
                          "MV_Abs_Functor: XMV is not rank 2");
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i) const {
            for (size_type j = 0; j < numCols; ++j) {
                R_(i, j) = ATS::abs(X_(i, j));
            }
        }
    };

    // Entry-wise, in-place absolute value / magnitude: R(i,j) = abs(R(i,j)).
    template<class RMV, class SizeType = typename RMV::size_type>
    struct MV_AbsSelf_Functor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename RMV::non_const_value_type> ATS;

        const size_type numCols;
        RMV R_;

        MV_AbsSelf_Functor(const RMV &R) : numCols(R.extent(1)), R_(R) {
            static_assert(flare::is_view<RMV>::value,
                          "flare::blas::detail::"
                          "MV_Abs_Functor: RMV is not a flare::View.");
            static_assert(RMV::rank == 2,
                          "flare::blas::detail::"
                          "MV_Abs_Functor: RMV is not rank 2");
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i) const {
            for (size_type j = 0; j < numCols; ++j) {
                R_(i, j) = ATS::abs(R_(i, j));
            }
        }
    };

    // Single-vector, entry-wise absolute value / magnitude: R(i) = abs(X(i)).
    template<class RV, class XV, class SizeType = typename RV::size_type>
    struct V_Abs_Functor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename XV::non_const_value_type> ATS;

        RV R_;
        XV X_;

        V_Abs_Functor(const RV &R, const XV &X) : R_(R), X_(X) {
            static_assert(flare::is_view<RV>::value,
                          "flare::blas::detail::"
                          "V_Abs_Functor: RV is not a flare::View.");
            static_assert(flare::is_view<XV>::value,
                          "flare::blas::detail::"
                          "V_Abs_Functor: XV is not a flare::View.");
            static_assert(RV::rank == 1,
                          "flare::blas::detail::"
                          "V_Abs_Functor: RV is not rank 1");
            static_assert(XV::rank == 1,
                          "flare::blas::detail::"
                          "V_Abs_Functor: XV is not rank 1");
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i) const { R_(i) = ATS::abs(X_(i)); }
    };

    // Single-vector, entry-wise, in-place absolute value / magnitude: R(i) =
    // abs(R(i)).
    template<class RV, class SizeType = typename RV::size_type>
    struct V_AbsSelf_Functor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename RV::non_const_value_type> ATS;

        RV R_;

        V_AbsSelf_Functor(const RV &R) : R_(R) {
            static_assert(flare::is_view<RV>::value,
                          "flare::blas::detail::"
                          "V_Abs_Functor: RV is not a flare::View.");
            static_assert(RV::rank == 1,
                          "flare::blas::detail::"
                          "V_Abs_Functor: RV is not rank 1");
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i) const { R_(i) = ATS::abs(R_(i)); }
    };

    // Invoke the "generic" (not unrolled) multivector functor that
    // computes entry-wise absolute value.
    template<class execution_space, class RMV, class XMV, class SizeType>
    void MV_Abs_Generic(const execution_space &space, const RMV &R, const XMV &X) {
        static_assert(flare::is_view<RMV>::value,
                      "flare::blas::detail::"
                      "MV_Abs_Generic: RMV is not a flare::View.");
        static_assert(flare::is_view<XMV>::value,
                      "flare::blas::detail::"
                      "MV_Abs_Generic: XMV is not a flare::View.");
        static_assert(RMV::rank == 2,
                      "flare::blas::detail::"
                      "MV_Abs_Generic: RMV is not rank 2");
        static_assert(XMV::rank == 2,
                      "flare::blas::detail::"
                      "MV_Abs_Generic: XMV is not rank 2");

        const SizeType numRows = X.extent(0);
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        if ((void *) (R.data()) ==
            (void *) (X.data())) {  // if R and X are the same (alias one another)
            MV_AbsSelf_Functor<RMV, SizeType> op(R);
            flare::parallel_for("flare::blas::Abs::S0", policy, op);
        } else {
            MV_Abs_Functor<RMV, XMV, SizeType> op(R, X);
            flare::parallel_for("flare::blas::Abs::S1", policy, op);
        }
    }

    // Variant of MV_Abs_Generic for single vectors (1-D Views) R and X.
    template<class execution_space, class RV, class XV, class SizeType>
    void V_Abs_Generic(const execution_space &space, const RV &R, const XV &X) {
        static_assert(flare::is_view<RV>::value,
                      "flare::blas::detail::"
                      "V_Abs_Generic: RV is not a flare::View.");
        static_assert(flare::is_view<XV>::value,
                      "flare::blas::detail::"
                      "V_Abs_Generic: XV is not a flare::View.");
        static_assert(RV::rank == 1,
                      "flare::blas::detail::"
                      "V_Abs_Generic: RV is not rank 1");
        static_assert(XV::rank == 1,
                      "flare::blas::detail::"
                      "V_Abs_Generic: XV is not rank 1");

        const SizeType numRows = X.extent(0);
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        if ((void *) (R.data()) ==
            (void *) (X.data())) {  // if R and X are the same (alias one another)
            V_AbsSelf_Functor<RV, SizeType> op(R);
            flare::parallel_for("flare::blas::Abs::S2", policy, op);
        } else {
            V_Abs_Functor<RV, XV, SizeType> op(R, X);
            flare::parallel_for("flare::blas::Abs::S3", policy, op);
        }
    }

    // Unification layer
    template <
            class execution_space, class RMV, class XMV, int rank = RMV::rank>
    struct Abs {
        static void abs(const execution_space& space, const RMV& R, const XMV& X);
    };

    template <class execution_space, class RMV, class XMV>
    struct Abs<execution_space, RMV, XMV, 1> {
        using size_type = typename XMV::size_type;

        static void abs(const execution_space& space, const RMV& R, const XMV& X) {
            static_assert(flare::is_view<RMV>::value,
                          "flare::base::detail::"
                          "Abs<1-D>: RMV is not a flare::View.");
            static_assert(flare::is_view<XMV>::value,
                          "flare::base::detail::"
                          "Abs<1-D>: XMV is not a flare::View.");
            static_assert(RMV::rank == 1,
                          "flare::base::detail::Abs<1-D>: "
                          "RMV is not rank 1.");
            static_assert(XMV::rank == 1,
                          "flare::base::detail::Abs<1-D>: "
                          "XMV is not rank 1.");
            flare::Profiling::pushRegion("flare::blas::abs");
            const size_type numRows = X.extent(0);

            if (numRows < static_cast<size_type>(INT_MAX)) {
                typedef int index_type;
                V_Abs_Generic<execution_space, RMV, XMV, index_type>(space, R, X);
            } else {
                typedef std::int64_t index_type;
                V_Abs_Generic<execution_space, RMV, XMV, index_type>(space, R, X);
            }
            flare::Profiling::popRegion();
        }
    };

    template <class execution_space, class RMV, class XMV>
    struct Abs<execution_space, RMV, XMV, 2> {
        using size_type = typename XMV::size_type;

        static void abs(const execution_space& space, const RMV& R, const XMV& X) {
            static_assert(flare::is_view<RMV>::value,
                          "flare::base::detail::"
                          "Abs<2-D>: RMV is not a flare::View.");
            static_assert(flare::is_view<XMV>::value,
                          "flare::base::detail::"
                          "Abs<2-D>: XMV is not a flare::View.");
            static_assert(RMV::rank == 2,
                          "flare::base::detail::Abs<2-D>: "
                          "RMV is not rank 2.");
            static_assert(XMV::rank == 2,
                          "flare::base::detail::Abs<2-D>: "
                          "XMV is not rank 2.");
            flare::Profiling::pushRegion("flare::blas::abs");
            const size_type numRows = X.extent(0);
            const size_type numCols = X.extent(1);
            if (numRows < static_cast<size_type>(INT_MAX) &&
                numRows * numCols < static_cast<size_type>(INT_MAX)) {
                typedef int index_type;
                MV_Abs_Generic<execution_space, RMV, XMV, index_type>(space, R, X);
            } else {
                typedef std::int64_t index_type;
                MV_Abs_Generic<execution_space, RMV, XMV, index_type>(space, R, X);
            }
            flare::Profiling::popRegion();
        }
    };
}  // namespace flare::blas::detail

//
// Macro for definition of full specialization of
// flare::blas::detail::Abs for rank == 2.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define FLARE_BLAS_ABS_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  template struct Abs<                                                       \
      EXEC_SPACE,                                                            \
      flare::View<SCALAR*, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>,   \
                   flare::MemoryTraits<flare::Unmanaged> >,                \
      flare::View<const SCALAR*, LAYOUT,                                    \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                    \
                   flare::MemoryTraits<flare::Unmanaged> >,                \
      1>;

//
// Macro for definition of full specialization of
// flare::blas::detail::Abs for rank == 2.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define FLARE_BLAS_ABS_MV_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE,        \
                                         MEM_SPACE)                         \
  template struct Abs<                                                      \
      EXEC_SPACE,                                                           \
      flare::View<SCALAR**, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>, \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      flare::View<const SCALAR**, LAYOUT,                                  \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                   \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      2>;
      
#endif // FLARE_KERNEL_BLAS_ABS_IMPL_H_
