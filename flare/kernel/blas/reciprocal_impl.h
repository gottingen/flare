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

#ifndef FLARE_KERNEL_BLAS_RECIPROCAL_IMPL_H_
#define FLARE_KERNEL_BLAS_RECIPROCAL_IMPL_H_

#include <flare/core.h>
#include <flare/kernel/common/inner_product_space_traits.h>
#include <flare/kernel/blas/utility.h>
#include <flare/core/arith_traits.h>
#include <flare/kernel/common/helper.h>

namespace flare::blas::detail {

    //
    // reciprocal
    //

    // Entry-wise reciprocalolute value / magnitude: R(i,j) = reciprocal(X(i,j)).
    template<class RMV, class XMV, class SizeType = typename RMV::size_type>
    struct MV_Reciprocal_Functor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename XMV::non_const_value_type> ATS;

        const size_type numCols;
        RMV R_;
        XMV X_;

        MV_Reciprocal_Functor(const RMV &R, const XMV &X)
                : numCols(X.extent(1)), R_(R), X_(X) {
            static_assert(flare::is_view<RMV>::value,
                          "flare::blas::Impl::"
                          "MV_Reciprocal_Functor: RMV is not a flare::View.");
            static_assert(flare::is_view<XMV>::value,
                          "flare::blas::Impl::"
                          "MV_Reciprocal_Functor: XMV is not a flare::View.");
            static_assert(RMV::rank == 2,
                          "flare::blas::Impl::"
                          "MV_Reciprocal_Functor: RMV is not rank 2");
            static_assert(XMV::rank == 2,
                          "flare::blas::Impl::"
                          "MV_Reciprocal_Functor: XMV is not rank 2");
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i) const {
            for (size_type j = 0; j < numCols; ++j) {
                R_(i, j) = ATS::one() / X_(i, j);
            }
        }
    };

    // Entry-wise, in-place reciprocalolute value / magnitude: R(i,j) =
    // reciprocal(R(i,j)).
    template<class RMV, class SizeType = typename RMV::size_type>
    struct MV_ReciprocalSelf_Functor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename RMV::non_const_value_type> ATS;

        const size_type numCols;
        RMV R_;

        MV_ReciprocalSelf_Functor(const RMV &R) : numCols(R.extent(1)), R_(R) {
            static_assert(flare::is_view<RMV>::value,
                          "flare::blas::Impl::"
                          "MV_Reciprocal_Functor: RMV is not a flare::View.");
            static_assert(RMV::rank == 2,
                          "flare::blas::Impl::"
                          "MV_Reciprocal_Functor: RMV is not rank 2");
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i) const {
            for (size_type j = 0; j < numCols; ++j) {
                R_(i, j) = ATS::one() / R_(i, j);
            }
        }
    };

    // Single-vector, entry-wise reciprocalolute value / magnitude: R(i) =
    // reciprocal(X(i)).
    template<class RV, class XV, class SizeType = typename RV::size_type>
    struct V_Reciprocal_Functor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename XV::non_const_value_type> ATS;

        RV R_;
        XV X_;

        V_Reciprocal_Functor(const RV &R, const XV &X) : R_(R), X_(X) {
            static_assert(flare::is_view<RV>::value,
                          "flare::blas::Impl::"
                          "V_Reciprocal_Functor: RV is not a flare::View.");
            static_assert(flare::is_view<XV>::value,
                          "flare::blas::Impl::"
                          "V_Reciprocal_Functor: XV is not a flare::View.");
            static_assert(RV::rank == 1,
                          "flare::blas::Impl::"
                          "V_Reciprocal_Functor: RV is not rank 1");
            static_assert(XV::rank == 1,
                          "flare::blas::Impl::"
                          "V_Reciprocal_Functor: XV is not rank 1");
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i) const { R_(i) = ATS::one() / X_(i); }
    };

    // Single-vector, entry-wise, in-place reciprocalolute value / magnitude: R(i) =
    // reciprocal(R(i)).
    template<class RV, class SizeType = typename RV::size_type>
    struct V_ReciprocalSelf_Functor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename RV::non_const_value_type> ATS;

        RV R_;

        V_ReciprocalSelf_Functor(const RV &R) : R_(R) {
            static_assert(flare::is_view<RV>::value,
                          "flare::blas::Impl::"
                          "V_Reciprocal_Functor: RV is not a flare::View.");
            static_assert(RV::rank == 1,
                          "flare::blas::Impl::"
                          "V_Reciprocal_Functor: RV is not rank 1");
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i) const { R_(i) = ATS::one() / R_(i); }
    };

    // Invoke the "generic" (not unrolled) multivector functor that
    // computes entry-wise reciprocalolute value.
    template<class execution_space, class RMV, class XMV, class SizeType>
    void MV_Reciprocal_Generic(const execution_space &space, const RMV &R,
                               const XMV &X) {
        static_assert(flare::is_view<RMV>::value,
                      "flare::blas::Impl::"
                      "MV_Reciprocal_Generic: RMV is not a flare::View.");
        static_assert(flare::is_view<XMV>::value,
                      "flare::blas::Impl::"
                      "MV_Reciprocal_Generic: XMV is not a flare::View.");
        static_assert(RMV::rank == 2,
                      "flare::blas::Impl::"
                      "MV_Reciprocal_Generic: RMV is not rank 2");
        static_assert(XMV::rank == 2,
                      "flare::blas::Impl::"
                      "MV_Reciprocal_Generic: XMV is not rank 2");

        const SizeType numRows = X.extent(0);
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        if (R == X) {  // if R and X are the same (alias one another)
            MV_ReciprocalSelf_Functor<RMV, SizeType> op(R);
            flare::parallel_for("flare::blas::Reciprocal::S0", policy, op);
        } else {
            MV_Reciprocal_Functor<RMV, XMV, SizeType> op(R, X);
            flare::parallel_for("flare::blas::Reciprocal::S1", policy, op);
        }
    }

    // Variant of MV_Reciprocal_Generic for single vectors (1-D Views) R and X.
    template<class execution_space, class RV, class XV, class SizeType>
    void V_Reciprocal_Generic(const execution_space &space, const RV &R,
                              const XV &X) {
        static_assert(flare::is_view<RV>::value,
                      "flare::blas::Impl::"
                      "V_Reciprocal_Generic: RV is not a flare::View.");
        static_assert(flare::is_view<XV>::value,
                      "flare::blas::Impl::"
                      "V_Reciprocal_Generic: XV is not a flare::View.");
        static_assert(RV::rank == 1,
                      "flare::blas::Impl::"
                      "V_Reciprocal_Generic: RV is not rank 1");
        static_assert(XV::rank == 1,
                      "flare::blas::Impl::"
                      "V_Reciprocal_Generic: XV is not rank 1");

        const SizeType numRows = X.extent(0);
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        if (R == X) {  // if R and X are the same (alias one another)
            V_ReciprocalSelf_Functor<RV, SizeType> op(R);
            flare::parallel_for("flare::blas::Reciprocal::S2", policy, op);
        } else {
            V_Reciprocal_Functor<RV, XV, SizeType> op(R, X);
            flare::parallel_for("flare::blas::Reciprocal::S3", policy, op);
        }
    }


    // Unification layer
    template<class execution_space, class RMV, class XMV, int rank = RMV::rank>
    struct Reciprocal {
        static void reciprocal(const execution_space &space, const RMV &R,
                               const XMV &X);
    };

    //! Full specialization of Reciprocal for single vectors (1-D Views).
    template<class execution_space, class RMV, class XMV>
    struct Reciprocal<execution_space, RMV, XMV, 1> {
        typedef typename XMV::size_type size_type;

        static void reciprocal(const execution_space &space, const RMV &R,
                               const XMV &X) {
            static_assert(flare::is_view<RMV>::value,
                          "flare::blas::Impl::"
                          "Reciprocal<1-D>: RMV is not a flare::View.");
            static_assert(flare::is_view<XMV>::value,
                          "flare::blas::Impl::"
                          "Reciprocal<1-D>: XMV is not a flare::View.");
            static_assert(RMV::rank == 1,
                          "flare::blas::Impl::Reciprocal<1-D>: "
                          "RMV is not rank 1.");
            static_assert(XMV::rank == 1,
                          "flare::blas::Impl::Reciprocal<1-D>: "
                          "XMV is not rank 1.");
            flare::Profiling::pushRegion("flare::blas::reciprocal");

            const size_type numRows = X.extent(0);

            if (numRows < static_cast<size_type>(INT_MAX)) {
                typedef int index_type;
                V_Reciprocal_Generic<execution_space, RMV, XMV, index_type>(space, R, X);
            } else {
                typedef std::int64_t index_type;
                V_Reciprocal_Generic<execution_space, RMV, XMV, index_type>(space, R, X);
            }
            flare::Profiling::popRegion();
        }
    };

    template<class execution_space, class RMV, class XMV>
    struct Reciprocal<execution_space, RMV, XMV, 2> {
        typedef typename XMV::size_type size_type;

        static void reciprocal(const execution_space &space, const RMV &R,
                               const XMV &X) {
            static_assert(flare::is_view<RMV>::value,
                          "flare::blas::Impl::"
                          "Reciprocal<2-D>: RMV is not a flare::View.");
            static_assert(flare::is_view<XMV>::value,
                          "flare::blas::Impl::"
                          "Reciprocal<2-D>: XMV is not a flare::View.");
            static_assert(RMV::rank == 2,
                          "flare::blas::Impl::Reciprocal<2-D>: "
                          "RMV is not rank 2.");
            static_assert(XMV::rank == 2,
                          "flare::blas::Impl::Reciprocal<2-D>: "
                          "XMV is not rank 2.");
            flare::Profiling::pushRegion("flare::blas::reciprocal");
            const size_type numRows = X.extent(0);
            const size_type numCols = X.extent(1);
            if (numRows < static_cast<size_type>(INT_MAX) &&
                numRows * numCols < static_cast<size_type>(INT_MAX)) {
                typedef int index_type;
                MV_Reciprocal_Generic<execution_space, RMV, XMV, index_type>(space, R, X);
            } else {
                typedef std::int64_t index_type;
                MV_Reciprocal_Generic<execution_space, RMV, XMV, index_type>(space, R, X);
            }
            flare::Profiling::popRegion();
        }
    };
}  // namespace flare::blas::detail

//
// Macro for definition of full specialization of
// flare::blas::Impl::Reciprocal for rank == 2.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define FLARE_BLAS_RECIPROCAL_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, \
                                             MEM_SPACE)                    \
  template struct Reciprocal<                                              \
      EXEC_SPACE,                                                          \
      flare::View<SCALAR*, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>, \
                   flare::MemoryTraits<flare::Unmanaged> >,              \
      flare::View<const SCALAR*, LAYOUT,                                  \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                  \
                   flare::MemoryTraits<flare::Unmanaged> >,              \
      1>;
//
// Macro for definition of full specialization of
// flare::blas::Impl::Reciprocal for rank == 2.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define FLARE_BLAS_RECIPROCAL_MV_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, \
                                                MEM_SPACE)                  \
  template struct Reciprocal<                                               \
      EXEC_SPACE,                                                           \
      flare::View<SCALAR**, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>, \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      flare::View<const SCALAR**, LAYOUT,                                  \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                   \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      2>;

#endif  // FLARE_KERNEL_BLAS_RECIPROCAL_IMPL_H_
