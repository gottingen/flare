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

#ifndef FLARE_KERNEL_BLAS_NRMINF_IMPL_H_
#define FLARE_KERNEL_BLAS_NRMINF_IMPL_H_

#include <flare/core.h>
#include <flare/kernel/common/inner_product_space_traits.h>
#include <flare/kernel/blas/utility.h>
#include <flare/core/arith_traits.h>
#include <flare/core/layout_utility.h>

namespace flare::blas::detail {
    //
    // nrminf_squared
    //

    /// \brief 2-norm (squared) functor for single vectors.
    ///
    /// \tparam RV 0-D output View
    /// \tparam XV 1-D input View
    /// \tparam SizeType Index type.  Use int (32 bits) if possible.
    template <class RV, class XV, class SizeType = typename XV::size_type>
    struct V_NrmInf_Functor {
        typedef typename XV::execution_space execution_space;
        typedef SizeType size_type;
        typedef typename XV::non_const_value_type xvalue_type;
        typedef flare::detail::InnerProductSpaceTraits<xvalue_type> IPT;
        typedef flare::ArithTraits<typename IPT::mag_type> AT;
        typedef typename IPT::mag_type value_type;

        typename XV::const_type m_x;

        V_NrmInf_Functor(const XV& x) : m_x(x) {
            static_assert(flare::is_view<RV>::value,
                          "flare::blas::Impl::V_NrmInf_Functor: "
                          "R is not a flare::View.");
            static_assert(flare::is_view<XV>::value,
                          "flare::blas::Impl::V_NrmInf_Functor: "
                          "X is not a flare::View.");
            static_assert(std::is_same<typename RV::value_type,
                                  typename RV::non_const_value_type>::value,
                          "flare::blas::Impl::V_NrmInf_Functor: R is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            static_assert(RV::rank == 0 && XV::rank == 1,
                          "flare::blas::Impl::V_NrmInf_Functor: "
                          "RV must have rank 0 and XV must have rank 1.");
        }

        FLARE_INLINE_FUNCTION void operator()(const size_type& i, value_type& max) const {
            value_type val = IPT::norm(m_x(i));
            if (val > max) max = val;
        }
    };

    /// \brief Compute the 2-norm (or its square) of the single vector (1-D
    ///   View) X, and store the result in the 0-D View r.
    template <class execution_space, class RV, class XV, class SizeType>
    void V_NrmInf_Invoke(const execution_space& space, const RV& r, const XV& X) {
        typedef flare::ArithTraits<typename RV::non_const_value_type> AT;

        const SizeType numRows = static_cast<SizeType>(X.extent(0));

        // Avoid Max Reduction if this is a zero length view
        if (numRows == 0) {
            flare::deep_copy(r, AT::zero());
            return;
        }

        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        typedef V_NrmInf_Functor<RV, XV, SizeType> functor_type;
        functor_type op(X);
        flare::parallel_reduce("flare::blas::NrmInf::S0", policy, op,
                                flare::Max<typename RV::non_const_value_type>(r()));
    }

    /// \brief Compute the 2-norms (or their square) of the columns of the
    ///   multivector (2-D View) X, and store result(s) in the 1-D View r.
    template <class execution_space, class RV, class XMV, class SizeType>
    void MV_NrmInf_Invoke(const execution_space& space, const RV& r, const XMV& X) {
        for (size_t i = 0; i < X.extent(1); i++) {
            auto ri = flare::subview(r, i);
            auto Xi = flare::subview(X, flare::ALL(), i);
            V_NrmInf_Invoke<execution_space, decltype(ri), decltype(Xi), SizeType>(
                    space, ri, Xi);
        }
    }


    // Unification layer
    template <class execution_space, class RMV, class XMV, int rank = XMV::rank>
    struct NrmInf {
        static void nrminf(const execution_space& space, const RMV& R, const XMV& X);
    };

    //! Full specialization of NrmInf for single vectors (1-D Views).
    template <class execution_space, class RMV, class XMV>
    struct NrmInf<execution_space, RMV, XMV, 1> {
        typedef typename XMV::size_type size_type;

        static void nrminf(const execution_space& space, const RMV& R, const XMV& X) {
            static_assert(flare::is_view<RMV>::value,
                          "flare::blas::Impl::"
                          "NrmInf<1-D>: RMV is not a flare::View.");
            static_assert(flare::is_view<XMV>::value,
                          "flare::blas::Impl::"
                          "NrmInf<1-D>: XMV is not a flare::View.");
            static_assert(RMV::rank == 0,
                          "flare::blas::Impl::NrmInf<1-D>: "
                          "RMV is not rank 0.");
            static_assert(XMV::rank == 1,
                          "flare::blas::Impl::NrmInf<1-D>: "
                          "XMV is not rank 1.");
            flare::Profiling::pushRegion("flare::blas::nrminf");
            const size_type numRows = X.extent(0);

            if (numRows < static_cast<size_type>(INT_MAX)) {
                V_NrmInf_Invoke<execution_space, RMV, XMV, int>(space, R, X);
            } else {
                typedef std::int64_t index_type;
                V_NrmInf_Invoke<execution_space, RMV, XMV, index_type>(space, R, X);
            }
            flare::Profiling::popRegion();
        }
    };

    template <class execution_space, class RV, class XMV>
    struct NrmInf<execution_space, RV, XMV, 2> {
        typedef typename XMV::size_type size_type;

        static void nrminf(const execution_space& space, const RV& R, const XMV& X) {
            static_assert(flare::is_view<RV>::value,
                          "flare::blas::Impl::"
                          "NrmInf<2-D>: RV is not a flare::View.");
            static_assert(flare::is_view<XMV>::value,
                          "flare::blas::Impl::"
                          "NrmInf<2-D>: XMV is not a flare::View.");
            static_assert(RV::rank == 1,
                          "flare::blas::Impl::NrmInf<2-D>: "
                          "RV is not rank 1.");
            static_assert(XMV::rank == 2,
                          "flare::blas::Impl::NrmInf<2-D>: "
                          "XMV is not rank 2.");
            flare::Profiling::pushRegion("flare::blas::nrminf");
            const size_type numRows = X.extent(0);
            const size_type numCols = X.extent(1);
            if (numRows < static_cast<size_type>(INT_MAX) &&
                numRows * numCols < static_cast<size_type>(INT_MAX)) {
                MV_NrmInf_Invoke<execution_space, RV, XMV, int>(space, R, X);
            } else {
                typedef std::int64_t index_type;
                MV_NrmInf_Invoke<execution_space, RV, XMV, index_type>(space, R, X);
            }
            flare::Profiling::popRegion();
        }
    };
}  // namespace flare::blas::detail

//
// Macro for definition of full specialization of
// flare::blas::Impl::NrmInf for rank == 2.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define FLARE_BLAS_NRMINF_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE,           \
                                         MEM_SPACE)                            \
  template struct NrmInf<                                                      \
      EXEC_SPACE,                                                              \
      flare::View<                                                            \
          typename flare::detail::InnerProductSpaceTraits<SCALAR>::mag_type, \
          LAYOUT, flare::HostSpace,                                           \
          flare::MemoryTraits<flare::Unmanaged> >,                           \
      flare::View<const SCALAR*, LAYOUT,                                      \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                      \
                   flare::MemoryTraits<flare::Unmanaged> >,                  \
      1>;

//
// Macro for definition of full specialization of
// flare::blas::Impl::NrmInf for rank == 2.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define FLARE_BLAS_NRMINF_MV_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, \
                                            MEM_SPACE)                  \
  template struct NrmInf<                                               \
      EXEC_SPACE,                                                       \
      flare::View<typename flare::detail::InnerProductSpaceTraits<   \
                       SCALAR>::mag_type*,                              \
                   LAYOUT,                                              \
                   flare::Device<flare::DefaultHostExecutionSpace,    \
                                  flare::HostSpace>,                   \
                   flare::MemoryTraits<flare::Unmanaged> >,           \
      flare::View<const SCALAR**, LAYOUT,                              \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,               \
                   flare::MemoryTraits<flare::Unmanaged> >,           \
      2>;

#endif  // FLARE_KERNEL_BLAS_NRMINF_IMPL_H_
