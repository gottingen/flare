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

#ifndef FLARE_KERNEL_BLAS_IAMAX_IMPL_H_
#define FLARE_KERNEL_BLAS_IAMAX_IMPL_H_

#include <flare/core.h>
#include <flare/kernel/common/utility.h>
#include <flare/kernel/common/helper.h>
#include <flare/kernel/common/inner_product_space_traits.h>
namespace flare::blas::detail {


    /// \brief Iamax functor for single vectors.
    ///
    /// \tparam RV 0-D output View
    /// \tparam XV 1-D input View
    /// \tparam MagType Magnitude type
    /// \tparam SizeType Index type.  Use int (32 bits) if possible.
    template <class RV, class XV, class MagType,
            class SizeType = typename XV::size_type>
    struct V_Iamax_Functor {
        using size_type   = SizeType;
        using mag_type    = MagType;
        using xvalue_type = typename XV::non_const_value_type;
        using IPT         = flare::detail::InnerProductSpaceTraits<xvalue_type>;
        using value_type  = typename RV::value_type;

        typename XV::const_type m_x;

        V_Iamax_Functor(const XV& x) : m_x(x) {
            static_assert(flare::is_view<RV>::value,
                          "flare::blas::detail::V_Iamax_Functor: "
                          "R is not a flare::View.");
            static_assert(flare::is_view<XV>::value,
                          "flare::blas::detail::V_Iamax_Functor: "
                          "X is not a flare::View.");
            static_assert(std::is_same<typename RV::value_type,
                                  typename RV::non_const_value_type>::value,
                          "flare::blas::detail::V_Iamax_Functor: R is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            static_assert(RV::rank == 0 && XV::rank == 1,
                          "flare::blas::detail::V_Iamax_Functor: "
                          "RV must have rank 0 and XV must have rank 1.");
        }

        FLARE_INLINE_FUNCTION void operator()(const size_type i,
                                               value_type& lmaxloc) const {
            mag_type val    = IPT::norm(m_x(i - 1));
            mag_type maxval = IPT::norm(m_x(lmaxloc - 1));
            if (val > maxval) lmaxloc = i;
        }

        FLARE_INLINE_FUNCTION void init(value_type& update) const {
            update = flare::reduction_identity<typename RV::value_type>::max() + 1;
        }

        FLARE_INLINE_FUNCTION void join(value_type& update,
                                         const value_type& source) const {
            mag_type source_val = IPT::norm(m_x(source - 1));
            mag_type update_val = IPT::norm(m_x(update - 1));
            if (update_val < source_val) update = source;
        }
    };

    /// \brief Find the index of the element with the maximum magnitude of the
    /// single vector (1-D
    ///   View) X, and store the result in the 0-D View r.
    template <class execution_space, class RV, class XV, class SizeType>
    void V_Iamax_Invoke(const execution_space& space, const RV& r, const XV& X) {
        using AT       = flare::ArithTraits<typename XV::non_const_value_type>;
        using mag_type = typename AT::mag_type;

        const SizeType numRows = static_cast<SizeType>(X.extent(0));

        // Avoid MaxLoc Reduction if this is a zero length view
        if (numRows == 0) {
            flare::deep_copy(space, r, 0u);
            return;
        }

        flare::RangePolicy<execution_space, SizeType> policy(space, 1, numRows + 1);

        using functor_type = V_Iamax_Functor<RV, XV, mag_type, SizeType>;
        functor_type op(X);
        flare::parallel_reduce("flare::blas::Iamax::S0", policy, op, r);
    }

    /// \brief Find the index of the element with the maximum magnitude of the
    /// columns of the
    ///   multivector (2-D View) X, and store result(s) in the 1-D View r.
    template <class execution_space, class RV, class XMV, class SizeType>
    void MV_Iamax_Invoke(const execution_space& space, const RV& r, const XMV& X) {
        for (size_t i = 0; i < X.extent(1); i++) {
            auto ri = flare::subview(r, i);
            auto Xi = flare::subview(X, flare::ALL(), i);
            V_Iamax_Invoke<execution_space, decltype(ri), decltype(Xi), SizeType>(
                    space, ri, Xi);
        }
    }


    // Unification layer
    template <class execution_space, class RMV, class XMV, int rank = XMV::rank>
    struct Iamax {
        static void iamax(const execution_space& space, const RMV& R, const XMV& X);
    };

    template <class execution_space, class RMV, class XMV>
    struct Iamax<execution_space, RMV, XMV, 1> {
        typedef typename XMV::size_type size_type;

        static void iamax(const execution_space& space, const RMV& R, const XMV& X) {
            static_assert(flare::is_view<RMV>::value,
                          "flare::blas::detail::"
                          "Iamax<1-D>: RMV is not a flare::View.");
            static_assert(flare::is_view<XMV>::value,
                          "flare::blas::detail::"
                          "Iamax<1-D>: XMV is not a flare::View.");
            static_assert(RMV::rank == 0,
                          "flare::blas::detail::Iamax<1-D>: "
                          "RMV is not rank 0.");
            static_assert(XMV::rank == 1,
                          "flare::blas::detail::Iamax<1-D>: "
                          "XMV is not rank 1.");
            flare::Profiling::pushRegion("flare::blas::iamax");
            const size_type numRows = X.extent(0);

            if (numRows < static_cast<size_type>(INT_MAX)) {
                V_Iamax_Invoke<execution_space, RMV, XMV, int>(space, R, X);
            } else {
                typedef std::int64_t index_type;
                V_Iamax_Invoke<execution_space, RMV, XMV, index_type>(space, R, X);
            }
            flare::Profiling::popRegion();
        }
    };

    template <class execution_space, class RV, class XMV>
    struct Iamax<execution_space, RV, XMV, 2> {
        typedef typename XMV::size_type size_type;

        static void iamax(const execution_space& space, const RV& R, const XMV& X) {
            static_assert(flare::is_view<RV>::value,
                          "flare::blas::detail::"
                          "Iamax<2-D>: RV is not a flare::View.");
            static_assert(flare::is_view<XMV>::value,
                          "flare::blas::detail::"
                          "Iamax<2-D>: XMV is not a flare::View.");
            static_assert(RV::rank == 1,
                          "flare::blas::detail::Iamax<2-D>: "
                          "RV is not rank 1.");
            static_assert(XMV::rank == 2,
                          "flare::blas::detail::Iamax<2-D>: "
                          "XMV is not rank 2.");
            flare::Profiling::pushRegion("flare::blas::iamax");
            const size_type numRows = X.extent(0);
            const size_type numCols = X.extent(1);
            if (numRows < static_cast<size_type>(INT_MAX) &&
                numRows * numCols < static_cast<size_type>(INT_MAX)) {
                MV_Iamax_Invoke<execution_space, RV, XMV, int>(space, R, X);
            } else {
                typedef std::int64_t index_type;
                MV_Iamax_Invoke<execution_space, RV, XMV, index_type>(space, R, X);
            }
            flare::Profiling::popRegion();
        }
    };

}  // namespace flare::blas::detail

//
// Macro for definition of full specialization of
// flare::blas::detail::Iamax for rank == 1.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define FLARE_BLAS_IAMAX_SPEC_INST_INDEX(INDEX_TYPE, SCALAR, LAYOUT,     \
                                              EXEC_SPACE, MEM_SPACE)          \
  template struct Iamax<                                                      \
      EXEC_SPACE,                                                             \
      flare::View<INDEX_TYPE, LAYOUT, flare::HostSpace,                     \
                   flare::MemoryTraits<flare::Unmanaged> >,                 \
      flare::View<const SCALAR*, LAYOUT,                                     \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                     \
                   flare::MemoryTraits<flare::Unmanaged> >,                 \
      1>;                                                                    \
  template struct Iamax<                                                      \
      EXEC_SPACE,                                                             \
      flare::View<INDEX_TYPE, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>, \
                   flare::MemoryTraits<flare::Unmanaged> >,                 \
      flare::View<const SCALAR*, LAYOUT,                                     \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                     \
                   flare::MemoryTraits<flare::Unmanaged> >,                 \
      1>;

#define FLARE_BLAS_IAMAX_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  FLARE_BLAS_IAMAX_SPEC_INST_INDEX(unsigned long, SCALAR, LAYOUT,         \
                                        EXEC_SPACE, MEM_SPACE)                 \
  FLARE_BLAS_IAMAX_SPEC_INST_INDEX(unsigned int, SCALAR, LAYOUT,          \
                                        EXEC_SPACE, MEM_SPACE)                 \
  FLARE_BLAS_IAMAX_SPEC_INST_INDEX(int, SCALAR, LAYOUT, EXEC_SPACE,       \
                                        MEM_SPACE)


//
// Macro for definition of full specialization of
// flare::blas::detail::Iamax for rank == 2.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define FLARE_BLAS_IAMAX_MV_SPEC_INST_INDEX(INDEX_TYPE, SCALAR, LAYOUT,   \
                                                 EXEC_SPACE, MEM_SPACE)        \
  template struct Iamax<                                                       \
      EXEC_SPACE,                                                              \
      flare::View<INDEX_TYPE*, LAYOUT, flare::HostSpace,                     \
                   flare::MemoryTraits<flare::Unmanaged> >,                  \
      flare::View<const SCALAR**, LAYOUT,                                     \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                      \
                   flare::MemoryTraits<flare::Unmanaged> >,                  \
      2>;                                                         \
  template struct Iamax<                                                       \
      EXEC_SPACE,                                                              \
      flare::View<INDEX_TYPE*, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>, \
                   flare::MemoryTraits<flare::Unmanaged> >,                  \
      flare::View<const SCALAR**, LAYOUT,                                     \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                      \
                   flare::MemoryTraits<flare::Unmanaged> >,                  \
      2>;

#define FLARE_BLAS_IAMAX_MV_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE,      \
                                           MEM_SPACE)                       \
  FLARE_BLAS_IAMAX_MV_SPEC_INST_INDEX(unsigned long, SCALAR, LAYOUT,   \
                                           EXEC_SPACE, MEM_SPACE)           \
  FLARE_BLAS_IAMAX_MV_SPEC_INST_INDEX(unsigned int, SCALAR, LAYOUT,    \
                                           EXEC_SPACE, MEM_SPACE)           \
  FLARE_BLAS_IAMAX_MV_SPEC_INST_INDEX(int, SCALAR, LAYOUT, EXEC_SPACE, \
                                           MEM_SPACE)


#endif  // FLARE_KERNEL_BLAS_IAMAX_IMPL_H_
