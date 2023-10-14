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

#ifndef FLARE_KERNEL_BLAS_MULTI_IMPL_H_
#define FLARE_KERNEL_BLAS_MULTI_IMPL_H_

#include <flare/core/arith_traits.h>
#include <flare/core.h>

namespace flare::blas::detail {

    /// \brief Functor for entry-wise multiply of multivectors.
    ///
    /// \tparam CMV 2-D flare::Tensor
    /// \tparam AV 1-D flare::Tensor
    /// \tparam BMV 2-D flare::Tensor
    /// \tparam scalar_ab 0 if ab is zero, else nonzero (preferably 2).
    /// \tparam scalar_c 0 if c is zero, else nonzero (preferably 2).
    /// \tparam SizeType Index type for iterating over rows.
    ///
    /// C(i,j) = c * C(i,j) + ab * A(i) * B(i,j), subject to the usual
    /// BLAS update rules.
    template <class CMV, class AV, class BMV, int scalar_ab, int scalar_c,
            class SizeType = typename CMV::size_type>
    struct MV_MultFunctor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename CMV::non_const_value_type> ATS;

        const size_type m_n;
        typename CMV::const_value_type m_c;
        CMV m_C;
        typename AV::const_value_type m_ab;
        AV m_A;
        BMV m_B;

        MV_MultFunctor(typename CMV::const_value_type& c, const CMV& C,
                       typename AV::const_value_type& ab, const AV& A, const BMV& B)
                : m_n(C.extent(1)), m_c(c), m_C(C), m_ab(ab), m_A(A), m_B(B) {}

        FLARE_INLINE_FUNCTION void operator()(const size_type& i) const {
            if (scalar_c == 0) {
                if (scalar_ab == 0) {
                    for (size_type j = 0; j < m_n; ++j) {
                        m_C(i, j) = ATS::zero();
                    }
                } else {  // ab != 0, c == 0
                    typename AV::const_value_type Ai = m_A(i);
                    for (size_type j = 0; j < m_n; ++j) {
                        m_C(i, j) = m_ab * Ai * m_B(i, j);
                    }
                }
            } else {  // c != 0
                if (scalar_ab == 0) {
                    for (size_type j = 0; j < m_n; ++j) {
                        m_C(i, j) = m_c * m_C(i, j);
                    }
                } else {  // m_ab != 0, and m_c != 0
                    typename AV::const_value_type Ai = m_A(i);
                    for (size_type j = 0; j < m_n; ++j) {
                        m_C(i, j) = m_c * m_C(i, j) + m_ab * Ai * m_B(i, j);
                    }
                }
            }
        }
    };

    /// \brief Functor for entry-wise multiply of vectors.
    ///
    /// \tparam CV 1-D flare::Tensor
    /// \tparam AV 1-D flare::Tensor
    /// \tparam BV 1-D flare::Tensor
    /// \tparam scalar_ab 0 if ab is zero, else nonzero (preferably 2).
    /// \tparam scalar_c 0 if c is zero, else nonzero (preferably 2).
    /// \tparam SizeType Index type for iterating over rows.
    ///
    /// C(i) = c * C(i) + ab * A(i) * B(i), subject to the usual
    /// BLAS update rules.
    template <class CV, class AV, class BV, int scalar_ab, int scalar_c,
            class SizeType = typename CV::size_type>
    struct V_MultFunctor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename CV::non_const_value_type> ATS;

        typename CV::const_value_type m_c;
        CV m_C;
        typename AV::const_value_type m_ab;
        AV m_A;
        BV m_B;

        V_MultFunctor(typename CV::const_value_type& c, const CV& C,
                      typename AV::const_value_type& ab, const AV& A, const BV& B)
                : m_c(c), m_C(C), m_ab(ab), m_A(A), m_B(B) {}

        FLARE_INLINE_FUNCTION void operator()(const size_type& i) const {
            if (scalar_c == 0) {
                if (scalar_ab == 0) {
                    m_C(i) = ATS::zero();
                } else {  // ab != 0, c == 0
                    m_C(i) = m_ab * m_A(i) * m_B(i);
                }
            } else {  // c != 0
                if (scalar_ab == 0) {
                    m_C(i) = m_c * m_C(i);
                } else {  // m_ab != 0, and m_c != 0
                    m_C(i) = m_c * m_C(i) + m_ab * m_A(i) * m_B(i);
                }
            }
        }
    };

    /// \brief Implementation of entry-wise multiply of vectors, that
    ///   dispatches to the right functor invocation.
    ///
    /// \tparam CV 1-D flare::Tensor
    /// \tparam AV 1-D flare::Tensor
    /// \tparam BV 1-D flare::Tensor
    /// \tparam SizeType Index type for iterating over rows.
    ///
    /// C(i) = c * C(i) + ab * A(i) * B(i), subject to the usual BLAS
    /// update rules.
    template <class execution_space, class CV, class AV, class BV, class SizeType>
    void V_Mult_Generic(const execution_space& space,
                        typename CV::const_value_type& c, const CV& C,
                        typename AV::const_value_type& ab, const AV& A,
                        const BV& B) {
        using flare::ALL;
        using flare::subtensor;
        typedef flare::ArithTraits<typename AV::non_const_value_type> ATA;
        typedef flare::ArithTraits<typename CV::non_const_value_type> ATC;

        const SizeType numRows = C.extent(0);
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        if (c == ATC::zero()) {
            if (ab == ATA::zero()) {
                typedef V_MultFunctor<CV, AV, BV, 0, 0, SizeType> functor_type;
                functor_type op(c, C, ab, A, B);
                flare::parallel_for("flare::blas::Mult::S0", policy, op);
            } else {
                typedef V_MultFunctor<CV, AV, BV, 2, 0, SizeType> functor_type;
                functor_type op(c, C, ab, A, B);
                flare::parallel_for("flare::blas::Mult::S1", policy, op);
            }
        } else {  // c != 0
            if (ab == ATA::zero()) {
                typedef V_MultFunctor<CV, AV, BV, 0, 2, SizeType> functor_type;
                functor_type op(c, C, ab, A, B);
                flare::parallel_for("flare::blas::Mult::S2", policy, op);
            } else {
                typedef V_MultFunctor<CV, AV, BV, 2, 2, SizeType> functor_type;
                functor_type op(c, C, ab, A, B);
                flare::parallel_for("flare::blas::Mult::S3", policy, op);
            }
        }
    }

    /// \brief Implementation of entry-wise multiply of multivectors, that
    ///   dispatches to the right functor invocation (or calls
    ///   V_Mult_Generic if C and B have one column).
    ///
    /// \tparam CMV 2-D flare::Tensor
    /// \tparam AV 1-D flare::Tensor
    /// \tparam BMV 2-D flare::Tensor
    /// \tparam SizeType Index type for iterating over rows.
    ///
    /// C(i,j) = c * C(i,j) + ab * A(i) * B(i,j), subject to the usual
    /// BLAS update rules.
    template <class execution_space, class CMV, class AV, class BMV, class SizeType>
    void MV_Mult_Generic(const execution_space& space,
                         typename CMV::const_value_type& c, const CMV& C,
                         typename AV::const_value_type& ab, const AV& A,
                         const BMV& B) {
        typedef flare::ArithTraits<typename AV::non_const_value_type> ATA;
        typedef flare::ArithTraits<typename CMV::non_const_value_type> ATC;

        if (C.extent(1) == 1) {
            auto C_0 = flare::subtensor(C, flare::ALL(), 0);
            auto B_0 = flare::subtensor(B, flare::ALL(), 0);
            typedef decltype(C_0) CV;
            typedef decltype(B_0) BV;

            V_Mult_Generic<execution_space, CV, AV, BV, SizeType>(space, c, C_0, ab, A,
                                                                  B_0);
            return;
        }

        const SizeType numRows = C.extent(0);
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        if (c == ATC::zero()) {
            if (ab == ATA::zero()) {
                typedef MV_MultFunctor<CMV, AV, BMV, 0, 0, SizeType> functor_type;
                functor_type op(c, C, ab, A, B);
                flare::parallel_for("flare::blas::Mult::S4", policy, op);
            } else {
                typedef MV_MultFunctor<CMV, AV, BMV, 2, 0, SizeType> functor_type;
                functor_type op(c, C, ab, A, B);
                flare::parallel_for("flare::blas::Mult::S5", policy, op);
            }
        } else {  // c != 0
            if (ab == ATA::zero()) {
                typedef MV_MultFunctor<CMV, AV, BMV, 0, 2, SizeType> functor_type;
                functor_type op(c, C, ab, A, B);
                flare::parallel_for("flare::blas::Mult::S6", policy, op);
            } else {
                typedef MV_MultFunctor<CMV, AV, BMV, 2, 2, SizeType> functor_type;
                functor_type op(c, C, ab, A, B);
                flare::parallel_for("flare::blas::Mult::S7", policy, op);
            }
        }
    }

    
    //
    // multi
    //
    /// \brief Implementation of entry-wise multiply of multivectors or
    ///   single vectors (depending on the rank template parameter).
    ///
    /// Compute
    ///
    /// Y(i,j) = alpha*A(i,j)*X(i,j) + gamma*Y(i,j)
    ///
    /// with special cases for alpha, or gamma = 0.
    template <class execution_space, class YMV, class AV, class XMV,
            int rank = XMV::rank>
    struct Mult {
        static void mult(const execution_space& space,
                         const typename YMV::non_const_value_type& gamma,
                         const YMV& Y,
                         const typename XMV::non_const_value_type& alpha, const AV& A,
                         const XMV& X);
    };

    // Partial specialization for YMV, AV, and XMV rank-2 Tensors.
    template <class execution_space, class YMV, class AV, class XMV>
    struct Mult<execution_space, YMV, AV, XMV, 2> {
        typedef typename YMV::size_type size_type;
        typedef typename YMV::non_const_value_type YMV_scalar;
        typedef typename XMV::non_const_value_type XMV_scalar;

        static void mult(const execution_space& space, const YMV_scalar& gamma,
                         const YMV& Y, const XMV_scalar& alpha, const AV& A,
                         const XMV& X) {
            static_assert(flare::is_tensor<YMV>::value,
                          "flare::blas::Impl::"
                          "Mult<rank 2>::mult: Y is not a flare::Tensor.");
            static_assert(flare::is_tensor<AV>::value,
                          "flare::blas::Impl::"
                          "Mult<rank 2>::mult: A is not a flare::Tensor.");
            static_assert(flare::is_tensor<XMV>::value,
                          "flare::blas::Impl::"
                          "Mult<rank 2>::mult: X is not a flare::Tensor.");
            static_assert(std::is_same<typename YMV::value_type,
                                  typename YMV::non_const_value_type>::value,
                          "flare::blas::Impl::Mult<rank 2>::mult: Y is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            // Casting to int avoids compiler warnings about comparing
            // different kinds of enum values.
            static_assert((int)XMV::rank == (int)YMV::rank && (int)XMV::rank == 2,
                          "flare::blas::Impl::Mult<rank 2>::mult: "
                          "X, and Y must have the rank 2.");
            static_assert(AV::rank == 1,
                          "flare::blas::Impl::Mult<rank 2>::mult: "
                          "AV must have rank 1.");
            flare::Profiling::pushRegion("flare::blas::multi");

            const size_type numRows = X.extent(0);
            const size_type numCols = X.extent(1);

            if (numRows < static_cast<int>(INT_MAX) &&
                numRows * numCols < static_cast<int>(INT_MAX)) {
                MV_Mult_Generic<execution_space, YMV, AV, XMV, int>(space, gamma, Y,
                                                                    alpha, A, X);
            } else {
                MV_Mult_Generic<execution_space, YMV, AV, XMV, int64_t>(space, gamma, Y,
                                                                        alpha, A, X);
            }
            flare::Profiling::popRegion();
        }
    };

    // Partial specialization for YV, AV, and XV rank-1 Tensors.
    template <class execution_space, class YV, class AV, class XV>
    struct Mult<execution_space, YV, AV, XV, 1> {
        typedef typename YV::size_type size_type;
        typedef typename YV::non_const_value_type YV_scalar;
        typedef typename XV::non_const_value_type XV_scalar;

        static void mult(const execution_space& space, const YV_scalar& gamma,
                         const YV& Y, const XV_scalar& alpha, const AV& A,
                         const XV& X) {
            // YV, AV, and XV must be flare::Tensor specializations.
            static_assert(flare::is_tensor<YV>::value,
                          "flare::blas::Impl::"
                          "Mult<rank 1>::mult: Y is not a flare::Tensor.");
            static_assert(flare::is_tensor<AV>::value,
                          "flare::blas::Impl::"
                          "Mult<rank 1>::mult: A is not a flare::Tensor.");
            static_assert(flare::is_tensor<XV>::value,
                          "flare::blas::Impl::"
                          "Mult<rank 1>::mult: X is not a flare::Tensor.");
            // XV must be nonconst (else it can't be an output argument).
            static_assert(std::is_same<typename YV::value_type,
                                  typename YV::non_const_value_type>::value,
                          "flare::blas::Impl::Mult<rank 1>::mult: Y is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            static_assert((int)XV::rank == (int)YV::rank && (int)AV::rank == 1,
                          "flare::blas::Impl::Mult<rank 1>::mult: "
                          "X, Y, and Z must have rank 1.");
            flare::Profiling::pushRegion("flare::blas::mult");

            const size_type numRows = Y.extent(0);
            if (numRows < static_cast<int>(INT_MAX)) {
                V_Mult_Generic<execution_space, YV, AV, XV, int>(space, gamma, Y, alpha,
                                                                 A, X);
            } else {
                V_Mult_Generic<execution_space, YV, AV, XV, int64_t>(space, gamma, Y,
                                                                     alpha, A, X);
            }
            flare::Profiling::popRegion();
        }
    };

}  // namespace flare::blas::detail

#define FLARE_BLAS_MULT_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  template struct Mult<                                                       \
      EXEC_SPACE,                                                             \
      flare::Tensor<SCALAR*, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>,    \
                   flare::MemoryTraits<flare::Unmanaged> >,                 \
      flare::Tensor<const SCALAR*, LAYOUT,                                     \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                     \
                   flare::MemoryTraits<flare::Unmanaged> >,                 \
      flare::Tensor<const SCALAR*, LAYOUT,                                     \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                     \
                   flare::MemoryTraits<flare::Unmanaged> >,                 \
      1>;

#define FLARE_BLAS_MULT_MV_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE,       \
                                          MEM_SPACE)                        \
  template struct Mult<                                                     \
      EXEC_SPACE,                                                           \
      flare::Tensor<SCALAR**, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>, \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      flare::Tensor<const SCALAR*, LAYOUT,                                   \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                   \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      flare::Tensor<const SCALAR**, LAYOUT,                                  \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                   \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      2>;

#endif  //  FLARE_KERNEL_BLAS_MULTI_IMPL_H_
