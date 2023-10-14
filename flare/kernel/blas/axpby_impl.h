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

#ifndef FLARE_KERNEL_BLAS_AXPBY_IMPL_H_
#define FLARE_KERNEL_BLAS_AXPBY_IMPL_H_

#include <flare/core.h>
#include <flare/kernel/common/inner_product_space_traits.h>

#ifndef FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY
#define FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY 2
#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY

namespace flare::blas::detail {

    //
    // axpby
    //

    // Single-vector version of Axpby_MV_Functor.  The definition
    // immediately below lets a and b both be 1-D Tensors (and only requires
    // that each have one entry).  Following this is a partial
    // specialization that lets both of them be scalars.  This functor
    // computes any of the following:
    //
    // 1. Y(i) = alpha*X(i) + beta*Y(i) for alpha,beta in -1,0,1
    // 2. Y(i) = a(0)*X(i) + beta*Y(i) for beta in -1,0,1
    // 3. Y(i) = alpha*X(i) + b(0)*Y(i) for alpha in -1,0,1
    // 4. Y(i) = a(0)*X(i) + b(0)*Y(i)
    //
    // The template parameters scalar_x and scalar_y correspond to alpha
    // resp. beta in the operation y = alpha*x + beta*y.  The values -1,
    // 0, and -1 correspond to literal values of those coefficients.  The
    // value 2 tells the functor to use the corresponding vector of
    // coefficients.  Any literal coefficient of zero has BLAS semantics
    // of ignoring the corresponding (multi)vector entry.  This does not
    // apply to coefficients in the a and b vectors, if they are used.
    template <class AV, class XV, class BV, class YV, int scalar_x, int scalar_y,
            class SizeType>
    struct Axpby_Functor {
        typedef typename YV::execution_space execution_space;
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename YV::non_const_value_type> ATS;

        XV m_x;
        YV m_y;
        AV m_a;
        BV m_b;

        Axpby_Functor(const XV& x, const YV& y, const AV& a, const BV& b,
                      const SizeType startingColumn)
                : m_x(x), m_y(y), m_a(a), m_b(b) {
            static_assert(flare::is_tensor<XV>::value,
                          "flare::blas::detail::"
                          "Axpby_Functor: X is not a flare::Tensor.");
            static_assert(flare::is_tensor<YV>::value,
                          "flare::blas::detail::"
                          "Axpby_Functor: Y is not a flare::Tensor.");
            static_assert(std::is_same<typename YV::value_type,
                                  typename YV::non_const_value_type>::value,
                          "flare::blas::detail::Axpby_Functor: Y is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            static_assert((int)YV::rank == (int)XV::rank,
                          "flare::blas::detail::"
                          "Axpby_Functor: X and Y must have the same rank.");
            static_assert(YV::rank == 1,
                          "flare::blas::detail::Axpby_Functor: "
                          "XV and YV must have rank 1.");

            if (startingColumn != 0) {
                m_a = flare::subtensor(
                        a, std::make_pair(startingColumn, SizeType(a.extent(0))));
                m_b = flare::subtensor(
                        b, std::make_pair(startingColumn, SizeType(b.extent(0))));
            }
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type& i) const {
            // scalar_x and scalar_y are compile-time constants (since they
            // are template parameters), so the compiler should evaluate these
            // branches at compile time.

#if FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY <= 2

            if (scalar_x == 0 && scalar_y == 0) {
                m_y(i) = ATS::zero();
            }
            if (scalar_x == 0 && scalar_y == 2) {
                m_y(i) = m_b(0) * m_y(i);
            }
            if (scalar_x == 2 && scalar_y == 0) {
                m_y(i) = m_a(0) * m_x(i);
            }
            if (scalar_x == 2 && scalar_y == 2) {
                m_y(i) = m_a(0) * m_x(i) + m_b(0) * m_y(i);
            }

#else  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2
            if (scalar_x == 0 && scalar_y == 0) {
                m_y(i) = ATS::zero();
            }
            if (scalar_x == 0 && scalar_y == -1) {
                m_y(i) = -m_y(i);
            }
            if (scalar_x == 0 && scalar_y == 1) {
              return;  // m_y(i) = m_y(i);
            }
            if (scalar_x == 0 && scalar_y == 2) {
              m_y(i) = m_b(0) * m_y(i);
            }
            if (scalar_x == -1 && scalar_y == 0) {
              m_y(i) = -m_x(i);
            }
            if (scalar_x == -1 && scalar_y == -1) {
              m_y(i) = -m_x(i) - m_y(i);
            }
            if (scalar_x == -1 && scalar_y == 1) {
              m_y(i) = -m_x(i) + m_y(i);
            }
            if (scalar_x == -1 && scalar_y == 2) {
              m_y(i) = -m_x(i) + m_b(0) * m_y(i);
            }
            if (scalar_x == 1 && scalar_y == 0) {
              m_y(i) = m_x(i);
            }
            if (scalar_x == 1 && scalar_y == -1) {
              m_y(i) = m_x(i) - m_y(i);
            }
            if (scalar_x == 1 && scalar_y == 1) {
              m_y(i) = m_x(i) + m_y(i);
            }
            if (scalar_x == 1 && scalar_y == 2) {
              m_y(i) = m_x(i) + m_b(0) * m_y(i);
            }
            if (scalar_x == 2 && scalar_y == 0) {
              m_y(i) = m_a(0) * m_x(i);
            }
            if (scalar_x == 2 && scalar_y == -1) {
              m_y(i) = m_a(0) * m_x(i) - m_y(i);
            }
            if (scalar_x == 2 && scalar_y == 1) {
              m_y(i) = m_a(0) * m_x(i) + m_y(i);
            }
            if (scalar_x == 2 && scalar_y == 2) {
              m_y(i) = m_a(0) * m_x(i) + m_b(0) * m_y(i);
            }

#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY
        }
    };

    // Partial specialization of Axpby_Functor that lets a and b be
    // scalars (rather than 1-D Tensors, as in the most general version
    // above).  This functor computes any of the following:
    //
    // 1. Y(i) = alpha*X(i) + beta*Y(i) for alpha,beta in -1,0,1
    // 2. Y(i) = a*X(i) + beta*Y(i) for beta in -1,0,1
    // 3. Y(i) = alpha*X(i) + b*Y(i) for alpha in -1,0,1
    // 4. Y(i) = a*X(i) + b*Y(i)
    //
    // The template parameters scalar_x and scalar_y correspond to alpha
    // resp. beta in the operation y = alpha*x + beta*y.  The values -1,
    // 0, and -1 correspond to literal values of those coefficients.  The
    // value 2 tells the functor to use the corresponding vector of
    // coefficients.  Any literal coefficient of zero has BLAS semantics
    // of ignoring the corresponding (multi)vector entry.  This does not
    // apply to coefficients in the a and b vectors, if they are used.
    template <class XV, class YV, int scalar_x, int scalar_y, class SizeType>
    struct Axpby_Functor<typename XV::non_const_value_type, XV,
            typename YV::non_const_value_type, YV, scalar_x, scalar_y,
            SizeType> {
        typedef typename YV::execution_space execution_space;
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename YV::non_const_value_type> ATS;

        XV m_x;
        YV m_y;
        const typename XV::non_const_value_type m_a;
        const typename YV::non_const_value_type m_b;

        Axpby_Functor(const XV& x, const YV& y,
                      const typename XV::non_const_value_type& a,
                      const typename YV::non_const_value_type& b,
                      const SizeType /* startingColumn */)
                : m_x(x), m_y(y), m_a(a), m_b(b) {
            static_assert(flare::is_tensor<XV>::value,
                          "flare::blas::detail::"
                          "Axpby_Functor: X is not a flare::Tensor.");
            static_assert(flare::is_tensor<YV>::value,
                          "flare::blas::detail::"
                          "Axpby_Functor: Y is not a flare::Tensor.");
            static_assert(std::is_same<typename YV::value_type,
                                  typename YV::non_const_value_type>::value,
                          "flare::blas::detail::Axpby_Functor: R is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            static_assert((int)YV::rank == (int)XV::rank,
                          "flare::blas::detail::"
                          "Axpby_Functor: X and Y must have the same rank.");
            static_assert(YV::rank == 1,
                          "flare::blas::detail::Axpby_Functor: "
                          "XV and YV must have rank 1.");
        }

        FLARE_INLINE_FUNCTION  void operator()(const size_type& i) const {
            // scalar_x and scalar_y are compile-time constants (since they
            // are template parameters), so the compiler should evaluate these
            // branches at compile time.

#if FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY <= 2

            if (scalar_x == 0 && scalar_y == 0) {
                m_y(i) = static_cast<typename YV::non_const_value_type>(ATS::zero());
            }
            if (scalar_x == 0 && scalar_y == 2) {
                m_y(i) = static_cast<typename YV::non_const_value_type>(m_b * m_y(i));
            }
            if (scalar_x == 2 && scalar_y == 0) {
                m_y(i) = static_cast<typename YV::non_const_value_type>(m_a * m_x(i));
            }
            if (scalar_x == 2 && scalar_y == 2) {
                m_y(i) = static_cast<typename YV::non_const_value_type>(m_a * m_x(i) +
                                                                        m_b * m_y(i));
            }

#else  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2

            if (scalar_x == 0 && scalar_y == 0) {
                m_y(i) = ATS::zero();
            }
            if (scalar_x == 0 && scalar_y == -1) {
              m_y(i) = -m_y(i);
            }
            if (scalar_x == 0 && scalar_y == 1) {
              return;  // m_y(i) = m_y(i);
            }
            if (scalar_x == 0 && scalar_y == 2) {
              m_y(i) = m_b * m_y(i);
            }
            if (scalar_x == -1 && scalar_y == 0) {
              m_y(i) = -m_x(i);
            }
            if (scalar_x == -1 && scalar_y == -1) {
              m_y(i) = -m_x(i) - m_y(i);
            }
            if (scalar_x == -1 && scalar_y == 1) {
              m_y(i) = -m_x(i) + m_y(i);
            }
            if (scalar_x == -1 && scalar_y == 2) {
              m_y(i) = -m_x(i) + m_b * m_y(i);
            }
            if (scalar_x == 1 && scalar_y == 0) {
              m_y(i) = m_x(i);
            }
            if (scalar_x == 1 && scalar_y == -1) {
              m_y(i) = m_x(i) - m_y(i);
            }
            if (scalar_x == 1 && scalar_y == 1) {
              m_y(i) = m_x(i) + m_y(i);
            }
            if (scalar_x == 1 && scalar_y == 2) {
              m_y(i) = m_x(i) + m_b * m_y(i);
            }
            if (scalar_x == 2 && scalar_y == 0) {
              m_y(i) = m_a * m_x(i);
            }
            if (scalar_x == 2 && scalar_y == -1) {
              m_y(i) = m_a * m_x(i) - m_y(i);
            }
            if (scalar_x == 2 && scalar_y == 1) {
              m_y(i) = m_a * m_x(i) + m_y(i);
            }
            if (scalar_x == 2 && scalar_y == 2) {
              m_y(i) = m_a * m_x(i) + m_b * m_y(i);
            }

#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY
        }
    };

    // Variant of Axpby_MV_Generic for single vectors (1-D Tensors) x and y.
    // As above, either av and bv are both 1-D Tensors (and only the first
    // entry of each will be read), or both av and bv are scalars.
    //
    // This takes the starting column, so that if av and bv are both 1-D
    // Tensors, then the functor can take a subtensor if appropriate.
    template <class execution_space, class AV, class XV, class BV, class YV,
            class SizeType>
    void Axpby_Generic(const execution_space& space, const AV& av, const XV& x,
                       const BV& bv, const YV& y, const SizeType startingColumn,
                       int a = 2, int b = 2) {
        static_assert(flare::is_tensor<XV>::value,
                      "flare::blas::detail::"
                      "Axpby_Generic: X is not a flare::Tensor.");
        static_assert(flare::is_tensor<YV>::value,
                      "flare::blas::detail::"
                      "Axpby_Generic: Y is not a flare::Tensor.");
        static_assert(std::is_same<typename YV::value_type,
                              typename YV::non_const_value_type>::value,
                      "flare::blas::detail::Axpby_Generic: Y is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");
        static_assert((int)YV::rank == (int)XV::rank,
                      "flare::blas::detail::"
                      "Axpby_Generic: X and Y must have the same rank.");
        static_assert(YV::rank == 1,
                      "flare::blas::detail::Axpby_Generic: "
                      "XV and YV must have rank 1.");

        const SizeType numRows = x.extent(0);
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        if (a == 0 && b == 0) {
            Axpby_Functor<AV, XV, BV, YV, 0, 0, SizeType> op(x, y, av, bv,
                                                             startingColumn);
            flare::parallel_for("flare::blas::Axpby::S0", policy, op);
            return;
        }

#if FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2
        if (a == 0 && b == -1) {
            Axpby_Functor<AV, XV, BV, YV, 0, -1, SizeType> op(x, y, av, bv, startingColumn);
            flare::parallel_for("flare::blas::Axpby::S1", policy, op);
            return;
        }
      if (a == 0 && b == 1) {
            Axpby_Functor<AV, XV, BV, YV, 0, 1, SizeType> op(x, y, av, bv, startingColumn);
            flare::parallel_for("flare::blas::Axpby::S2", policy, op);
            return;
      }
#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2

        if (a == 0 && b == 2) {
            Axpby_Functor<AV, XV, BV, YV, 0, 2, SizeType> op(x, y, av, bv, startingColumn);
            flare::parallel_for("flare::blas::Axpby::S3", policy, op);
            return;
        }

#if FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2
        // a == -1
        if (a == -1 && b == 0) {
            Axpby_Functor<AV, XV, BV, YV, -1, 0, SizeType> op(x, y, av, bv,
                                                              startingColumn);
            flare::parallel_for("flare::blas::Axpby::S4", policy, op);
            return;
        }
        if (a == -1 && b == -1) {
            Axpby_Functor<AV, XV, BV, YV, -1, -1, SizeType> op(x, y, av, bv, startingColumn);
            flare::parallel_for("flare::blas::Axpby::S5", policy, op);
            return;
        }
        if (a == -1 && b == 1) {
            Axpby_Functor<AV, XV, BV, YV, -1, 1, SizeType> op(x, y, av, bv,
                                                              startingColumn);
            flare::parallel_for("flare::blas::Axpby::S6", policy, op);
            return;
        }
        if (a == -1 && b == 2) {
            Axpby_Functor<AV, XV, BV, YV, -1, 2, SizeType> op(x, y, av, bv,
                                                              startingColumn);
            flare::parallel_for("flare::blas::Axpby::S7", policy, op);
            return;
        }
        // a == 1
        if (a == 1 && b == 0) {
            Axpby_Functor<AV, XV, BV, YV, 1, 0, SizeType> op(x, y, av, bv,
                                                             startingColumn);
            flare::parallel_for("flare::blas::Axpby::S8", policy, op);
            return;
        }
        if (a == 1 && b == -1) {
            Axpby_Functor<AV, XV, BV, YV, 1, -1, SizeType> op(x, y, av, bv,
                                                              startingColumn);
            flare::parallel_for("flare::blas::Axpby::S9", policy, op);
            return;
        }
        if (a == 1 && b == 1) {
            Axpby_Functor<AV, XV, BV, YV, 1, 1, SizeType> op(x, y, av, bv,
                                                             startingColumn);
            flare::parallel_for("flare::blas::Axpby::S10", policy, op);
            return;
        }
        if (a == 1 && b == 2) {
            Axpby_Functor<AV, XV, BV, YV, 1, 2, SizeType> op(x, y, av, bv,
                                                             startingColumn);
            flare::parallel_for("flare::blas::Axpby::S11", policy, op);
            return;
        }
#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2

        // a == 2
        if (a == 2 && b == 0) {
            Axpby_Functor<AV, XV, BV, YV, 2, 0, SizeType> op(x, y, av, bv,
                                                             startingColumn);
            flare::parallel_for("flare::blas::Axpby::S12", policy, op);
            return;
        }

#if FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2
        if (a == 2 && b == -1) {
            Axpby_Functor<AV, XV, BV, YV, 2, -1, SizeType> op(x, y, av, bv,
                                                              startingColumn);
            flare::parallel_for("flare::blas::Axpby::S13", policy, op);
            return;
          }
        if (a == 2 && b == 1) {
            Axpby_Functor<AV, XV, BV, YV, 2, 1, SizeType> op(x, y, av, bv,
                                                             startingColumn);
            flare::parallel_for("flare::blas::Axpby::S14", policy, op);
            return;
      }
#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2

        // a and b arbitrary (not -1, 0, or 1)
        Axpby_Functor<AV, XV, BV, YV, 2, 2, SizeType> op(x, y, av, bv,
                                                         startingColumn);
        flare::parallel_for("flare::blas::Axpby::S15", policy, op);
    }


    //
    // axpby
    //

    // Functor for multivectors X and Y and 1-D tensors a and b, that
    // computes any of the following:
    //
    // 1. Y(i,j) = alpha*X(i,j) + beta*Y(i,j) for alpha,beta in -1,0,1
    // 2. Y(i,j) = a(j)*X(i,j) + beta*Y(i,j) for beta in -1,0,1
    // 3. Y(i,j) = alpha*X(i,j) + beta*Y(i,j) for alpha in -1,0,1
    // 4. Y(i,j) = a(j)*X(i,j) + b(j)*Y(i,j)
    //
    // The template parameters scalar_x and scalar_y correspond to alpha
    // resp. beta in the operation y = alpha*x + beta*y.  The values -1,
    // 0, and -1 correspond to literal values of those coefficients.  The
    // value 2 tells the functor to use the corresponding vector of
    // coefficients.  Any literal coefficient of zero has BLAS semantics
    // of ignoring the corresponding (multi)vector entry.  This does not
    // apply to coefficients in the a and b vectors, if they are used.
    template <class AV, class XMV, class BV, class YMV, int scalar_x, int scalar_y,
            class SizeType = typename YMV::size_type>
    struct Axpby_MV_Functor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename YMV::non_const_value_type> ATS;

        const size_type numCols;
        XMV m_x;
        YMV m_y;
        AV m_a;
        BV m_b;

        Axpby_MV_Functor(const XMV& X, const YMV& Y, const AV& a, const BV& b)
                : numCols(X.extent(1)), m_x(X), m_y(Y), m_a(a), m_b(b) {
            // XMV and YMV must be flare::Tensor specializations.
            static_assert(flare::is_tensor<AV>::value,
                          "flare::blas::detail::"
                          "Axpby_MV_Functor: a is not a flare::Tensor.");
            static_assert(flare::is_tensor<XMV>::value,
                          "flare::blas::detail::"
                          "Axpby_MV_Functor: X is not a flare::Tensor.");
            static_assert(flare::is_tensor<BV>::value,
                          "flare::blas::detail::"
                          "Axpby_MV_Functor: b is not a flare::Tensor.");
            static_assert(flare::is_tensor<YMV>::value,
                          "flare::blas::detail::"
                          "Axpby_MV_Functor: Y is not a flare::Tensor.");
            // YMV must be nonconst (else it can't be an output argument).
            static_assert(std::is_same<typename YMV::value_type,
                                  typename YMV::non_const_value_type>::value,
                          "flare::blas::detail::Axpby_MV_Functor: Y is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            static_assert((int)YMV::rank == (int)XMV::rank,
                          "flare::blas::detail::Axpby_MV_Functor: "
                          "X and Y must have the same rank.");
            static_assert(YMV::rank == 2,
                          "flare::blas::detail::Axpby_MV_Functor: "
                          "XMV and YMV must have rank 2.");
            static_assert(AV::rank == 1,
                          "flare::blas::detail::Axpby_MV_Functor: "
                          "AV must have rank 1.");
            static_assert(BV::rank == 1,
                          "flare::blas::detail::Axpby_MV_Functor: "
                          "BV must have rank 1.");
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type& i) const {
            // scalar_x and scalar_y are compile-time constants (since they
            // are template parameters), so the compiler should evaluate these
            // branches at compile time.
            if (scalar_x == 0 && scalar_y == 0) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = ATS::zero();
                }
            }
            if (scalar_x == 0 && scalar_y == -1) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = -m_y(i, k);
                }
            }
            if (scalar_x == 0 && scalar_y == 1) {
                return;  // Y(i,j) := Y(i,j)
            }
            if (scalar_x == 0 && scalar_y == 2) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_b(k) * m_y(i, k);
                }
            }
            if (scalar_x == -1 && scalar_y == 0) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = -m_x(i, k);
                }
            }
            if (scalar_x == -1 && scalar_y == -1) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = -m_x(i, k) - m_y(i, k);
                }
            }
            if (scalar_x == -1 && scalar_y == 1) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = -m_x(i, k) + m_y(i, k);
                }
            }
            if (scalar_x == -1 && scalar_y == 2) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = -m_x(i, k) + m_b(k) * m_y(i, k);
                }
            }
            if (scalar_x == 1 && scalar_y == 0) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_x(i, k);
                }
            }
            if (scalar_x == 1 && scalar_y == -1) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_x(i, k) - m_y(i, k);
                }
            }
            if (scalar_x == 1 && scalar_y == 1) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_x(i, k) + m_y(i, k);
                }
            }
            if (scalar_x == 1 && scalar_y == 2) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_x(i, k) + m_b(k) * m_y(i, k);
                }
            }
            if (scalar_x == 2 && scalar_y == 0) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_a(k) * m_x(i, k);
                }
            }
            if (scalar_x == 2 && scalar_y == -1) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_a(k) * m_x(i, k) - m_y(i, k);
                }
            }
            if (scalar_x == 2 && scalar_y == 1) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_a(k) * m_x(i, k) + m_y(i, k);
                }
            }
            if (scalar_x == 2 && scalar_y == 2) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_a(k) * m_x(i, k) + m_b(k) * m_y(i, k);
                }
            }
        }
    };

    // Variant of Axpby_MV_Functor, where a and b are scalars.
    // This functor computes any of the following:
    //
    // 1. Y(i,j) = alpha*X(i,j) + beta*Y(i,j) for alpha,beta in -1,0,1
    // 2. Y(i,j) = a*X(i,j) + beta*Y(i,j) for beta in -1,0,1
    // 3. Y(i,j) = alpha*X(i,j) + beta*Y(i,j) for alpha in -1,0,1
    // 4. Y(i,j) = a*X(i,j) + b*Y(i,j)
    //
    // The template parameters scalar_x and scalar_y correspond to alpha
    // resp. beta in the operation y = alpha*x + beta*y.  The values -1,
    // 0, and -1 correspond to literal values of those coefficients.  The
    // value 2 tells the functor to use the corresponding vector of
    // coefficients.  Any literal coefficient of zero has BLAS semantics
    // of ignoring the corresponding (multi)vector entry.  This does not
    // apply to coefficients in the a and b vectors, if they are used.
    //
    // This version works by partial specialization on AV and BV.
    // In this partial specialization, both AV and BV are scalars.
    template <class XMV, class YMV, int scalar_x, int scalar_y, class SizeType>
    struct Axpby_MV_Functor<typename XMV::non_const_value_type, XMV,
            typename YMV::non_const_value_type, YMV, scalar_x,
            scalar_y, SizeType> {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename YMV::non_const_value_type> ATS;

        const size_type numCols;
        XMV m_x;
        YMV m_y;
        const typename XMV::non_const_value_type m_a;
        const typename YMV::non_const_value_type m_b;

        Axpby_MV_Functor(const XMV& X, const YMV& Y,
                         const typename XMV::non_const_value_type& a,
                         const typename YMV::non_const_value_type& b)
                : numCols(X.extent(1)), m_x(X), m_y(Y), m_a(a), m_b(b) {
            static_assert(flare::is_tensor<XMV>::value,
                          "flare::blas::detail::"
                          "Axpby_MV_Functor: X is not a flare::Tensor.");
            static_assert(flare::is_tensor<YMV>::value,
                          "flare::blas::detail::"
                          "Axpby_MV_Functor: Y is not a flare::Tensor.");
            static_assert(std::is_same<typename YMV::value_type,
                                  typename YMV::non_const_value_type>::value,
                          "flare::blas::detail::Axpby_MV_Functor: Y is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            static_assert((int)YMV::rank == (int)XMV::rank,
                          "flare::blas::detail::"
                          "Axpby_MV_Functor: X and Y must have the same rank.");
            static_assert(YMV::rank == 2,
                          "flare::blas::detail::Axpby_MV_Functor: "
                          "XMV and YMV must have rank 2.");
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type& i) const {
            // scalar_x and scalar_y are compile-time constants (since they
            // are template parameters), so the compiler should evaluate these
            // branches at compile time.
            if (scalar_x == 0 && scalar_y == 0) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = ATS::zero();
                }
            }
            if (scalar_x == 0 && scalar_y == -1) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = -m_y(i, k);
                }
            }
            if (scalar_x == 0 && scalar_y == 1) {
                return;  // Y(i,j) := Y(i,j)
            }
            if (scalar_x == 0 && scalar_y == 2) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_b * m_y(i, k);
                }
            }
            if (scalar_x == -1 && scalar_y == 0) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = -m_x(i, k);
                }
            }
            if (scalar_x == -1 && scalar_y == -1) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = -m_x(i, k) - m_y(i, k);
                }
            }
            if (scalar_x == -1 && scalar_y == 1) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = -m_x(i, k) + m_y(i, k);
                }
            }
            if (scalar_x == -1 && scalar_y == 2) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = -m_x(i, k) + m_b * m_y(i, k);
                }
            }
            if (scalar_x == 1 && scalar_y == 0) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_x(i, k);
                }
            }
            if (scalar_x == 1 && scalar_y == -1) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_x(i, k) - m_y(i, k);
                }
            }
            if (scalar_x == 1 && scalar_y == 1) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_x(i, k) + m_y(i, k);
                }
            }
            if (scalar_x == 1 && scalar_y == 2) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_x(i, k) + m_b * m_y(i, k);
                }
            }
            if (scalar_x == 2 && scalar_y == 0) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_a * m_x(i, k);
                }
            }
            if (scalar_x == 2 && scalar_y == -1) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_a * m_x(i, k) - m_y(i, k);
                }
            }
            if (scalar_x == 2 && scalar_y == 1) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_a * m_x(i, k) + m_y(i, k);
                }
            }
            if (scalar_x == 2 && scalar_y == 2) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_y(i, k) = m_a * m_x(i, k) + m_b * m_y(i, k);
                }
            }
        }
    };

    // Column-unrolled variant of Axpby_MV_Functor.  The number of columns
    // in X and Y, UNROLL, is a compile-time constant.
    template <class AV, class XMV, class BV, class YMV, int scalar_x, int scalar_y,
            int UNROLL, class SizeType>
    struct Axpby_MV_Unroll_Functor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename YMV::non_const_value_type> ATS;

        XMV m_x;
        YMV m_y;
        AV m_a;
        BV m_b;

        Axpby_MV_Unroll_Functor(const XMV& x, const YMV& y, const AV& a, const BV& b,
                                const SizeType startingColumn)
                : m_x(x), m_y(y), m_a(a), m_b(b) {
            static_assert(flare::is_tensor<AV>::value,
                          "flare::blas::detail::"
                          "Axpby_MV_Unroll_Functor: a is not a flare::Tensor.");
            static_assert(flare::is_tensor<XMV>::value,
                          "flare::blas::detail::"
                          "Axpby_MV_Unroll_Functor: X is not a flare::Tensor.");
            static_assert(flare::is_tensor<BV>::value,
                          "flare::blas::detail::"
                          "Axpby_MV_Unroll_Functor: b is not a flare::Tensor.");
            static_assert(flare::is_tensor<YMV>::value,
                          "flare::blas::detail::"
                          "Axpby_MV_Unroll_Functor: Y is not a flare::Tensor.");
            static_assert(std::is_same<typename YMV::value_type,
                                  typename YMV::non_const_value_type>::value,
                          "flare::blas::detail::Axpby_MV_Unroll_Functor: Y is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            static_assert((int)YMV::rank == (int)XMV::rank,
                          "flare::blas::detail::Axpby_MV_Unroll_Functor: "
                          "X and Y must have the same rank.");
            static_assert(YMV::rank == 2,
                          "flare::blas::detail::Axpby_MV_Unroll_Functor: "
                          "XMV and YMV must have rank 2.");
            static_assert(AV::rank == 1,
                          "flare::blas::detail::Axpby_MV_Unroll_Functor: "
                          "AV must have rank 1.");
            static_assert(BV::rank == 1,
                          "flare::blas::detail::Axpby_MV_Unroll_Functor: "
                          "BV must have rank 1.");

            if (startingColumn != 0) {
                m_a = flare::subtensor(
                        a, std::make_pair(startingColumn, SizeType(a.extent(0))));
                m_b = flare::subtensor(
                        b, std::make_pair(startingColumn, SizeType(b.extent(0))));
            }
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type& i) const {
            // scalar_x and scalar_y are compile-time constants (since they
            // are template parameters), so the compiler should evaluate these
            // branches at compile time.

#if FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY <= 2

            if (scalar_x == 0 && scalar_y == 0) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                for (int k = 0; k < UNROLL; ++k) {
                    m_y(i, k) = ATS::zero();
                }
            }
            if (scalar_x == 0 && scalar_y == 2) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                for (int k = 0; k < UNROLL; ++k) {
                    m_y(i, k) = m_b(k) * m_y(i, k);
                }
            }
            if (scalar_x == 2 && scalar_y == 0) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                for (int k = 0; k < UNROLL; ++k) {
                    m_y(i, k) = m_a(k) * m_x(i, k);
                }
            }
            if (scalar_x == 2 && scalar_y == 2) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                for (int k = 0; k < UNROLL; ++k) {
                    m_y(i, k) = m_a(k) * m_x(i, k) + m_b(k) * m_y(i, k);
                }
            }

#else  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY >= 3

            if (scalar_x == 0 && scalar_y == 0) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = ATS::zero();
      }
    }
    if (scalar_x == 0 && scalar_y == -1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = -m_y(i, k);
      }
    }
    if (scalar_x == 0 && scalar_y == 1) {
      return;  // Y(i,j) := Y(i,j)
    }
    if (scalar_x == 0 && scalar_y == 2) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_b(k) * m_y(i, k);
      }
    }
    if (scalar_x == -1 && scalar_y == 0) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = -m_x(i, k);
      }
    }
    if (scalar_x == -1 && scalar_y == -1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = -m_x(i, k) - m_y(i, k);
      }
    }
    if (scalar_x == -1 && scalar_y == 1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = -m_x(i, k) + m_y(i, k);
      }
    }
    if (scalar_x == -1 && scalar_y == 2) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = -m_x(i, k) + m_b(k) * m_y(i, k);
      }
    }
    if (scalar_x == 1 && scalar_y == 0) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_x(i, k);
      }
    }
    if (scalar_x == 1 && scalar_y == -1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_x(i, k) - m_y(i, k);
      }
    }
    if (scalar_x == 1 && scalar_y == 1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_x(i, k) + m_y(i, k);
      }
    }
    if (scalar_x == 1 && scalar_y == 2) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_x(i, k) + m_b(k) * m_y(i, k);
      }
    }
    if (scalar_x == 2 && scalar_y == 0) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_a(k) * m_x(i, k);
      }
    }
    if (scalar_x == 2 && scalar_y == -1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_a(k) * m_x(i, k) - m_y(i, k);
      }
    }
    if (scalar_x == 2 && scalar_y == 1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_a(k) * m_x(i, k) + m_y(i, k);
      }
    }
    if (scalar_x == 2 && scalar_y == 2) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_a(k) * m_x(i, k) + m_b(k) * m_y(i, k);
      }
    }
#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY
        }
    };

// Variant of Axpby_MV_Unroll_Functor for single coefficients (rather
// than vectors of coefficients) a and b.  The number of columns in X
// and Y, UNROLL, is a compile-time constant.
    template <class XMV, class YMV, int scalar_x, int scalar_y, int UNROLL,
            class SizeType>
    struct Axpby_MV_Unroll_Functor<typename XMV::non_const_value_type, XMV,
            typename YMV::non_const_value_type, YMV,
            scalar_x, scalar_y, UNROLL, SizeType> {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename YMV::non_const_value_type> ATS;

        XMV m_x;
        YMV m_y;
        const typename XMV::non_const_value_type m_a;
        const typename YMV::non_const_value_type m_b;

        Axpby_MV_Unroll_Functor(const XMV& X, const YMV& Y,
                                const typename XMV::non_const_value_type& a,
                                const typename YMV::non_const_value_type& b,
                                const SizeType /* startingColumn */)
                : m_x(X), m_y(Y), m_a(a), m_b(b) {
            static_assert(flare::is_tensor<XMV>::value,
                          "flare::blas::detail::"
                          "Axpby_MV_Unroll_Functor: X is not a flare::Tensor.");
            static_assert(flare::is_tensor<YMV>::value,
                          "flare::blas::detail::"
                          "Axpby_MV_Unroll_Functor: Y is not a flare::Tensor.");
            static_assert(std::is_same<typename YMV::value_type,
                                  typename YMV::non_const_value_type>::value,
                          "flare::blas::detail::Axpby_MV_Unroll_Functor: Y is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            static_assert((int)YMV::rank == (int)XMV::rank,
                          "flare::blas::detail::"
                          "Axpby_MV_Unroll_Functor: X and Y must have the same rank.");
            static_assert(YMV::rank == 2,
                          "flare::blas::detail::Axpby_MV_Unroll_Functor: "
                          "XMV and YMV must have rank 2.");
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type& i) const {
            // scalar_x and scalar_y are compile-time constants (since they
            // are template parameters), so the compiler should evaluate these
            // branches at compile time.

#if FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY <= 2

            if (scalar_x == 0 && scalar_y == 0) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                for (int k = 0; k < UNROLL; ++k) {
                    m_y(i, k) = ATS::zero();
                }
            }
            if (scalar_x == 0 && scalar_y == 2) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                for (int k = 0; k < UNROLL; ++k) {
                    m_y(i, k) = m_b * m_y(i, k);
                }
            }
            if (scalar_x == 2 && scalar_y == 0) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                for (int k = 0; k < UNROLL; ++k) {
                    m_y(i, k) = m_a * m_x(i, k);
                }
            }
            if (scalar_x == 2 && scalar_y == 2) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                for (int k = 0; k < UNROLL; ++k) {
                    m_y(i, k) = m_a * m_x(i, k) + m_b * m_y(i, k);
                }
            }

#else  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2

            if (scalar_x == 0 && scalar_y == 0) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = ATS::zero();
      }
    }
    if (scalar_x == 0 && scalar_y == -1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = -m_y(i, k);
      }
    }
    if (scalar_x == 0 && scalar_y == 1) {
      return;  // Y(i,j) := Y(i,j)
    }
    if (scalar_x == 0 && scalar_y == 2) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_b * m_y(i, k);
      }
    }
    if (scalar_x == -1 && scalar_y == 0) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = -m_x(i, k);
      }
    }
    if (scalar_x == -1 && scalar_y == -1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = -m_x(i, k) - m_y(i, k);
      }
    }
    if (scalar_x == -1 && scalar_y == 1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = -m_x(i, k) + m_y(i, k);
      }
    }
    if (scalar_x == -1 && scalar_y == 2) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = -m_x(i, k) + m_b * m_y(i, k);
      }
    }
    if (scalar_x == 1 && scalar_y == 0) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_x(i, k);
      }
    }
    if (scalar_x == 1 && scalar_y == -1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_x(i, k) - m_y(i, k);
      }
    }
    if (scalar_x == 1 && scalar_y == 1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_x(i, k) + m_y(i, k);
      }
    }
    if (scalar_x == 1 && scalar_y == 2) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_x(i, k) + m_b * m_y(i, k);
      }
    }
    if (scalar_x == 2 && scalar_y == 0) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_a * m_x(i, k);
      }
    }
    if (scalar_x == 2 && scalar_y == -1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_a * m_x(i, k) - m_y(i, k);
      }
    }
    if (scalar_x == 2 && scalar_y == 1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_a * m_x(i, k) + m_y(i, k);
      }
    }
    if (scalar_x == 2 && scalar_y == 2) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        m_y(i, k) = m_a * m_x(i, k) + m_b * m_y(i, k);
      }
    }

#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY
        }
    };

    // Invoke the unrolled multivector functor that computes any of the
    // following:
    //
    // 1. Y(i,j) = a*X(i,j) + b*Y(i,j) for a,b in -1,0,1
    // 2. Y(i,j) = av(j)*X(i,j) + b*Y(i,j) for b in -1,0,1
    // 3. Y(i,j) = a*X(i,j) + b*Y(i,j) for a in -1,0,1
    // 4. Y(i,j) = av(j)*X(i,j) + bv(j)*Y(i,j)
    //
    // a and b come in as integers.  The values -1, 0, and 1 correspond to
    // the literal values of the coefficients.  The value 2 tells the
    // functor to use the corresponding vector of coefficients: a == 2
    // means use av, and b == 2 means use bv.  Otherwise, av resp. vb are
    // ignored.
    //
    // Any literal coefficient of zero has BLAS semantics of ignoring the
    // corresponding (multi)vector entry.  This does NOT apply to
    // coefficients in av and bv vectors, if they are used.
    //
    // Either av and bv are both 1-D Tensors, or av and bv are both scalars.
    template <class execution_space, class AV, class XMV, class BV, class YMV,
            int UNROLL, class SizeType>
    void Axpby_MV_Unrolled(const execution_space& space, const AV& av, const XMV& x,
                           const BV& bv, const YMV& y,
                           const SizeType startingColumn, int a = 2, int b = 2) {
        static_assert(flare::is_tensor<XMV>::value,
                      "flare::blas::detail::"
                      "Axpby_MV_Unrolled: X is not a flare::Tensor.");
        static_assert(flare::is_tensor<YMV>::value,
                      "flare::blas::detail::"
                      "Axpby_MV_Unrolled: Y is not a flare::Tensor.");
        static_assert(std::is_same<typename YMV::value_type,
                              typename YMV::non_const_value_type>::value,
                      "flare::blas::detail::Axpby_MV_Unrolled: Y is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");
        static_assert((int)YMV::rank == (int)XMV::rank,
                      "flare::blas::detail::"
                      "Axpby_MV_Unrolled: X and Y must have the same rank.");
        static_assert(YMV::rank == 2,
                      "flare::blas::detail::Axpby_MV_Unrolled: "
                      "XMV and YMV must have rank 2.");

        const SizeType numRows = x.extent(0);
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        if (a == 0 && b == 0) {
            Axpby_MV_Unroll_Functor<AV, XMV, BV, YMV, 0, 0, UNROLL, SizeType> op(
                    x, y, av, bv, startingColumn);
            flare::parallel_for("flare::blas::Axpby::MV::S0", policy, op);
            return;
        }

#if FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2
        if (a == 0 && b == -1) {
    Axpby_MV_Unroll_Functor<AV, XMV, BV, YMV, 0, -1, UNROLL, SizeType> op(
        x, y, av, bv, startingColumn);
    flare::parallel_for("flare::blas::Axpby::MV::S1", policy, op);
    return;
  }
  if (a == 0 && b == 1) {
    Axpby_MV_Unroll_Functor<AV, XMV, BV, YMV, 0, 1, UNROLL, SizeType> op(
        x, y, av, bv, startingColumn);
    flare::parallel_for("flare::blas::Axpby::MV::S2", policy, op);
    return;
  }
#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY

        if (a == 0 && b == 2) {
            Axpby_MV_Unroll_Functor<AV, XMV, BV, YMV, 0, 2, UNROLL, SizeType> op(
                    x, y, av, bv, startingColumn);
            flare::parallel_for("flare::blas::Axpby::MV::S3", policy, op);
            return;
        }

#if FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2
        // a == -1
  if (a == -1 && b == 0) {
    Axpby_MV_Unroll_Functor<AV, XMV, BV, YMV, -1, 0, UNROLL, SizeType> op(
        x, y, av, bv, startingColumn);
    flare::parallel_for("flare::blas::Axpby::MV::S4", policy, op);
    return;
  }
  if (a == -1 && b == -1) {
    Axpby_MV_Unroll_Functor<AV, XMV, BV, YMV, -1, -1, UNROLL, SizeType> op(
        x, y, av, bv, startingColumn);
    flare::parallel_for("flare::blas::Axpby::MV::S5", policy, op);
    return;
  }
  if (a == -1 && b == 1) {
    Axpby_MV_Unroll_Functor<AV, XMV, BV, YMV, -1, 1, UNROLL, SizeType> op(
        x, y, av, bv, startingColumn);
    flare::parallel_for("flare::blas::Axpby::MV::S6", policy, op);
    return;
  }
  if (a == -1 && b == 2) {
    Axpby_MV_Unroll_Functor<AV, XMV, BV, YMV, -1, 2, UNROLL, SizeType> op(
        x, y, av, bv, startingColumn);
    flare::parallel_for("flare::blas::Axpby::MV::S7", policy, op);
    return;
  }
  // a == 1
  if (a == 1 && b == 0) {
    Axpby_MV_Unroll_Functor<AV, XMV, BV, YMV, 1, 0, UNROLL, SizeType> op(
        x, y, av, bv, startingColumn);
    flare::parallel_for("flare::blas::Axpby::MV::S8", policy, op);
    return;
  }
  if (a == 1 && b == -1) {
    Axpby_MV_Unroll_Functor<AV, XMV, BV, YMV, 1, -1, UNROLL, SizeType> op(
        x, y, av, bv, startingColumn);
    flare::parallel_for("flare::blas::Axpby::MV::S9", policy, op);
    return;
  }
  if (a == 1 && b == 1) {
    Axpby_MV_Unroll_Functor<AV, XMV, BV, YMV, 1, 1, UNROLL, SizeType> op(
        x, y, av, bv, startingColumn);
    flare::parallel_for("flare::blas::Axpby::MV::S10", policy, op);
    return;
  }
  if (a == 1 && b == 2) {
    Axpby_MV_Unroll_Functor<AV, XMV, BV, YMV, 1, 2, UNROLL, SizeType> op(
        x, y, av, bv, startingColumn);
    flare::parallel_for("flare::blas::Axpby::MV::S11", policy, op);
    return;
  }
#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2

        // a == 2
        if (a == 2 && b == 0) {
            Axpby_MV_Unroll_Functor<AV, XMV, BV, YMV, 2, 0, UNROLL, SizeType> op(
                    x, y, av, bv, startingColumn);
            flare::parallel_for("flare::blas::Axpby::MV::S12", policy, op);
            return;
        }

#if FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2
        if (a == 2 && b == -1) {
            Axpby_MV_Unroll_Functor<AV, XMV, BV, YMV, 2, -1, UNROLL, SizeType> op(
                x, y, av, bv, startingColumn);
            flare::parallel_for("flare::blas::Axpby::MV::S13", policy, op);
            return;
        }
        if (a == 2 && b == 1) {
            Axpby_MV_Unroll_Functor<AV, XMV, BV, YMV, 2, 1, UNROLL, SizeType> op(
                x, y, av, bv, startingColumn);
            flare::parallel_for("flare::blas::Axpby::MV::S14", policy, op);
            return;
        }
#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2

        // a and b arbitrary (not -1, 0, or 1)
        Axpby_MV_Unroll_Functor<AV, XMV, BV, YMV, 2, 2, UNROLL, SizeType> op(x, y, av, bv, startingColumn);
        flare::parallel_for("flare::blas::Axpby::MV::S15", policy, op);
    }

    // Invoke the "generic" (not unrolled) multivector functor that
    // computes any of the following:
    //
    // 1. Y(i,j) = a*X(i,j) + b*Y(i,j) for a,b in -1,0,1
    // 2. Y(i,j) = av(j)*X(i,j) + b*Y(i,j) for b in -1,0,1
    // 3. Y(i,j) = a*X(i,j) + b*Y(i,j) for a in -1,0,1
    // 4. Y(i,j) = av(j)*X(i,j) + bv(j)*Y(i,j)
    //
    // a and b come in as integers.  The values -1, 0, and 1 correspond to
    // the literal values of the coefficients.  The value 2 tells the
    // functor to use the corresponding vector of coefficients: a == 2
    // means use av, and b == 2 means use bv.  Otherwise, av resp. vb are
    // ignored.
    //
    // Any literal coefficient of zero has BLAS semantics of ignoring the
    // corresponding (multi)vector entry.  This does NOT apply to
    // coefficients in av and bv vectors, if they are used.
    //
    // Either av and bv are both 1-D Tensors, or av and bv are both scalars.
    template <class execution_space, class AV, class XMV, class BV, class YMV,
            class SizeType>
    void Axpby_MV_Generic(const execution_space& space, const AV& av, const XMV& x,
                          const BV& bv, const YMV& y, int a = 2, int b = 2) {
        static_assert(flare::is_tensor<XMV>::value,
                      "flare::blas::detail::"
                      "Axpby_MV_Generic: X is not a flare::Tensor.");
        static_assert(flare::is_tensor<YMV>::value,
                      "flare::blas::detail::"
                      "Axpby_MV_Generic: Y is not a flare::Tensor.");
        static_assert(std::is_same<typename YMV::value_type,
                              typename YMV::non_const_value_type>::value,
                      "flare::blas::detail::Axpby_MV_Generic: Y is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");
        static_assert((int)YMV::rank == (int)XMV::rank,
                      "flare::blas::detail::"
                      "Axpby_MV_Generic: X and Y must have the same rank.");
        static_assert(YMV::rank == 2,
                      "flare::blas::detail::Axpby_MV_Generic: "
                      "XMV and YMV must have rank 2.");

        const SizeType numRows = x.extent(0);
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        if (a == 0 && b == 0) {
            Axpby_MV_Functor<AV, XMV, BV, YMV, 0, 0, SizeType> op(x, y, av, bv);
            flare::parallel_for("flare::blas::Axpby::MV::S16", policy, op);
            return;
        }

#if FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2
        if (a == 0 && b == -1) {
            Axpby_MV_Functor<AV, XMV, BV, YMV, 0, -1, SizeType> op(x, y, av, bv);
            flare::parallel_for("flare::blas::Axpby::MV::S17", policy, op);
            return;
        }
        if (a == 0 && b == 1) {
            Axpby_MV_Functor<AV, XMV, BV, YMV, 0, 1, SizeType> op(x, y, av, bv);
            flare::parallel_for("flare::blas::Axpby::MV::S18", policy, op);
            return;
        }
#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2

        if (a == 0 && b == 2) {
            Axpby_MV_Functor<AV, XMV, BV, YMV, 0, 2, SizeType> op(x, y, av, bv);
            flare::parallel_for("flare::blas::Axpby::MV::S19", policy, op);
            return;
        }

#if FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2
        // a == -1
        if (a == -1 && b == 0) {
            Axpby_MV_Functor<AV, XMV, BV, YMV, -1, 0, SizeType> op(x, y, av, bv);
            flare::parallel_for("flare::blas::Axpby::MV::S20", policy, op);
            return;
        }
        if (a == -1 && b == -1) {
            Axpby_MV_Functor<AV, XMV, BV, YMV, -1, -1, SizeType> op(x, y, av, bv);
            flare::parallel_for("flare::blas::Axpby::MV::S21", policy, op);
            return;
        }
        if (a == -1 && b == 1) {
            Axpby_MV_Functor<AV, XMV, BV, YMV, -1, 1, SizeType> op(x, y, av, bv);
            flare::parallel_for("flare::blas::Axpby::MV::S22", policy, op);
            return;
        }
        if (a == -1 && b == 2) {
            Axpby_MV_Functor<AV, XMV, BV, YMV, -1, 2, SizeType> op(x, y, av, bv);
            flare::parallel_for("flare::blas::Axpby::MV::S23", policy, op);
            return;
        }
        // a == 1
        if (a == 1 && b == 0) {
            Axpby_MV_Functor<AV, XMV, BV, YMV, 1, 0, SizeType> op(x, y, av, bv);
            flare::parallel_for("flare::blas::Axpby::MV::S24", policy, op);
            return;
        }
        if (a == 1 && b == -1) {
            Axpby_MV_Functor<AV, XMV, BV, YMV, 1, -1, SizeType> op(x, y, av, bv);
            flare::parallel_for("flare::blas::Axpby::MV::S25", policy, op);
            return;
        }
        if (a == 1 && b == 1) {
            Axpby_MV_Functor<AV, XMV, BV, YMV, 1, 1, SizeType> op(x, y, av, bv);
            flare::parallel_for("flare::blas::Axpby::MV::S26", policy, op);
            return;
        }
        if (a == 1 && b == 2) {
            Axpby_MV_Functor<AV, XMV, BV, YMV, 1, 2, SizeType> op(x, y, av, bv);
            flare::parallel_for("flare::blas::Axpby::MV::S27", policy, op);
            return;
        }
#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2

        // a == 2
        if (a == 2 && b == 0) {
            Axpby_MV_Functor<AV, XMV, BV, YMV, 2, 0, SizeType> op(x, y, av, bv);
            flare::parallel_for("flare::blas::Axpby::MV::S28", policy, op);
            return;
        }

#if FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2
        if (a == 2 && b == -1) {
            Axpby_MV_Functor<AV, XMV, BV, YMV, 2, -1, SizeType> op(x, y, av, bv);
            flare::parallel_for("flare::blas::Axpby::MV::S29", policy, op);
            return;
        }
        if (a == 2 && b == 1) {
            Axpby_MV_Functor<AV, XMV, BV, YMV, 2, 1, SizeType> op(x, y, av, bv);
            flare::parallel_for("flare::blas::Axpby::MV::S30", policy, op);
            return;
        }
#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2

        // a and b arbitrary (not -1, 0, or 1)
        Axpby_MV_Functor<AV, XMV, BV, YMV, 2, 2, SizeType> op(x, y, av, bv);
        flare::parallel_for("flare::blas::Axpby::MV::S31", policy, op);
    }

    // Compute any of the following, in a way optimized for X and Y
    // being LayoutLeft:
    //
    // 1. Y(i,j) = a*X(i,j) + b*Y(i,j) for a,b in -1,0,1
    // 2. Y(i,j) = av(j)*X(i,j) + b*Y(i,j) for b in -1,0,1
    // 3. Y(i,j) = a*X(i,j) + b*Y(i,j) for a in -1,0,1
    // 4. Y(i,j) = av(j)*X(i,j) + bv(j)*Y(i,j)
    //
    // a and b come in as integers.  The values -1, 0, and 1 correspond to
    // the literal values of the coefficients.  The value 2 tells the
    // functor to use the corresponding vector of coefficients: a == 2
    // means use av, and b == 2 means use bv.  Otherwise, av resp. vb are
    // ignored.
    //
    // Any literal coefficient of zero has BLAS semantics of ignoring the
    // corresponding (multi)vector entry.  This does NOT apply to
    // coefficients in av and bv vectors, if they are used.
    //
    // Either av and bv are both 1-D Tensors, or av and bv are both scalars.
    template <class execution_space, class AV, class XMV, class BV, class YMV,
            class SizeType>
    struct Axpby_MV_Invoke_Left {
        static void run(const execution_space& space, const AV& av, const XMV& x,
                        const BV& bv, const YMV& y, int a = 2, int b = 2) {
            static_assert(flare::is_tensor<XMV>::value,
                          "flare::blas::detail::"
                          "Axpby_MV_Invoke_Left: X is not a flare::Tensor.");
            static_assert(flare::is_tensor<YMV>::value,
                          "flare::blas::detail::"
                          "Axpby_MV_Invoke_Left: Y is not a flare::Tensor.");
            static_assert(std::is_same<typename YMV::value_type,
                                  typename YMV::non_const_value_type>::value,
                          "flare::blas::detail::Axpby_MV_Invoke_Left: Y is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            static_assert((int)YMV::rank == (int)XMV::rank,
                          "flare::blas::detail::"
                          "Axpby_MV_Invoke_Left: X and Y must have the same rank.");
            static_assert(YMV::rank == 2,
                          "flare::blas::detail::Axpby_MV_Invoke_Left: "
                          "X and Y must have rank 2.");

            const SizeType numCols = x.extent(1);

            // Strip-mine by 8, then 4.  After that, do one column at a time.
            // We limit the number of strip-mine values in order to keep down
            // the amount of code to compile.
            SizeType j = 0;
            for (; j + 8 <= numCols; j += 8) {
                XMV X_cur = flare::subtensor(x, flare::ALL(), std::make_pair(j, j + 8));
                YMV Y_cur = flare::subtensor(y, flare::ALL(), std::make_pair(j, j + 8));

                // Passing in the starting column index lets the functor take
                // subtensors of av and bv, if they are Tensors.  If they are scalars,
                // the functor doesn't have to do anything to them.
                Axpby_MV_Unrolled<execution_space, AV, XMV, BV, YMV, 8, SizeType>(
                        space, av, X_cur, bv, Y_cur, j, a, b);
            }
            for (; j + 4 <= numCols; j += 4) {
                XMV X_cur = flare::subtensor(x, flare::ALL(), std::make_pair(j, j + 4));
                YMV Y_cur = flare::subtensor(y, flare::ALL(), std::make_pair(j, j + 4));

                // Passing in the starting column index lets the functor take
                // subtensors of av and bv, if they are Tensors.  If they are scalars,
                // the functor doesn't have to do anything to them.
                Axpby_MV_Unrolled<execution_space, AV, XMV, BV, YMV, 4, SizeType>(
                        space, av, X_cur, bv, Y_cur, j, a, b);
            }
            for (; j < numCols; ++j) {
                auto x_cur = flare::subtensor(x, flare::ALL(), j);
                auto y_cur = flare::subtensor(y, flare::ALL(), j);

                // Passing in the starting column index lets the functor take
                // subtensors of av and bv, if they are Tensors.  If they are scalars,
                // the functor doesn't have to do anything to them.
                typedef decltype(x_cur) XV;
                typedef decltype(y_cur) YV;
                Axpby_Generic<execution_space, AV, XV, BV, YV, SizeType>(
                        space, av, x_cur, bv, y_cur, j, a, b);
            }
        }
    };

    // Compute any of the following, in a way optimized for X, Y, and R
    // being LayoutRight:
    //
    // 1. Y(i,j) = a*X(i,j) + b*Y(i,j) for a,b in -1,0,1
    // 2. Y(i,j) = av(j)*X(i,j) + b*Y(i,j) for b in -1,0,1
    // 3. Y(i,j) = a*X(i,j) + b*Y(i,j) for a in -1,0,1
    // 4. Y(i,j) = av(j)*X(i,j) + bv(j)*Y(i,j)
    //
    // a and b come in as integers.  The values -1, 0, and 1 correspond to
    // the literal values of the coefficients.  The value 2 tells the
    // functor to use the corresponding vector of coefficients: a == 2
    // means use av, and b == 2 means use bv.  Otherwise, av resp. vb are
    // ignored.
    //
    // Any literal coefficient of zero has BLAS semantics of ignoring the
    // corresponding (multi)vector entry.  This does NOT apply to
    // coefficients in av and bv vectors, if they are used.
    //
    // Either av and bv are both 1-D Tensors, or av and bv are both scalars.
    template <class execution_space, class AV, class XMV, class BV, class YMV,
            class SizeType>
    struct Axpby_MV_Invoke_Right {
        static void run(const execution_space& space, const AV& av, const XMV& x,
                        const BV& bv, const YMV& y, int a = 2, int b = 2) {
            static_assert(flare::is_tensor<XMV>::value,
                          "flare::blas::detail::"
                          "Axpby_MV_Invoke_Right: X is not a flare::Tensor.");
            static_assert(flare::is_tensor<YMV>::value,
                          "flare::blas::detail::"
                          "Axpby_MV_Invoke_Right: Y is not a flare::Tensor.");
            static_assert(std::is_same<typename YMV::value_type,
                                  typename YMV::non_const_value_type>::value,
                          "flare::blas::detail::Axpby_MV_Invoke_Right: Y is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            static_assert((int)YMV::rank == (int)XMV::rank,
                          "flare::blas::detail::"
                          "Axpby_MV_Invoke_Right: X and Y must have the same rank.");
            static_assert(YMV::rank == 2,
                          "flare::blas::detail::Axpby_MV_Invoke_Right: "
                          "X and Y must have rank 2.");

            const SizeType numCols = x.extent(1);
            if (numCols == 1) {
                auto x_0 = flare::subtensor(x, flare::ALL(), 0);
                auto y_0 = flare::subtensor(y, flare::ALL(), 0);
                typedef decltype(x_0) XV;
                typedef decltype(y_0) YV;
                Axpby_Generic<execution_space, AV, XV, BV, YV, SizeType>(
                        space, av, x_0, bv, y_0, 0, a, b);
            } else {
                Axpby_MV_Generic<execution_space, AV, XMV, BV, YMV, SizeType>(
                        space, av, x, bv, y, a, b);
            }
        }
    };


    //
    // axpby
    //

    /// \brief Implementation of flare::blas::axpby for (multi)vectors.
    ///
    /// Compute any of the following, depending on the types of the input
    /// arguments of axpxy():
    ///
    /// 1. Y(i,j) = av(j)*X(i,j) + bv(j)*Y(i,j) (if R, X, and Y are 2-D,
    ///    and av and bv are 1-D)
    ///
    /// 2. Y(i,j) = av*X(i,j) + bv*Y(i,j) (if R, X, and Y are 2-D,
    ///    and av and bv are scalars)
    ///
    /// 3. Y(i) = av()*X(i) + bv()*Y(i) (if R, X, and Y are 1-D, and av
    ///    and bv are 0-D Tensors (not scalars))
    ///
    /// 4. Y(i) = av*X(i) + bv*Y(i) (if R, X, and Y are 1-D, and av and bv
    ///    are scalars)
    ///
    /// Any <i>scalar</i> coefficient of zero has BLAS semantics of
    /// ignoring the corresponding (multi)vector entry.  This does NOT
    /// apply to coefficients in av and bv vectors, if they are used.
    template <class execution_space, class AV, class XMV, class BV, class YMV,
            int rank = YMV::rank>
    struct Axpby {
        static void axpby(const execution_space& space, const AV& av, const XMV& X,
                          const BV& bv, const YMV& Y);
    };

    template <class execution_space, class AV, class XMV, class BV, class YMV>
    struct Axpby<execution_space, AV, XMV, BV, YMV, 0> {
        static void axpby(const execution_space& /*space*/, const AV& /* av */,
                          const XMV& /* X */, const BV& /* bv */,
                          const YMV& /* Y */) {
            static_assert(YMV::rank == 0, "Oh My God");
        }
    };

    // Full specialization for XMV and YMV rank-2 Tensors.
    template <class execution_space, class AV, class XMV, class BV, class YMV>
    struct Axpby<execution_space, AV, XMV, BV, YMV, 2> {
        typedef typename YMV::size_type size_type;

        static void axpby(const execution_space& space, const AV& av, const XMV& X,
                          const BV& bv, const YMV& Y) {
            static_assert(flare::is_tensor<XMV>::value,
                          "flare::blas::detail::"
                          "Axpby<rank-2>::axpby: X is not a flare::Tensor.");
            static_assert(flare::is_tensor<YMV>::value,
                          "flare::blas::detail::"
                          "Axpby<rank-2>::axpby: Y is not a flare::Tensor.");
            static_assert(std::is_same<typename YMV::value_type,
                                  typename YMV::non_const_value_type>::value,
                          "flare::blas::detail::Axpby<rank-2>::axpby: Y is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            static_assert((int)YMV::rank == (int)XMV::rank,
                          "flare::blas::detail::Axpby<rank-2>::axpby (MV): "
                          "X and Y must have the same rank.");
            static_assert(YMV::rank == 2,
                          "flare::blas::detail::Axpby<rank-2>::axpby: "
                          "X and Y must have rank 2.");
            flare::Profiling::pushRegion("flare::blas::axpby");

            const size_type numRows = X.extent(0);
            const size_type numCols = X.extent(1);
            int a = 2, b = 2;
            if (av.extent(0) == 0) {
                a = 0;
            }
            if (bv.extent(0) == 0) {
                b = 0;
            }

            if (numRows < static_cast<size_type>(INT_MAX) &&
                numRows * numCols < static_cast<size_type>(INT_MAX)) {
                typedef int index_type;
                typedef typename std::conditional<
                        std::is_same<typename XMV::array_layout, flare::LayoutLeft>::value,
                        Axpby_MV_Invoke_Right<execution_space, AV, XMV, BV, YMV, index_type>,
                        Axpby_MV_Invoke_Left<execution_space, AV, XMV, BV, YMV,
                                index_type> >::type Axpby_MV_Invoke_Layout;
                Axpby_MV_Invoke_Layout::run(space, av, X, bv, Y, a, b);
            } else {
                typedef typename XMV::size_type index_type;
                typedef typename std::conditional<
                        std::is_same<typename XMV::array_layout, flare::LayoutLeft>::value,
                        Axpby_MV_Invoke_Right<execution_space, AV, XMV, BV, YMV, index_type>,
                        Axpby_MV_Invoke_Left<execution_space, AV, XMV, BV, YMV,
                                index_type> >::type Axpby_MV_Invoke_Layout;
                Axpby_MV_Invoke_Layout::run(space, av, X, bv, Y, a, b);
            }
            flare::Profiling::popRegion();
        }
    };

    // Partial specialization for XMV, and YMV rank-2 Tensors,
    // and AV and BV scalars.
    template <class execution_space, class XMV, class YMV>
    struct Axpby<execution_space, typename XMV::non_const_value_type, XMV,
            typename YMV::non_const_value_type, YMV, 2> {
        typedef typename XMV::non_const_value_type AV;
        typedef typename YMV::non_const_value_type BV;
        typedef typename YMV::size_type size_type;
        typedef flare::ArithTraits<typename XMV::non_const_value_type> ATA;
        typedef flare::ArithTraits<typename YMV::non_const_value_type> ATB;

        static void axpby(const execution_space& space, const AV& alpha, const XMV& X,
                          const BV& beta, const YMV& Y) {
            static_assert(flare::is_tensor<XMV>::value,
                          "flare::blas::detail::Axpby::axpby (MV): "
                          "X is not a flare::Tensor.");
            static_assert(flare::is_tensor<YMV>::value,
                          "flare::blas::detail::Axpby::axpby (MV): "
                          "Y is not a flare::Tensor.");
            static_assert(std::is_same<typename YMV::value_type,
                                  typename YMV::non_const_value_type>::value,
                          "flare::blas::detail::Axpby::axpby (MV): Y is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            static_assert((int)YMV::rank == (int)XMV::rank,
                          "flare::blas::detail::Axpby::axpby (MV): "
                          "X and Y must have the same rank.");
            static_assert(YMV::rank == 2,
                          "flare::blas::detail::Axpby::axpby (MV): "
                          "X and Y must have rank 2.");
            flare::Profiling::pushRegion("flare::blas::axpby");

            const size_type numRows = X.extent(0);
            const size_type numCols = X.extent(1);
            int a, b;
            if (alpha == ATA::zero()) {
                a = 0;
            }
#if FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2
            else if (alpha == -ATA::one()) {
                a = -1;
            } else if (alpha == ATA::one()) {
              a = 1;
            }
#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2
            else {
                a = 2;
            }
            if (beta == ATB::zero()) {
                b = 0;
            }
#if FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2
            else if (beta == -ATB::one()) {
                b = -1;
            } else if (beta == ATB::one()) {
              b = 1;
            }
#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2
            else {
                b = 2;
            }

            if (numRows < static_cast<size_type>(INT_MAX) &&
                numRows * numCols < static_cast<size_type>(INT_MAX)) {
                typedef int index_type;
                typedef typename std::conditional<
                        std::is_same<typename XMV::array_layout, flare::LayoutLeft>::value,
                        Axpby_MV_Invoke_Right<execution_space, AV, XMV, BV, YMV, index_type>,
                        Axpby_MV_Invoke_Left<execution_space, AV, XMV, BV, YMV,
                                index_type> >::type Axpby_MV_Invoke_Layout;
                Axpby_MV_Invoke_Layout::run(space, alpha, X, beta, Y, a, b);
            } else {
                typedef typename XMV::size_type index_type;
                typedef typename std::conditional<
                        std::is_same<typename XMV::array_layout, flare::LayoutLeft>::value,
                        Axpby_MV_Invoke_Right<execution_space, AV, XMV, BV, YMV, index_type>,
                        Axpby_MV_Invoke_Left<execution_space, AV, XMV, BV, YMV,
                                index_type> >::type Axpby_MV_Invoke_Layout;
                Axpby_MV_Invoke_Layout::run(space, alpha, X, beta, Y, a, b);
            }
            flare::Profiling::popRegion();
        }
    };

    // Partial specialization for XV and YV rank-1 Tensors,
    // and AV and BV scalars.
    template <class execution_space, class XV, class YV>
    struct Axpby<execution_space, typename XV::non_const_value_type, XV,
            typename YV::non_const_value_type, YV, 1> {
        typedef typename XV::non_const_value_type AV;
        typedef typename YV::non_const_value_type BV;
        typedef typename YV::size_type size_type;
        typedef flare::ArithTraits<typename XV::non_const_value_type> ATA;
        typedef flare::ArithTraits<typename YV::non_const_value_type> ATB;

        static void axpby(const execution_space& space, const AV& alpha, const XV& X,
                          const BV& beta, const YV& Y) {
            static_assert(flare::is_tensor<XV>::value,
                          "flare::blas::detail::"
                          "Axpby<rank-1>::axpby: X is not a flare::Tensor.");
            static_assert(flare::is_tensor<YV>::value,
                          "flare::blas::detail::"
                          "Axpby<rank-1>::axpby: Y is not a flare::Tensor.");
            static_assert(std::is_same<typename YV::value_type,
                                  typename YV::non_const_value_type>::value,
                          "flare::blas::detail::Axpby<rank-1>::axpby: Y is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            static_assert((int)YV::rank == (int)XV::rank,
                          "flare::blas::detail::"
                          "Axpby<rank-1>::axpby: X and Y must have the same rank.");
            static_assert(YV::rank == 1,
                          "flare::blas::detail::Axpby<rank-1>::axpby: "
                          "X and Y must have rank 1.");

            flare::Profiling::pushRegion("flare::blas::axpby");

            const size_type numRows = X.extent(0);
            int a                   = 2;
            if (alpha == ATA::zero()) {
                a = 0;
            }
#if FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2
            else if (alpha == -ATA::one()) {\
                a = -1;
            } else if (alpha == ATA::one()) {
                a = 1;
            }
#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2

            int b = 2;
            if (beta == ATB::zero()) {
                b = 0;
            }
#if FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2
            else if (beta == -ATB::one()) {
                b = -1;
            } else if (beta == ATB::one()) {
                b = 1;
            }
#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_AXPBY > 2

            if (numRows < static_cast<size_type>(INT_MAX)) {
                typedef int index_type;
                Axpby_Generic<execution_space, typename XV::non_const_value_type, XV,
                        typename YV::non_const_value_type, YV, index_type>(
                        space, alpha, X, beta, Y, 0, a, b);
            } else {
                typedef typename XV::size_type index_type;
                Axpby_Generic<execution_space, typename XV::non_const_value_type, XV,
                        typename YV::non_const_value_type, YV, index_type>(
                        space, alpha, X, beta, Y, 0, a, b);
            }
            flare::Profiling::popRegion();
        }
    };

    ///
    /// Serial Internal Impl
    /// ====================
    template <typename ScalarType, typename ValueType>
    FLARE_INLINE_FUNCTION static void serial_axpy(
            const int m, const ScalarType alpha, const ValueType *FLARE_RESTRICT X,
            /* */ ValueType *FLARE_RESTRICT Y, const int xs0, const int ys0) {
#if defined(FLARE_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for (int i = 0; i < m; ++i) Y[i * ys0] += alpha * X[i * xs0];

        return;
    }

    template <typename ScalarType, typename ValueType>
    FLARE_INLINE_FUNCTION static void serial_axpy_mv(
            const int m, const int n, const ScalarType alpha,
            const ValueType *FLARE_RESTRICT X,
            /* */ ValueType *FLARE_RESTRICT Y, const int xs0, const int xs1,
            const int ys0, const int ys1) {
        if (xs0 > xs1) {
            for (int i = 0; i < m; ++i)
                serial_axpy(n, alpha, X + i * xs0, Y + i * ys0, xs1, ys1);
        } else {
            for (int j = 0; j < n; ++j)
                serial_axpy(m, alpha, X + j * xs1, Y + j * ys1, xs0, ys0);
        }

        return;
    }


}  // namespace flare::blas::detail

#define FLARE_BLAS_AXPBY_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  template struct Axpby<                                                       \
      EXEC_SPACE, SCALAR,                                                      \
      flare::Tensor<const SCALAR*, LAYOUT,                                      \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                      \
                   flare::MemoryTraits<flare::Unmanaged> >,                  \
      SCALAR,                                                                  \
      flare::Tensor<SCALAR*, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>,     \
                   flare::MemoryTraits<flare::Unmanaged> >,                  \
      1>;

#define FLARE_BLAS_AXPBY_MV_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE,      \
                                           MEM_SPACE)                       \
  template struct Axpby<                                                    \
      EXEC_SPACE, SCALAR,                                                   \
      flare::Tensor<const SCALAR**, LAYOUT,                                  \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                   \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      SCALAR,                                                               \
      flare::Tensor<SCALAR**, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>, \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      2>;                                                                  \
  template struct Axpby<                                                    \
      EXEC_SPACE,                                                           \
      flare::Tensor<const SCALAR*, LAYOUT,                                   \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                   \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      flare::Tensor<const SCALAR**, LAYOUT,                                  \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                   \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      flare::Tensor<const SCALAR*, LAYOUT,                                   \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                   \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      flare::Tensor<SCALAR**, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>, \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      2>;

#endif  // FLARE_KERNEL_BLAS_AXPBY_IMPL_H_
