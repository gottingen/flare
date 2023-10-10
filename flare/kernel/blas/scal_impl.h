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

#ifndef FLARE_KERNEL_BLAS_SCAL_IMPL_H_
#define FLARE_KERNEL_BLAS_SCAL_IMPL_H_

#include <flare/core.h>
#include <flare/core/arith_traits.h>
#include <flare/kernel/common/helper.h>

#ifndef FLARE_BLAS_OPTIMIZATION_LEVEL_SCAL
#define FLARE_BLAS_OPTIMIZATION_LEVEL_SCAL 2
#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_SCAL

namespace flare::blas::detail {

    // Single-vector version of MV_Scal_Functor.  By default, a is still a
    // 1-D View.  Below is a partial specialization that lets a be a
    // scalar.  This functor computes any of the following:
    //
    // 1. Y(i) = alpha*X(i) for alpha in -1,0,1
    // 2. Y(i) = a(0)*X(i)
    //
    // The template parameter scalar_x corresponds to alpha in the
    // operation y = alpha*x + beta*y.  The values -1, 0, and -1
    // correspond to literal values of this coefficient.  The value 2
    // tells the functor to use the corresponding vector of coefficients.
    // Any literal coefficient of zero has BLAS semantics of ignoring the
    // corresponding (multi)vector entry.  This does not apply to
    // coefficients in the a vector, if used.
    template<class RV, class AV, class XV, int scalar_x, class SizeType>
    struct V_Scal_Functor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename RV::non_const_value_type> ATS;

        RV m_r;
        XV m_x;
        AV m_a;

        V_Scal_Functor(const RV &r, const XV &x, const AV &a,
                       const SizeType startingColumn)
                : m_r(r), m_x(x), m_a(a) {
            static_assert(flare::is_view<RV>::value,
                          "V_Scal_Functor: RV is not a flare::View.");
            static_assert(flare::is_view<AV>::value,
                          "V_Scal_Functor: AV is not a flare::View.");
            static_assert(flare::is_view<XV>::value,
                          "V_Scal_Functor: XV is not a flare::View.");
            static_assert(RV::rank == 1, "V_Scal_Functor: RV is not rank 1.");
            static_assert(AV::rank == 1, "V_Scal_Functor: AV is not rank 1.");
            static_assert(XV::rank == 1, "V_Scal_Functor: XV is not rank 1.");

            if (startingColumn != 0) {
                m_a = flare::subview(
                        a,
                        std::make_pair(startingColumn, static_cast<SizeType>(a.extent(0))));
            }
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i) const {
            // scalar_x is a compile-time constant (since it is a template
            // parameter), so the compiler should evaluate these branches at
            // compile time.
            if (scalar_x == 0) {
                m_r(i) = ATS::zero();
            }
            if (scalar_x == -1) {
                m_r(i) = -m_x(i);
            }
            if (scalar_x == 1) {
                m_r(i) = m_x(i);
            }
            if (scalar_x == 2) {
                m_r(i) = m_a(0) * m_x(i);
            }
        }
    };

    // Partial specialization of V_Scal_Functor that lets a be a scalar
    // (rather than a 1-D View, as in the most general version above).
    // This functor computes any of the following:
    //
    // 1. Y(i) = alpha*X(i) for alpha in -1,0,1
    // 2. Y(i) = a*X(i)
    template<class RV, class XV, int scalar_x, class SizeType>
    struct V_Scal_Functor<RV, typename XV::non_const_value_type, XV, scalar_x,
            SizeType> {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename RV::non_const_value_type> ATS;

        RV m_r;
        XV m_x;
        const typename XV::non_const_value_type m_a;

        V_Scal_Functor(const RV &r, const XV &x,
                       const typename XV::non_const_value_type &a,
                       const SizeType /* startingColumn */)
                : m_r(r), m_x(x), m_a(a) {}

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i) const {
            if (scalar_x == 0) {
                m_r(i) = ATS::zero();
            }
            if (scalar_x == -1) {
                m_r(i) = -m_x(i);
            }
            if (scalar_x == 1) {
                m_r(i) = m_x(i);
            }
            if (scalar_x == 2) {
                m_r(i) = m_a * m_x(i);
            }
        }
    };

    // Variant of MV_Scal_Generic for single vectors (1-D Views) r and x.
    // As above, av is either a 1-D View (and only its first entry will be
    // read), or a scalar.
    template<class execution_space, class RV, class AV, class XV, class SizeType>
    void V_Scal_Generic(const execution_space &space, const RV &r, const AV &av,
                        const XV &x, const SizeType startingColumn, int a = 2) {
        static_assert(flare::is_view<RV>::value,
                      "V_Scal_Generic: RV is not a flare::View.");
        static_assert(flare::is_view<XV>::value,
                      "V_Scal_Generic: XV is not a flare::View.");
        static_assert(RV::rank == 1, "V_Scal_Generic: RV is not rank 1.");
        static_assert(XV::rank == 1, "V_Scal_Generic: XV is not rank 1.");

        const SizeType numRows = x.extent(0);
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        if (a == 0) {
            V_Scal_Functor<RV, AV, XV, 0, SizeType> op(r, x, av, startingColumn);
            flare::parallel_for("flare::blas::Scal::S0", policy, op);
            return;
        }
        if (a == -1) {
            V_Scal_Functor<RV, AV, XV, -1, SizeType> op(r, x, av, startingColumn);
            flare::parallel_for("flare::blas::Scal::S1", policy, op);
            return;
        }
        if (a == 1) {
            V_Scal_Functor<RV, AV, XV, 1, SizeType> op(r, x, av, startingColumn);
            flare::parallel_for("flare::blas::Scal::S2", policy, op);
            return;
        }

        // a arbitrary (not -1, 0, or 1)
        V_Scal_Functor<RV, AV, XV, 2, SizeType> op(r, x, av, startingColumn);
        flare::parallel_for("flare::blas::Scal::S3", policy, op);
    }

    /// mv

    // Functor for multivectors R and X and 1-D View a, that computes any
    // of the following:
    //
    // 1. R(i,j) = alpha*X(i,j) for alpha in -1,0,1
    // 2. R(i,j) = a(j)*X(i,j)
    //
    // The template parameter scalar_x corresponds to alpha in the
    // operation y = alpha*x.  The values -1, 0, and -1 correspond to
    // literal values of this coefficient.  The value 2 tells the functor
    // to use the corresponding vector of coefficients.  Any literal
    // coefficient of zero has BLAS semantics of ignoring the
    // corresponding (multi)vector entry.  This does not apply to
    // coefficients in the a vector, if they are used.
    template<class RMV, class aVector, class XMV, int scalar_x,
            class SizeType = typename RMV::size_type>
    struct MV_Scal_Functor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename RMV::non_const_value_type> ATS;

        const size_type numCols;
        RMV R_;
        XMV X_;
        aVector a_;

        MV_Scal_Functor(const RMV &R, const XMV &X, const aVector &a,
                        const SizeType startingColumn)
                : numCols(X.extent(1)), R_(R), X_(X), a_(a) {
            if (startingColumn != 0) {
                auto rng =
                        std::make_pair(startingColumn, static_cast<SizeType>(a.extent(0)));
                a_ = flare::subview(a, rng);
            }
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i) const {
            // scalar_x is a compile-time constant (since it is a template
            // parameter), so the compiler should evaluate these branches at
            // compile time.
            if (scalar_x == 0) {
                for (size_type k = 0; k < numCols; ++k) {
                    R_(i, k) = ATS::zero();
                }
            }
            if (scalar_x == -1) {
                for (size_type k = 0; k < numCols; ++k) {
                    R_(i, k) = -X_(i, k);
                }
            }
            if (scalar_x == 1) {
                for (size_type k = 0; k < numCols; ++k) {
                    R_(i, k) = X_(i, k);
                }
            }
            if (scalar_x == 2) {
                for (size_type k = 0; k < numCols; ++k) {
                    R_(i, k) = a_(k) * X_(i, k);
                }
            }
        }
    };

    // Variant of MV_Scal_Functor, where a is a scalar.
    // This functor computes any of the following:
    //
    // 1. R(i,j) = alpha*X(i,j) for alpha,beta in -1,0,1
    // 2. R(i,j) = a*X(i,j)
    //
    // This version works by partial specialization on aVector.
    // In this partial specialization, aVector is a scalar.
    template<class RMV, class XMV, int scalar_x, class SizeType>
    struct MV_Scal_Functor<RMV, typename XMV::non_const_value_type, XMV, scalar_x,
            SizeType> {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename RMV::non_const_value_type> ATS;

        const size_type numCols;
        RMV m_r;
        XMV m_x;
        const typename XMV::non_const_value_type m_a;

        MV_Scal_Functor(const RMV &r, const XMV &x,
                        const typename XMV::non_const_value_type &a,
                        const SizeType /* startingColumn */)
                : numCols(x.extent(1)), m_r(r), m_x(x), m_a(a) {}

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i) const {
            // scalar_x and scalar_y are compile-time constants (since they
            // are template parameters), so the compiler should evaluate these
            // branches at compile time.
            if (scalar_x == 0) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_r(i, k) = ATS::zero();
                }
            }
            if (scalar_x == -1) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_r(i, k) = -m_x(i, k);
                }
            }
            if (scalar_x == 1) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_r(i, k) = m_x(i, k);
                }
            }
            if (scalar_x == 2) {
                for (size_type k = 0; k < numCols; ++k) {
                    m_r(i, k) = m_a * m_x(i, k);
                }
            }
        }
    };

    // Column-unrolled variant of MV_Scal_Functor.  The number of columns
    // in X and Y, UNROLL, is a compile-time constant.
    template<class RMV, class aVector, class XMV, int scalar_x, int UNROLL,
            class SizeType>
    struct MV_Scal_Unroll_Functor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename RMV::non_const_value_type> ATS;

        RMV m_r;
        XMV m_x;
        aVector m_a;

        MV_Scal_Unroll_Functor(const RMV &r, const XMV &x, const aVector &a,
                               const SizeType startingColumn)
                : m_r(r), m_x(x), m_a(a) {
            if (startingColumn != 0) {
                auto rng =
                        std::make_pair(startingColumn, static_cast<SizeType>(a.extent(0)));
                m_a = flare::subview(a, rng);
            }
        }

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i) const {
            if (scalar_x == 0) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                for (int k = 0; k < UNROLL; ++k) {
                    m_r(i, k) = ATS::zero();
                }
            }
            if (scalar_x == -1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                for (int k = 0; k < UNROLL; ++k) {
                    m_r(i, k) = -m_x(i, k);
                }
            }
            if (scalar_x == 1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                for (int k = 0; k < UNROLL; ++k) {
                    m_r(i, k) = m_x(i, k);
                }
            }
            if (scalar_x == 2) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                for (int k = 0; k < UNROLL; ++k) {
                    m_r(i, k) = m_a(k) * m_x(i, k);
                }
            }
        }
    };

// Variant of MV_Scal_Unroll_Functor for a single coefficient (rather
// than a vector of coefficients) a.  The number of columns in X,
// UNROLL, is a compile-time constant.
    template<class RMV, class XMV, int scalar_x, int UNROLL, class SizeType>
    struct MV_Scal_Unroll_Functor<RMV, typename XMV::non_const_value_type, XMV,
            scalar_x, UNROLL, SizeType> {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename RMV::non_const_value_type> ATS;

        RMV m_r;
        XMV m_x;
        const typename XMV::non_const_value_type m_a;

        MV_Scal_Unroll_Functor(const RMV &r, const XMV &x,
                               const typename XMV::non_const_value_type &a,
                               const SizeType /* startingColumn */)
                : m_r(r), m_x(x), m_a(a) {}

        FLARE_INLINE_FUNCTION
        void operator()(const size_type &i) const {
            if (scalar_x == 0) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                for (int k = 0; k < UNROLL; ++k) {
                    m_r(i, k) = ATS::zero();
                }
            }
            if (scalar_x == -1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                for (int k = 0; k < UNROLL; ++k) {
                    m_r(i, k) = -m_x(i, k);
                }
            }
            if (scalar_x == 1) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                for (int k = 0; k < UNROLL; ++k) {
                    m_r(i, k) = m_x(i, k);
                }
            }
            if (scalar_x == 2) {
#ifdef FLARE_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                for (int k = 0; k < UNROLL; ++k) {
                    m_r(i, k) = m_a * m_x(i, k);
                }
            }
        }
    };

    // Invoke the unrolled multivector functor that computes any of the
    // following:
    //
    // 1. R(i,j) = a*X(i,j) for a in -1,0,1
    // 2. R(i,j) = av(j)*X(i,j)
    //
    // a comes in as an int.  The values -1, 0, and 1 correspond to the
    // literal values of this coefficient.  The value 2 tells the functor
    // to use av, which may be either a 1-D View or a scalar.  Otherwise,
    // av is ignored.
    //
    // Any literal coefficient of zero has BLAS semantics of ignoring the
    // corresponding (multi)vector entry.  This does NOT apply to
    // coefficient(s) in av, if used.
    template<class execution_space, class RMV, class aVector, class XMV,
            int UNROLL, class SizeType>
    void MV_Scal_Unrolled(const execution_space &space, const RMV &r,
                          const aVector &av, const XMV &x,
                          const SizeType startingColumn, int a = 2) {
        if (a == 0) {
            MV_Scal_Unroll_Functor<RMV, aVector, XMV, 0, UNROLL, SizeType> op(
                    r, x, av, startingColumn);
            const SizeType numRows = x.extent(0);
            flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);
            flare::parallel_for("flare::blas::Scal::MV::S0", policy, op);
            return;
        }
        if (a == -1) {
            MV_Scal_Unroll_Functor<RMV, aVector, XMV, -1, UNROLL, SizeType> op(
                    r, x, av, startingColumn);
            const SizeType numRows = x.extent(0);
            flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);
            flare::parallel_for("flare::blas::Scal::MV::S1", policy, op);
            return;
        }
        if (a == 1) {
            MV_Scal_Unroll_Functor<RMV, aVector, XMV, 1, UNROLL, SizeType> op(
                    r, x, av, startingColumn);
            const SizeType numRows = x.extent(0);
            flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);
            flare::parallel_for("flare::blas::Scal::MV::S2", policy, op);
            return;
        }

        // a arbitrary (not -1, 0, or 1)
        MV_Scal_Unroll_Functor<RMV, aVector, XMV, 2, UNROLL, SizeType> op(
                r, x, av, startingColumn);
        const SizeType numRows = x.extent(0);
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);
        flare::parallel_for("flare::blas::Scal::MV::S3", policy, op);
    }

    // Invoke the "generic" (not unrolled) multivector functor that
    // computes any of the following:
    //
    // 1. R(i,j) = a*X(i,j) for a in -1,0,1
    // 2. R(i,j) = av(j)*X(i,j)
    //
    // a comes in as an int.  The values -1, 0, and 1 correspond to the
    // literal values of this coefficient.  The value 2 tells the functor
    // to use av, which may be either a 1-D View or a scalar.  Otherwise,
    // av is ignored.
    //
    // Any literal coefficient of zero has BLAS semantics of ignoring the
    // corresponding (multi)vector entry.  This does NOT apply to
    // coefficient(s) in av, if used.
    template<class execution_space, class RVector, class aVector, class XVector,
            class SizeType>
    void MV_Scal_Generic(const execution_space &space, const RVector &r,
                         const aVector &av, const XVector &x,
                         const SizeType startingColumn, int a = 2) {
        const SizeType numRows = x.extent(0);
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        if (a == 0) {
            MV_Scal_Functor<RVector, aVector, XVector, 0, SizeType> op(r, x, av,
                                                                       startingColumn);
            flare::parallel_for("flare::blas::Scal::MV::S4", policy, op);
            return;
        }
        if (a == -1) {
            MV_Scal_Functor<RVector, aVector, XVector, -1, SizeType> op(r, x, av,
                                                                        startingColumn);
            flare::parallel_for("flare::blas::Scal::MV::S5", policy, op);
            return;
        }
        if (a == 1) {
            MV_Scal_Functor<RVector, aVector, XVector, 1, SizeType> op(r, x, av,
                                                                       startingColumn);
            flare::parallel_for("flare::blas::Scal::MV::S6", policy, op);
            return;
        }

        // a arbitrary (not -1, 0, or 1)
        MV_Scal_Functor<RVector, aVector, XVector, 2, SizeType> op(r, x, av,
                                                                   startingColumn);
        flare::parallel_for("flare::blas::Scal::MV::S7", policy, op);
    }

    // Compute any of the following, in a way optimized for X, Y, and R
    // being LayoutLeft:
    //
    // 1. R(i,j) = a*X(i,j) for a in -1,0,1
    // 2. R(i,j) = av(j)*X(i,j)
    //
    // a comes in as an int.  The values -1, 0, and 1 correspond to the
    // literal values of this coefficient.  The value 2 tells the functor
    // to use av, which may be either a 1-D View or a scalar.  Otherwise,
    // av is ignored.
    //
    // Any literal coefficient of zero has BLAS semantics of ignoring the
    // corresponding (multi)vector entry.  This does NOT apply to
    // coefficient(s) in av, if used.
    template<class execution_space, class RMV, class AV, class XMV, class SizeType>
    void MV_Scal_Invoke_Left(const execution_space &space, const RMV &r,
                             const AV &av, const XMV &x, int a = 2) {
        const SizeType numCols = x.extent(1);

#if FLARE_BLAS_OPTIMIZATION_LEVEL_SCAL <= 2

        // Strip-mine by 8, then 4.  After that, do one column at a time.
        // We limit the number of strip-mine values in order to keep down
        // the amount of code to compile.

        SizeType j = 0;  // the current column of X and Y
        for (; j + 8 <= numCols; j += 8) {
            const std::pair<SizeType, SizeType> rng(j, j + 8);
            auto X_cur = flare::subview(x, flare::ALL(), rng);
            auto R_cur = flare::subview(r, flare::ALL(), rng);
            typedef decltype(X_cur) XMV2D;
            typedef decltype(R_cur) RMV2D;

            MV_Scal_Unrolled<execution_space, RMV2D, AV, XMV2D, 8, SizeType>(
                    space, R_cur, av, X_cur, j, a);
        }
        for (; j + 4 <= numCols; j += 4) {
            const std::pair<SizeType, SizeType> rng(j, j + 4);
            auto X_cur = flare::subview(x, flare::ALL(), rng);
            auto R_cur = flare::subview(r, flare::ALL(), rng);
            typedef decltype(X_cur) XMV2D;
            typedef decltype(R_cur) RMV2D;

            MV_Scal_Unrolled<execution_space, RMV2D, AV, XMV2D, 4, SizeType>(
                    space, R_cur, av, X_cur, j, a);
        }
        for (; j < numCols; ++j) {
            // RMV and XMV need to turn 1-D.
            auto x_cur = flare::subview(x, flare::ALL(), j);
            auto r_cur = flare::subview(r, flare::ALL(), j);
            typedef decltype(r_cur) RV;
            typedef decltype(x_cur) XV;

            V_Scal_Generic<execution_space, RV, AV, XV, SizeType>(space, r_cur, av,
                                                                  x_cur, j, a);
        }

#else  // FLARE_BLAS_OPTIMIZATION_LEVEL_SCAL > 2

        switch (numCols) {
    case 1: {
      auto r_0 = flare::subview(r, flare::ALL(), 0);
      auto x_0 = flare::subview(x, flare::ALL(), 0);
      typedef decltype(r_0) RV;
      typedef decltype(x_0) XV;

      V_Scal_Generic<execution_space, RV, AV, XV, SizeType>(space, r_0, av, x_0,
                                                            0, a);
      break;
    }
    case 2:
      MV_Scal_Unrolled<execution_space, RMV, AV, XMV, 2, SizeType>(space, r, av,
                                                                   x, 0, a);
      break;
    case 3:
      MV_Scal_Unrolled<execution_space, RMV, AV, XMV, 3, SizeType>(space, r, av,
                                                                   x, 0, a);
      break;
    case 4:
      MV_Scal_Unrolled<execution_space, RMV, AV, XMV, 4, SizeType>(space, r, av,
                                                                   x, 0, a);
      break;
    case 5:
      MV_Scal_Unrolled<execution_space, RMV, AV, XMV, 5, SizeType>(space, r, av,
                                                                   x, 0, a);
      break;
    case 6:
      MV_Scal_Unrolled<execution_space, RMV, AV, XMV, 6, SizeType>(space, r, av,
                                                                   x, 0, a);
      break;
    case 7:
      MV_Scal_Unrolled<execution_space, RMV, AV, XMV, 7, SizeType>(space, r, av,
                                                                   x, 0, a);
      break;
    case 8:
      MV_Scal_Unrolled<execution_space, RMV, AV, XMV, 8, SizeType>(space, r, av,
                                                                   x, 0, a);
      break;
    case 9:
      MV_Scal_Unrolled<execution_space, RMV, AV, XMV, 9, SizeType>(space, r, av,
                                                                   x, 0, a);
      break;
    case 10:
      MV_Scal_Unrolled<execution_space, RMV, AV, XMV, 10, SizeType>(
          space, r, av, x, 0, a);
      break;
    case 11:
      MV_Scal_Unrolled<execution_space, RMV, AV, XMV, 11, SizeType>(
          space, r, av, x, 0, a);
      break;
    case 12:
      MV_Scal_Unrolled<execution_space, RMV, AV, XMV, 12, SizeType>(
          space, r, av, x, 0, a);
      break;
    case 13:
      MV_Scal_Unrolled<execution_space, RMV, AV, XMV, 13, SizeType>(
          space, r, av, x, 0, a);
      break;
    case 14:
      MV_Scal_Unrolled<execution_space, RMV, AV, XMV, 14, SizeType>(
          space, r, av, x, 0, a);
      break;
    case 15:
      MV_Scal_Unrolled<execution_space, RMV, AV, XMV, 15, SizeType>(
          space, r, av, x, 0, a);
      break;
    case 16:
      MV_Scal_Unrolled<execution_space, RMV, AV, XMV, 16, SizeType>(
          space, r, av, x, 0, a);
      break;
    default:
      MV_Scal_Generic<execution_space, RMV, AV, XMV, SizeType>(space, r, av, x,
                                                               0, a);
  }

#endif  // FLARE_BLAS_OPTIMIZATION_LEVEL_SCAL
    }

    // Compute any of the following, in a way optimized for X, Y, and R
    // being LayoutRight:
    //
    // 1. R(i,j) = a*X(i,j) for a in -1,0,1
    // 2. R(i,j) = av(j)*X(i,j)
    //
    // a comes in as an int.  The values -1, 0, and 1 correspond to the
    // literal values of this coefficient.  The value 2 tells the functor
    // to use av, which may be either a 1-D View or a scalar.  Otherwise,
    // av is ignored.
    //
    // Any literal coefficient of zero has BLAS semantics of ignoring the
    // corresponding (multi)vector entry.  This does NOT apply to
    // coefficient(s) in av, if used.
    template<class execution_space, class RMV, class aVector, class XMV,
            class SizeType>
    void MV_Scal_Invoke_Right(const execution_space &space, const RMV &r,
                              const aVector &av, const XMV &x, int a = 2) {
        const SizeType numCols = x.extent(1);

        if (numCols == 1) {
            typedef flare::View<typename RMV::value_type *, typename RMV::array_layout,
                    typename RMV::device_type, typename RMV::memory_traits>
                    RV;
            typedef flare::View<typename XMV::value_type *, typename XMV::array_layout,
                    typename XMV::device_type, typename XMV::memory_traits>
                    XV;

            RV r_0 = flare::subview(r, flare::ALL(), 0);
            XV x_0 = flare::subview(x, flare::ALL(), 0);
            V_Scal_Generic<execution_space, RMV, aVector, XMV, 1, SizeType>(space, r_0,
                                                                            av, x_0, a);
        } else {
            MV_Scal_Generic<execution_space, RMV, aVector, XMV, SizeType>(space, r, av,
                                                                          x, a);
        }
    }

    /// spec

    // Unification layer
    template<class execution_space, class RV, class AV, class XV,
            int XV_Rank = XV::rank>
    struct Scal {
        static void scal(const execution_space &space, const RV &R, const AV &A,
                         const XV &X);
    };

    template<class execution_space, class RV, class XV>
    struct Scal<execution_space, RV, typename XV::non_const_value_type, XV, 1> {
        typedef typename XV::non_const_value_type AV;
        typedef typename XV::size_type size_type;
        typedef flare::ArithTraits<typename XV::non_const_value_type> ATA;

        static void scal(const execution_space &space, const RV &R, const AV &alpha,
                         const XV &X) {
            static_assert(flare::is_view<RV>::value,
                          "flare::blas::Impl::"
                          "Scal<1-D>: RV is not a flare::View.");
            static_assert(flare::is_view<XV>::value,
                          "flare::blas::Impl::"
                          "Scal<1-D>: XV is not a flare::View.");
            static_assert(RV::rank == 1,
                          "flare::blas::Impl::Scal<1-D>: "
                          "RV is not rank 1.");
            static_assert(XV::rank == 1,
                          "flare::blas::Impl::Scal<1-D>: "
                          "XV is not rank 1.");
            flare::Profiling::pushRegion("flare::blas::scal");

            const size_type numRows = X.extent(0);
            int a = 2;
            if (alpha == ATA::zero()) {
                a = 0;
            } else if (alpha == -ATA::one()) {
                a = -1;
            } else if (alpha == ATA::one()) {
                a = 1;
            }

            if (numRows < static_cast<size_type>(INT_MAX)) {
                typedef int index_type;
                V_Scal_Generic<execution_space, RV, AV, XV, index_type>(space, R, alpha,
                                                                        X, a);
            } else {
                typedef typename XV::size_type index_type;
                V_Scal_Generic<execution_space, RV, AV, XV, index_type>(space, R, alpha,
                                                                        X, a);
            }
            flare::Profiling::popRegion();
        }
    };

    /// \brief Partial specialization of Scal for 2-D Views and 1-D View AV.
    ///
    /// Compute any of the following:
    ///
    /// 1. R(i,j) = a*X(i,j) for a in -1,0,1
    /// 2. R(i,j) = alpha(j)*X(i,j)
    template<class execution_space, class RMV, class AV, class XMV>
    struct Scal<execution_space, RMV, AV, XMV, 2> {
        typedef typename XMV::size_type size_type;
        typedef flare::ArithTraits<typename XMV::non_const_value_type> ATA;

        static void scal(const execution_space &space, const RMV &R, const AV &av,
                         const XMV &X) {
            static_assert(flare::is_view<RMV>::value,
                          "flare::blas::Impl::"
                          "Scal<2-D>: RMV is not a flare::View.");
            static_assert(flare::is_view<AV>::value,
                          "flare::blas::Impl::"
                          "Scal<2-D>: AV is not a flare::View.");
            static_assert(flare::is_view<XMV>::value,
                          "flare::blas::Impl::"
                          "Scal<2-D>: XMV is not a flare::View.");
            static_assert(RMV::rank == 2,
                          "flare::blas::Impl::Scal<2-D>: "
                          "RMV is not rank 2.");
            static_assert(AV::rank == 1,
                          "flare::blas::Impl::Scal<2-D>: "
                          "AV is not rank 1.");
            static_assert(XMV::rank == 2,
                          "flare::blas::Impl::Scal<2-D>: "
                          "XMV is not rank 2.");
            flare::Profiling::pushRegion("flare::blas::scal");

            const size_type numRows = X.extent(0);
            const size_type numCols = X.extent(1);
            const int a = (av.extent(0) == 0) ? 0 : 2;
            if (numRows < static_cast<size_type>(INT_MAX) &&
                numRows * numCols < static_cast<size_type>(INT_MAX)) {
                typedef int index_type;
                MV_Scal_Invoke_Left<execution_space, RMV, AV, XMV, index_type>(space, R,
                                                                               av, X, a);
            } else {
                typedef typename XMV::size_type index_type;
                MV_Scal_Invoke_Left<execution_space, RMV, AV, XMV, index_type>(space, R,
                                                                               av, X, a);
            }
            flare::Profiling::popRegion();
        }
    };

    /// \brief Partial specialization of Scal for 2-D Views and scalar AV.
    ///
    /// Compute any of the following:
    ///
    /// 1. R(i,j) = a*X(i,j) for a in -1,0,1
    /// 2. R(i,j) = alpha*X(i,j)
    template<class execution_space, class RMV, class XMV>
    struct Scal<execution_space, RMV, typename XMV::non_const_value_type, XMV, 2> {
        typedef typename XMV::non_const_value_type AV;
        typedef typename XMV::size_type size_type;
        typedef flare::ArithTraits<typename XMV::non_const_value_type> ATA;

        static void scal(const execution_space &space, const RMV &R, const AV &alpha,
                         const XMV &X) {
            static_assert(flare::is_view<RMV>::value,
                          "flare::blas::Impl::"
                          "Scal<2-D, AV=scalar>: RMV is not a flare::View.");
            static_assert(flare::is_view<XMV>::value,
                          "flare::blas::Impl::"
                          "Scal<2-D, AV=scalar>: XMV is not a flare::View.");
            static_assert(RMV::rank == 2,
                          "flare::blas::Impl::Scal<2-D, AV=scalar>: "
                          "RMV is not rank 2.");
            static_assert(XMV::rank == 2,
                          "flare::blas::Impl::Scal<2-D, AV=scalar>: "
                          "XMV is not rank 2.");
            flare::Profiling::pushRegion("flare::blas::scal");


            const size_type numRows = X.extent(0);
            const size_type numCols = X.extent(1);
            int a = 2;
            if (alpha == ATA::zero()) {
                a = 0;
            } else if (alpha == -ATA::one()) {
                a = -1;
            } else if (alpha == ATA::one()) {
                a = 1;
            }

            if (numRows < static_cast<size_type>(INT_MAX) &&
                numRows * numCols < static_cast<size_type>(INT_MAX)) {
                typedef int index_type;
                MV_Scal_Invoke_Left<execution_space, RMV,
                        typename XMV::non_const_value_type, XMV, index_type>(
                        space, R, alpha, X, a);
            } else {
                typedef typename XMV::size_type index_type;
                MV_Scal_Invoke_Left<execution_space, RMV,
                        typename XMV::non_const_value_type, XMV, index_type>(
                        space, R, alpha, X, a);
            }
            flare::Profiling::popRegion();
        }
    };

    /// serial
    ///
    /// Serial Internal Impl
    /// ====================
    struct SerialScaleInternal {
        template<typename ScalarType, typename ValueType>
        FLARE_INLINE_FUNCTION static int invoke(const int m, const ScalarType alpha,
                /* */ ValueType *FLARE_RESTRICT A,
                                                const int as0) {
#if defined(FLARE_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
            for (int i = 0; i < m; ++i) A[i * as0] *= alpha;

            return 0;
        }

        template<typename ScalarType, typename ValueType>
        FLARE_INLINE_FUNCTION static int invoke(const int m, const int n,
                                                const ScalarType alpha,
                /* */ ValueType *FLARE_RESTRICT A,
                                                const int as0, const int as1) {
            if (as0 > as1)
                for (int i = 0; i < m; ++i) invoke(n, alpha, A + i * as0, as1);
            else
                for (int j = 0; j < n; ++j) invoke(m, alpha, A + j * as1, as0);

            return 0;
        }
    };

    /// team scal

///
/// Team Internal Impl
/// ====================
    struct TeamScaleInternal {
        template<typename MemberType, typename ScalarType, typename ValueType>
        FLARE_INLINE_FUNCTION static int invoke(const MemberType &member,
                                                const int m, const ScalarType alpha,
                /* */ ValueType *FLARE_RESTRICT A,
                                                const int as0) {
            flare::parallel_for(flare::TeamThreadRange(member, m),
                                [&](const int &i) { A[i * as0] *= alpha; });
            // member.team_barrier();
            return 0;
        }

        template<typename MemberType, typename ScalarType, typename ValueType>
        FLARE_INLINE_FUNCTION static int invoke(const MemberType &member,
                                                const int m, const int n,
                                                const ScalarType alpha,
                /* */ ValueType *FLARE_RESTRICT A,
                                                const int as0, const int as1) {
            if (m > n) {
                flare::parallel_for(
                        flare::TeamThreadRange(member, m), [&](const int &i) {
                            SerialScaleInternal::invoke(n, alpha, A + i * as0, as1);
                        });
            } else {
                flare::parallel_for(
                        flare::TeamThreadRange(member, n), [&](const int &j) {
                            SerialScaleInternal::invoke(m, alpha, A + j * as1, as0);
                        });
            }
            // member.team_barrier();
            return 0;
        }
    };

    ///
    /// TeamVector Internal Impl
    /// ========================
    struct TeamVectorScaleInternal {
        template<typename MemberType, typename ScalarType, typename ValueType>
        FLARE_INLINE_FUNCTION static int invoke(const MemberType &member,
                                                const int m, const ScalarType alpha,
                /* */ ValueType *FLARE_RESTRICT A,
                                                const int as0) {
            flare::parallel_for(flare::TeamVectorRange(member, m),
                                [&](const int &i) { A[i * as0] *= alpha; });
            // member.team_barrier();
            return 0;
        }

        template<typename MemberType, typename ScalarType, typename ValueType>
        FLARE_INLINE_FUNCTION static int invoke(const MemberType &member,
                                                const int m, const int n,
                                                const ScalarType alpha,
                /* */ ValueType *FLARE_RESTRICT A,
                                                const int as0, const int as1) {
            if (as0 > as1) {
                flare::parallel_for(
                        flare::TeamThreadRange(member, m), [&](const int &i) {
                            flare::parallel_for(
                                    flare::ThreadVectorRange(member, n),
                                    [&](const int &j) { A[i * as0 + j * as1] *= alpha; });
                        });
            } else {
                flare::parallel_for(
                        flare::ThreadVectorRange(member, m), [&](const int &i) {
                            flare::parallel_for(
                                    flare::TeamThreadRange(member, n),
                                    [&](const int &j) { A[i * as0 + j * as1] *= alpha; });
                        });
            }
            // member.team_barrier();
            return 0;
        }
    };


    template<class RV, class XV>
    struct team_scal_tpl_spec_avail {
        constexpr static bool value = false;
    };

    // Unification and Specialization layer
    template<class TeamType, class RV, class XV,
            bool tpl_spec_avail = team_scal_tpl_spec_avail<RV, XV>::value>
    struct TeamScal {
        static FLARE_INLINE_FUNCTION void team_scal(
                const TeamType &team, const RV &R,
                const typename XV::non_const_value_type &a, const XV &X);
    };

    template<class TeamType, class RV, class XV>
    struct TeamScal<TeamType, RV, XV, false> {
        static FLARE_INLINE_FUNCTION void team_scal(
                const TeamType &team, const RV &R,
                const typename XV::non_const_value_type &a, const XV &X) {
            const int N = X.extent(0);
            flare::parallel_for(flare::TeamThreadRange(team, N),
                                [&](const int &i) { R(i) = a * X(i); });
        }
    };

}  // namespace flare::blas::detail


#define FLARE_BLAS_SCAL_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  template struct Scal<                                                       \
      EXEC_SPACE,                                                             \
      flare::View<SCALAR*, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>,    \
                   flare::MemoryTraits<flare::Unmanaged> >,                 \
      SCALAR,                                                                 \
      flare::View<const SCALAR*, LAYOUT,                                     \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                     \
                   flare::MemoryTraits<flare::Unmanaged> >,                 \
      1>;

#define FLARE_BLAS_SCAL_MV_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, \
                                          MEM_SPACE)                        \
  template struct Scal<                                                     \
      EXEC_SPACE,                                                           \
      flare::View<SCALAR**, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>, \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      flare::View<const SCALAR*, LAYOUT,                                   \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                   \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      flare::View<const SCALAR**, LAYOUT,                                  \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                   \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      2>;                                                                 \
  template struct Scal<                                                     \
      EXEC_SPACE,                                                           \
      flare::View<SCALAR**, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>, \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      SCALAR,                                                               \
      flare::View<const SCALAR**, LAYOUT,                                  \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                   \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      2>;
#endif  // FLARE_KERNEL_BLAS_SCAL_IMPL_H_
