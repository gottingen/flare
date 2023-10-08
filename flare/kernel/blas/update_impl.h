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

#ifndef FLARE_KERNEL_BLAS_UPDATE_IMPL_H_
#define FLARE_KERNEL_BLAS_UPDATE_IMPL_H_

#include <flare/core.h>
#include <flare/kernel/common/inner_product_space_traits.h>
#include <flare/kernel/blas/utility.h>
#include <flare/kernel/common/arith_traits.h>
#include <flare/kernel/common/helper.h>


namespace flare::blas::detail {


//
// update
//

// Functor for multivectors X, Y, and Z, that computes
//
// Z(i,j) = alpha*X(i,j) + beta*Y(i,j) + gamma*Z(i,j)
//
// with special cases for alpha, beta, or gamma = 0.
//
// The template parameters scalar_x, scalar_y, and scalar_z correspond
// to alpha, beta, resp. gammar in the operation Z = alpha*X + beta*Y
// + gamma*Z.  The value 0 corresponds to literal values of those
// coefficients.  The value 2 tells the functor to use the
// corresponding input coefficient.  Any literal coefficient of zero
// has BLAS semantics of ignoring the corresponding (multi)vector
// entry.
    template <class XMV, class YMV, class ZMV, int scalar_x, int scalar_y,
            int scalar_z, class SizeType = typename ZMV::size_type>
    struct MV_Update_Functor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename ZMV::non_const_value_type> ATS;

        const size_type numCols;
        const typename XMV::non_const_value_type alpha_;
        XMV X_;
        const typename YMV::non_const_value_type beta_;
        YMV Y_;
        const typename ZMV::non_const_value_type gamma_;
        ZMV Z_;

        MV_Update_Functor(const typename XMV::non_const_value_type& alpha,
                          const XMV& X,
                          const typename YMV::non_const_value_type& beta,
                          const YMV& Y,
                          const typename ZMV::non_const_value_type& gamma,
                          const ZMV& Z)
                : numCols(X.extent(1)),
                  alpha_(alpha),
                  X_(X),
                  beta_(beta),
                  Y_(Y),
                  gamma_(gamma),
                  Z_(Z) {
            static_assert(flare::is_view<XMV>::value,
                          "flare::blas::Impl::"
                          "MV_Update_Functor: X is not a flare::View.");
            static_assert(flare::is_view<YMV>::value,
                          "flare::blas::Impl::"
                          "MV_Update_Functor: Y is not a flare::View.");
            static_assert(flare::is_view<ZMV>::value,
                          "flare::blas::Impl::"
                          "MV_Update_Functor: Z is not a flare::View.");
            static_assert(std::is_same<typename ZMV::value_type,
                                  typename ZMV::non_const_value_type>::value,
                          "flare::blas::Impl::MV_Update_Functor: Z is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            // Casting enum values to int avoids compiler warnings about
            // comparing different kinds of enum values.
            static_assert(
                    (int)ZMV::rank == (int)XMV::rank && (int)ZMV::rank == (int)YMV::rank,
                    "flare::blas::Impl::MV_Update_Functor: "
                    "X, Y, and Z must have the same rank.");
            static_assert(ZMV::rank == 2,
                          "flare::blas::Impl::MV_Update_Functor: "
                          "XMV, YMV, and ZMV must have rank 2.");
        }

        FLARE_INLINE_FUNCTION void operator()(const size_type& i) const {
            // scalar_x, scalar_y, and scalar_z are compile-time constants
            // (since they are template parameters), so the compiler should
            // evaluate these branches at compile time.
            if (scalar_x == 0) {
                if (scalar_y == 0) {
                    if (scalar_z == 0) {
                        for (size_type k = 0; k < numCols; ++k) {
                            Z_(i, k) = ATS::zero();
                        }
                    } else {
                        for (size_type k = 0; k < numCols; ++k) {
                            Z_(i, k) = gamma_ * Z_(i, k);
                        }
                    }
                } else {
                    if (scalar_z == 0) {
                        for (size_type k = 0; k < numCols; ++k) {
                            Z_(i, k) = beta_ * Y_(i, k);
                        }
                    } else {
                        for (size_type k = 0; k < numCols; ++k) {
                            Z_(i, k) = beta_ * Y_(i, k) + gamma_ * Z_(i, k);
                        }
                    }
                }
            }
                //
                // scalar_x == 2
                //
            else {
                if (scalar_y == 0) {
                    if (scalar_z == 0) {
                        for (size_type k = 0; k < numCols; ++k) {
                            Z_(i, k) = alpha_ * X_(i, k);
                        }
                    } else {
                        for (size_type k = 0; k < numCols; ++k) {
                            Z_(i, k) = alpha_ * X_(i, k) + gamma_ * Z_(i, k);
                        }
                    }
                } else {
                    if (scalar_z == 0) {
                        for (size_type k = 0; k < numCols; ++k) {
                            Z_(i, k) = alpha_ * X_(i, k) + beta_ * Y_(i, k);
                        }
                    } else {
                        for (size_type k = 0; k < numCols; ++k) {
                            Z_(i, k) = alpha_ * X_(i, k) + beta_ * Y_(i, k) + gamma_ * Z_(i, k);
                        }
                    }
                }
            }
        }
    };

    // Functor for vectors X, Y, and Z, that computes
    //
    // Z(i) = alpha*X(i) + beta*Y(i) + gamma*Z(i)
    //
    // with special cases for alpha, beta, or gamma = 0.
    //
    // The template parameters scalar_x, scalar_y, and scalar_z correspond
    // to alpha, beta, resp. gammar in the operation Z = alpha*X + beta*Y
    // + gamma*Z.  The value 0 corresponds to literal values of those
    // coefficients.  The value 2 tells the functor to use the
    // corresponding input coefficient.  Any literal coefficient of zero
    // has BLAS semantics of ignoring the corresponding vector entry.
    template <class XV, class YV, class ZV, int scalar_x, int scalar_y,
            int scalar_z, class SizeType = typename ZV::size_type>
    struct V_Update_Functor {
        typedef SizeType size_type;
        typedef flare::ArithTraits<typename ZV::non_const_value_type> ATS;

        const size_type numCols;
        const typename XV::non_const_value_type alpha_;
        XV X_;
        const typename YV::non_const_value_type beta_;
        YV Y_;
        const typename ZV::non_const_value_type gamma_;
        ZV Z_;

        V_Update_Functor(const typename XV::non_const_value_type& alpha, const XV& X,
                         const typename YV::non_const_value_type& beta, const YV& Y,
                         const typename ZV::non_const_value_type& gamma, const ZV& Z)
                : numCols(X.extent(1)),
                  alpha_(alpha),
                  X_(X),
                  beta_(beta),
                  Y_(Y),
                  gamma_(gamma),
                  Z_(Z) {
            static_assert(flare::is_view<XV>::value,
                          "flare::blas::Impl::"
                          "V_Update_Functor: X is not a flare::View.");
            static_assert(flare::is_view<YV>::value,
                          "flare::blas::Impl::"
                          "V_Update_Functor: Y is not a flare::View.");
            static_assert(flare::is_view<ZV>::value,
                          "flare::blas::Impl::"
                          "V_Update_Functor: Z is not a flare::View.");
            static_assert(std::is_same<typename ZV::value_type,
                                  typename ZV::non_const_value_type>::value,
                          "flare::blas::Impl::V_Update_Functor: Z is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            // Casting to int avoids compiler warnings about comparing
            // different kinds of enum values.
            static_assert(
                    (int)ZV::rank == (int)XV::rank && (int)ZV::rank == (int)YV::rank,
                    "flare::blas::Impl::V_Update_Functor: "
                    "X, Y, and Z must have the same rank.");
            static_assert(ZV::rank == 1,
                          "flare::blas::Impl::V_Update_Functor: "
                          "XV, YV, and ZV must have rank 1.");
        }

        FLARE_INLINE_FUNCTION void operator()(const size_type& i) const {
            // scalar_x, scalar_y, and scalar_z are compile-time constants
            // (since they are template parameters), so the compiler should
            // evaluate these branches at compile time.
            if (scalar_x == 0) {
                if (scalar_y == 0) {
                    if (scalar_z == 0) {
                        Z_(i) = ATS::zero();
                    } else {
                        Z_(i) = gamma_ * Z_(i);
                    }
                } else {
                    if (scalar_z == 0) {
                        Z_(i) = beta_ * Y_(i);
                    } else {
                        Z_(i) = beta_ * Y_(i) + gamma_ * Z_(i);
                    }
                }
            }
                //
                // scalar_ x == 2
                //
            else {
                if (scalar_y == 0) {
                    if (scalar_z == 0) {
                        Z_(i) = alpha_ * X_(i);
                    } else {
                        Z_(i) = alpha_ * X_(i) + gamma_ * Z_(i);
                    }
                } else {
                    if (scalar_z == 0) {
                        Z_(i) = alpha_ * X_(i) + beta_ * Y_(i);
                    } else {
                        Z_(i) = alpha_ * X_(i) + beta_ * Y_(i) + gamma_ * Z_(i);
                    }
                }
            }
        }
    };

// Invoke the "generic" (not unrolled) multivector functor that
// computes
//
// Z(i,j) = alpha*X(i,j) + beta*Y(i,j) + gamma*Z(i,j)
//
// with special cases for alpha, beta, or gamma = 0.
//
// a, b, and c come in as integers.  The value 0 corresponds to the
// literal values of the coefficients.  The value 2 tells the functor
// to use the corresponding coefficients: a == 2 means use alpha, b ==
// 2 means use beta, and c == 2 means use gamma.  Otherwise, the
// corresponding coefficients are ignored.
//
// Any literal coefficient of zero has BLAS semantics of ignoring the
// corresponding multivector entry.
    template <class execution_space, class XMV, class YMV, class ZMV,
            class SizeType>
    void MV_Update_Generic(const execution_space& space,
                           const typename XMV::non_const_value_type& alpha,
                           const XMV& X,
                           const typename YMV::non_const_value_type& beta,
                           const YMV& Y,
                           const typename ZMV::non_const_value_type& gamma,
                           const ZMV& Z, int a = 2, int b = 2, int c = 2) {
        static_assert(flare::is_view<XMV>::value,
                      "flare::blas::Impl::"
                      "MV_Update_Generic: X is not a flare::View.");
        static_assert(flare::is_view<YMV>::value,
                      "flare::blas::Impl::"
                      "MV_Update_Generic: Y is not a flare::View.");
        static_assert(flare::is_view<ZMV>::value,
                      "flare::blas::Impl::"
                      "MV_Update_Generic: Z is not a flare::View.");
        static_assert(std::is_same<typename ZMV::value_type,
                              typename ZMV::non_const_value_type>::value,
                      "flare::blas::Impl::MV_Update_Generic: Z is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");
        // Casting to int avoids compiler warnings about comparing different
        // kinds of enum values.
        static_assert(
                (int)ZMV::rank == (int)XMV::rank && (int)ZMV::rank == (int)YMV::rank,
                "flare::blas::Impl::MV_Update_Generic: "
                "X, Y, and Z must have the same rank.");
        static_assert(ZMV::rank == 2,
                      "flare::blas::Impl::MV_Update_Generic: "
                      "XMV, YMV, and ZMV must have rank 2.");

        const SizeType numRows = X.extent(0);
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        if (a == 0) {
            if (b == 0) {
                if (c == 0) {
                    MV_Update_Functor<XMV, YMV, ZMV, 0, 0, 0, SizeType> op(alpha, X, beta,
                                                                           Y, gamma, Z);
                    flare::parallel_for("flare::blas::update<MV,0,0,0>", policy, op);
                } else {
                    MV_Update_Functor<XMV, YMV, ZMV, 0, 0, 2, SizeType> op(alpha, X, beta,
                                                                           Y, gamma, Z);
                    flare::parallel_for("flare::blas::update<MV,0,0,c>", policy, op);
                }
            } else {
                if (c == 0) {
                    MV_Update_Functor<XMV, YMV, ZMV, 0, 2, 0, SizeType> op(alpha, X, beta,
                                                                           Y, gamma, Z);
                    flare::parallel_for("flare::blas::update<MV,0,b,0>", policy, op);
                } else {
                    MV_Update_Functor<XMV, YMV, ZMV, 0, 2, 2, SizeType> op(alpha, X, beta,
                                                                           Y, gamma, Z);
                    flare::parallel_for("flare::blas::update<MV,0,b,c>", policy, op);
                }
            }
        }
            //
            // a == 2
            //
        else {
            if (b == 0) {
                if (c == 0) {
                    MV_Update_Functor<XMV, YMV, ZMV, 2, 0, 0, SizeType> op(alpha, X, beta,
                                                                           Y, gamma, Z);
                    flare::parallel_for("flare::blas::update<MV,a,0,0>", policy, op);
                } else {
                    MV_Update_Functor<XMV, YMV, ZMV, 2, 0, 2, SizeType> op(alpha, X, beta,
                                                                           Y, gamma, Z);
                    flare::parallel_for("flare::blas::update<MV,a,0,c>", policy, op);
                }
            } else {
                if (c == 0) {
                    MV_Update_Functor<XMV, YMV, ZMV, 2, 2, 0, SizeType> op(alpha, X, beta,
                                                                           Y, gamma, Z);
                    flare::parallel_for("flare::blas::update<MV,a,b,0>", policy, op);
                } else {
                    MV_Update_Functor<XMV, YMV, ZMV, 2, 2, 2, SizeType> op(alpha, X, beta,
                                                                           Y, gamma, Z);
                    flare::parallel_for("flare::blas::update<MV,a,b,c>", policy, op);
                }
            }
        }
    }

// Invoke the "generic" (not unrolled) single-vector functor that
// computes
//
// Z(i) = alpha*X(i) + beta*Y(i) + gamma*Z(i)
//
// with special cases for alpha, beta, or gamma = 0.
//
// a, b, and c come in as integers.  The value 0 corresponds to the
// literal values of the coefficients.  The value 2 tells the functor
// to use the corresponding coefficients: a == 2 means use alpha, b ==
// 2 means use beta, and c == 2 means use gamma.  Otherwise, the
// corresponding coefficients are ignored.
//
// Any literal coefficient of zero has BLAS semantics of ignoring the
// corresponding vector entry.
    template <class execution_space, class XV, class YV, class ZV, class SizeType>
    void V_Update_Generic(const execution_space& space,
                          const typename XV::non_const_value_type& alpha,
                          const XV& X,
                          const typename YV::non_const_value_type& beta,
                          const YV& Y,
                          const typename ZV::non_const_value_type& gamma,
                          const ZV& Z, int a = 2, int b = 2, int c = 2) {
        static_assert(flare::is_view<XV>::value,
                      "flare::blas::Impl::"
                      "V_Update_Generic: X is not a flare::View.");
        static_assert(flare::is_view<YV>::value,
                      "flare::blas::Impl::"
                      "V_Update_Generic: Y is not a flare::View.");
        static_assert(flare::is_view<ZV>::value,
                      "flare::blas::Impl::"
                      "V_Update_Generic: Z is not a flare::View.");
        static_assert(std::is_same<typename ZV::value_type,
                              typename ZV::non_const_value_type>::value,
                      "flare::blas::Impl::V_Update_Generic: Z is const.  "
                      "It must be nonconst, because it is an output argument "
                      "(we have to be able to write to its entries).");
        // Casting to int avoids compiler warnings about comparing
        // different kinds of enum values.
        static_assert(
                (int)ZV::rank == (int)XV::rank && (int)ZV::rank == (int)YV::rank,
                "flare::blas::Impl::V_Update_Generic: "
                "X, Y, and Z must have the same rank.");
        static_assert(ZV::rank == 1,
                      "flare::blas::Impl::V_Update_Generic: "
                      "XV, YV, and ZV must have rank 1.");

        const SizeType numRows = X.extent(0);
        flare::RangePolicy<execution_space, SizeType> policy(space, 0, numRows);

        if (a == 0) {
            if (b == 0) {
                if (c == 0) {
                    V_Update_Functor<XV, YV, ZV, 0, 0, 0, SizeType> op(alpha, X, beta, Y,
                                                                       gamma, Z);
                    flare::parallel_for("flare::blas::update<0,0,0>", policy, op);
                } else {
                    V_Update_Functor<XV, YV, ZV, 0, 0, 2, SizeType> op(alpha, X, beta, Y,
                                                                       gamma, Z);
                    flare::parallel_for("flare::blas::update<0,0,c>", policy, op);
                }
            } else {
                if (c == 0) {
                    V_Update_Functor<XV, YV, ZV, 0, 2, 0, SizeType> op(alpha, X, beta, Y,
                                                                       gamma, Z);
                    flare::parallel_for("flare::blas::update<0,b,0>", policy, op);
                } else {
                    V_Update_Functor<XV, YV, ZV, 0, 2, 2, SizeType> op(alpha, X, beta, Y,
                                                                       gamma, Z);
                    flare::parallel_for("flare::blas::update<0,b,c>", policy, op);
                }
            }
        }
            //
            // a == 2
            //
        else {
            if (b == 0) {
                if (c == 0) {
                    V_Update_Functor<XV, YV, ZV, 2, 0, 0, SizeType> op(alpha, X, beta, Y,
                                                                       gamma, Z);
                    flare::parallel_for("flare::blas::update<a,0,0>", policy, op);
                } else {
                    V_Update_Functor<XV, YV, ZV, 2, 0, 2, SizeType> op(alpha, X, beta, Y,
                                                                       gamma, Z);
                    flare::parallel_for("flare::blas::update<a,0,c>", policy, op);
                }
            } else {
                if (c == 0) {
                    V_Update_Functor<XV, YV, ZV, 2, 2, 0, SizeType> op(alpha, X, beta, Y,
                                                                       gamma, Z);
                    flare::parallel_for("flare::blas::update<a,b,0>", policy, op);
                } else {
                    V_Update_Functor<XV, YV, ZV, 2, 2, 2, SizeType> op(alpha, X, beta, Y,
                                                                       gamma, Z);
                    flare::parallel_for("flare::blas::update<a,b,c>", policy, op);
                }
            }
        }
    }


    //
    // update
    //

    /// \brief Implementation of flare::blas::update for single vectors and
    ///   multivectors.
    ///
    /// Compute
    ///
    /// Z(i,j) = alpha*X(i,j) + beta*Y(i,j) + gamma*Z(i,j),
    ///
    /// with special cases for alpha, beta, or gamma = 0.
    template <class execution_space, class XMV, class YMV, class ZMV,
            int rank = ZMV::rank>
    struct Update {
        static void update(const execution_space& space,
                           const typename XMV::non_const_value_type& alpha,
                           const XMV& X,
                           const typename YMV::non_const_value_type& beta,
                           const YMV& Y,
                           const typename ZMV::non_const_value_type& gamma,
                           const ZMV& Z);
    };

    // Partial specialization for XMV, YMV, and ZMV rank-2 Views.
    template <class execution_space, class XMV, class YMV, class ZMV>
    struct Update<execution_space, XMV, YMV, ZMV, 2> {
        typedef typename XMV::size_type size_type;
        typedef flare::ArithTraits<typename XMV::non_const_value_type> ATA;
        typedef flare::ArithTraits<typename YMV::non_const_value_type> ATB;
        typedef flare::ArithTraits<typename ZMV::non_const_value_type> ATC;

        static void update(const execution_space& space,
                           const typename XMV::non_const_value_type& alpha,
                           const XMV& X,
                           const typename YMV::non_const_value_type& beta,
                           const YMV& Y,
                           const typename ZMV::non_const_value_type& gamma,
                           const ZMV& Z) {
            static_assert(flare::is_view<XMV>::value,
                          "flare::blas::Impl::"
                          "Update<rank 2>::update: X is not a flare::View.");
            static_assert(flare::is_view<YMV>::value,
                          "flare::blas::Impl::"
                          "Update<rank 2>::update: Y is not a flare::View.");
            static_assert(flare::is_view<ZMV>::value,
                          "flare::blas::Impl::"
                          "Update<rank 2>::update: Z is not a flare::View.");
            static_assert(std::is_same<typename ZMV::value_type,
                                  typename ZMV::non_const_value_type>::value,
                          "flare::blas::Impl::Update<rank 2>::update: Z is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            // Casting to int avoids compiler warnings about comparing
            // different kinds of enum values.
            static_assert(
                    (int)ZMV::rank == (int)XMV::rank && (int)ZMV::rank == (int)YMV::rank,
                    "flare::blas::Impl::Update<rank 2>::update: "
                    "X, Y, and Z must have the same rank.");
            static_assert(ZMV::rank == 2,
                          "flare::blas::Impl::Update<rank 2>::update: "
                          "XMV, YMV, and ZMV must have rank 2.");
            flare::Profiling::pushRegion("flare::blas::update");

            const size_type numRows = X.extent(0);
            const size_type numCols = X.extent(1);
            int a = 2, b = 2, c = 2;

            if (alpha == ATA::zero()) {
                a = 0;
            } else {
                a = 2;
            }
            if (beta == ATB::zero()) {
                b = 0;
            } else {
                b = 2;
            }
            if (gamma == ATC::zero()) {
                c = 0;
            } else {
                c = 2;
            }

            if (numCols == static_cast<size_type>(1)) {
                // Special case: ZMV has rank 2, but only 1 column.
                // Dispatch to the rank-1 version for better performance.
                auto X_0 = flare::subview(X, flare::ALL(), 0);
                auto Y_0 = flare::subview(Y, flare::ALL(), 0);
                auto Z_0 = flare::subview(Z, flare::ALL(), 0);

                if (numRows * numCols < static_cast<size_type>(INT_MAX)) {
                    typedef int index_type;
                    V_Update_Generic<execution_space, decltype(X_0), decltype(Y_0),
                            decltype(Z_0), index_type>(space, alpha, X_0, beta,
                                                       Y_0, gamma, Z_0, a, b, c);
                } else {
                    typedef typename XMV::size_type index_type;
                    V_Update_Generic<execution_space, decltype(X_0), decltype(Y_0),
                            decltype(Z_0), index_type>(space, alpha, X_0, beta,
                                                       Y_0, gamma, Z_0, a, b, c);
                }
            } else {
                if (numRows * numCols < static_cast<size_type>(INT_MAX)) {
                    typedef int index_type;
                    MV_Update_Generic<execution_space, XMV, YMV, ZMV, index_type>(
                            space, alpha, X, beta, Y, gamma, Z, a, b, c);
                } else {
                    typedef typename XMV::size_type index_type;
                    MV_Update_Generic<execution_space, XMV, YMV, ZMV, index_type>(
                            space, alpha, X, beta, Y, gamma, Z, a, b, c);
                }
            }
            flare::Profiling::popRegion();
        }
    };

    // Partial specialization for XV, YV, and ZV rank-1 Views.
    template <class execution_space, class XV, class YV, class ZV>
    struct Update<execution_space, XV, YV, ZV, 1> {
        typedef typename XV::size_type size_type;
        typedef flare::ArithTraits<typename XV::non_const_value_type> ATA;
        typedef flare::ArithTraits<typename YV::non_const_value_type> ATB;
        typedef flare::ArithTraits<typename ZV::non_const_value_type> ATC;

        static void update(const execution_space& space,
                           const typename XV::non_const_value_type& alpha,
                           const XV& X, const typename YV::non_const_value_type& beta,
                           const YV& Y,
                           const typename ZV::non_const_value_type& gamma,
                           const ZV& Z) {
            // XV, YV, and ZV must be flare::View specializations.
            static_assert(flare::is_view<XV>::value,
                          "flare::blas::Impl::"
                          "Update<rank 1>::update: X is not a flare::View.");
            static_assert(flare::is_view<YV>::value,
                          "flare::blas::Impl::"
                          "Update<rank 1>::update: Y is not a flare::View.");
            static_assert(flare::is_view<ZV>::value,
                          "flare::blas::Impl::"
                          "Update<rank 1>::update: Z is not a flare::View.");
            // ZV must be nonconst (else it can't be an output argument).
            static_assert(std::is_same<typename ZV::value_type,
                                  typename ZV::non_const_value_type>::value,
                          "flare::blas::Impl::Update<rank 1>::update: Z is const.  "
                          "It must be nonconst, because it is an output argument "
                          "(we have to be able to write to its entries).");
            static_assert(
                    (int)ZV::rank == (int)XV::rank && (int)ZV::rank == (int)YV::rank,
                    "flare::blas::Impl::Update<rank 1>::update: "
                    "X, Y, and Z must have the same rank.");
            static_assert(ZV::rank == 1,
                          "flare::blas::Impl::Update<rank 1>::update: "
                          "XV, YV, and ZV must have rank 1.");
            flare::Profiling::pushRegion("flare::blas::update");

            const size_type numRows = X.extent(0);
            const size_type numCols = X.extent(1);
            int a = 2, b = 2, c = 2;

            if (alpha == ATA::zero()) {
                a = 0;
            } else {
                a = 2;
            }
            if (beta == ATB::zero()) {
                b = 0;
            } else {
                b = 2;
            }
            if (gamma == ATC::zero()) {
                c = 0;
            } else {
                c = 2;
            }

            if (numRows < static_cast<size_type>(INT_MAX) &&
                numRows * numCols < static_cast<size_type>(INT_MAX)) {
                typedef int index_type;
                V_Update_Generic<execution_space, XV, YV, ZV, index_type>(
                        space, alpha, X, beta, Y, gamma, Z, a, b, c);
            } else {
                typedef typename XV::size_type index_type;
                V_Update_Generic<execution_space, XV, YV, ZV, index_type>(
                        space, alpha, X, beta, Y, gamma, Z, a, b, c);
            }
            flare::Profiling::popRegion();
        }
    };
}  // namespace flare::blas::detail

#define FLARE_BLAS_UPDATE_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE,       \
                                         MEM_SPACE)                        \
  template struct Update<                                                  \
      EXEC_SPACE,                                                          \
      flare::View<const SCALAR*, LAYOUT,                                  \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                  \
                   flare::MemoryTraits<flare::Unmanaged> >,              \
      flare::View<const SCALAR*, LAYOUT,                                  \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                  \
                   flare::MemoryTraits<flare::Unmanaged> >,              \
      flare::View<SCALAR*, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>, \
                   flare::MemoryTraits<flare::Unmanaged> >,              \
      1>;


#define FLARE_BLAS_UPDATE_MV_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE,     \
                                            MEM_SPACE)                      \
  template struct Update<                                                   \
      EXEC_SPACE,                                                           \
      flare::View<const SCALAR**, LAYOUT,                                  \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                   \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      flare::View<const SCALAR**, LAYOUT,                                  \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                   \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      flare::View<SCALAR**, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>, \
                   flare::MemoryTraits<flare::Unmanaged> >,               \
      2>;


#endif  // FLARE_KERNEL_BLAS_UPDATE_IMPL_H_
