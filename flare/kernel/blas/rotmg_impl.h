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


#ifndef FLARE_KERNEL_BLAS_ROTMG_IMPL_H_
#define FLARE_KERNEL_BLAS_ROTMG_IMPL_H_

#include <flare/core.h>
#include <flare/kernel/common/inner_product_space_traits.h>
#include <flare/kernel/blas/utility.h>
#include <flare/core/arith_traits.h>
#include <flare/core/layout_utility.h>


namespace flare::blas::detail {

    template <class DXTensor, class YTensor, class PTensor>
    FLARE_INLINE_FUNCTION void rotmg_impl(DXTensor const& d1, DXTensor const& d2,
                                           DXTensor const& x1, YTensor const& y1,
                                           PTensor const& param) {
        using Scalar = typename DXTensor::non_const_value_type;

        const Scalar one  = flare::ArithTraits<Scalar>::one();
        const Scalar zero = flare::ArithTraits<Scalar>::zero();

        const Scalar gamma      = 4096;
        const Scalar gammasq    = 4096 * 4096;
        const Scalar gammasqinv = one / gammasq;

        Scalar flag = zero;
        Scalar h11 = zero, h12 = zero, h21 = zero, h22 = zero;

        // Quick exit if d1 negative
        if (d1() < 0) {
            flag = -one;

            d1() = zero;
            d2() = zero;
            x1() = zero;
        } else {
            Scalar p2 = d2() * y1();

            // Trivial case p2 == 0
            if (p2 == zero) {
                flag     = -(one + one);
                param(0) = flag;
                return;
            }

            // General case
            Scalar p1 = d1() * x1();
            Scalar q1 = p1 * x1();
            Scalar q2 = p2 * y1();
            if (flare::abs(q1) > flare::abs(q2)) {
                h21      = -y1() / x1();
                h12      = p2 / p1;
                Scalar u = one - h12 * h21;
                if (u > zero) {
                    flag = zero;
                    d1() = d1() / u;
                    d2() = d2() / u;
                    x1() = x1() * u;
                } else {
                    flag = -one;
                    h11  = zero;
                    h12  = zero;
                    h21  = zero;
                    h22  = zero;

                    d1() = zero;
                    d2() = zero;
                    x1() = zero;
                }
            } else {
                if (q2 < 0) {
                    flag = -one;
                    h11  = zero;
                    h12  = zero;
                    h21  = zero;
                    h22  = zero;

                    d1() = zero;
                    d2() = zero;
                    x1() = zero;
                } else {
                    flag       = one;
                    h11        = p1 / p2;
                    h22        = x1() / y1();
                    Scalar u   = one + h11 * h22;
                    Scalar tmp = d2() / u;
                    d2()       = d1() / u;
                    d1()       = tmp;
                    x1()       = y1() * u;
                }
            }

            // Rescale d1, h11 and h12
            if (d1() != zero) {
                while ((d1() <= gammasqinv) || (d1() >= gammasq)) {
                    if (flag == zero) {
                        h11  = one;
                        h22  = one;
                        flag = -one;
                    } else {
                        h21  = -one;
                        h12  = one;
                        flag = -one;
                    }

                    if (d1() <= gammasqinv) {
                        d1() = d1() * gammasq;
                        x1() = x1() / gamma;
                        h11  = h11 / gamma;
                        h12  = h12 / gamma;
                    } else {
                        d1() = d1() / gammasq;
                        x1() = x1() * gamma;
                        h11  = h11 * gamma;
                        h12  = h12 * gamma;
                    }
                }
            }

            // Rescale d2, h21 and h22
            if (d2() != zero) {
                while ((flare::abs(d2()) <= gammasqinv) ||
                       (flare::abs(d2()) >= gammasq)) {
                    if (flag == zero) {
                        h11  = one;
                        h22  = one;
                        flag = -one;
                    } else {
                        h21  = -one;
                        h12  = one;
                        flag = -one;
                    }

                    if (flare::abs(d2()) <= gammasqinv) {
                        d2() = d2() * gammasq;
                        h21  = h21 / gamma;
                        h22  = h22 / gamma;
                    } else {
                        d2() = d2() / gammasq;
                        h21  = h21 * gamma;
                        h22  = h22 * gamma;
                    }
                }
            }

            // Setup output parameters
            if (flag < zero) {
                param(1) = h11;
                param(2) = h21;
                param(3) = h12;
                param(4) = h22;
            } else if (flag == zero) {
                param(2) = h21;
                param(3) = h12;
            } else {
                param(1) = h11;
                param(4) = h22;
            }
            param(0) = flag;
        }
    }

    template <class DXTensor, class YTensor, class PTensor>
    struct rotmg_functor {
        using Scalar = typename DXTensor::non_const_value_type;

        DXTensor d1, d2, x1;
        YTensor y1;
        PTensor param;

        rotmg_functor(DXTensor& d1_, DXTensor& d2_, DXTensor& x1_, const YTensor& y1_,
                      PTensor& param_)
                : d1(d1_), d2(d2_), x1(x1_), y1(y1_), param(param_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const int) const { rotmg_impl(d1, d2, x1, y1, param); }
    };

    template <class execution_space, class DXTensor, class YTensor, class PTensor>
    void Rotmg_Invoke(execution_space const& space, DXTensor const& d1,
                      DXTensor const& d2, DXTensor const& x1, YTensor const& y1,
                      PTensor const& param) {
        using Scalar = typename DXTensor::value_type;
        static_assert(!flare::ArithTraits<Scalar>::is_complex,
                      "rotmg is not defined for complex types!");

        rotmg_functor myFunc(d1, d2, x1, y1, param);
        flare::RangePolicy<execution_space> rotmg_policy(space, 0, 1);
        flare::parallel_for("flare::blas::rotmg", rotmg_policy, myFunc);
    }

    // Unification layer

    template <class execution_space, class DXTensor, class YTensor, class PTensor>
    struct Rotmg {
        static void rotmg(execution_space const& space, DXTensor& d1, DXTensor& d2,
                          DXTensor& x1, YTensor& y1, PTensor& param) {
            flare::Profiling::pushRegion("flare::blas::rotmg");
            Rotmg_Invoke<execution_space, DXTensor, YTensor, PTensor>(space, d1, d2, x1, y1,
                                                                param);
            flare::Profiling::popRegion();
        }
    };
}  // namespace flare::blas::detail

//
// Macro for definition of full specialization of
// flare::blas::Impl::Rotmg.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define FLARE_BLAS_ROTMG_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  template struct Rotmg<                                                       \
      EXEC_SPACE,                                                              \
      flare::Tensor<SCALAR, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>,      \
                   flare::MemoryTraits<flare::Unmanaged>>,                   \
      flare::Tensor<const SCALAR, LAYOUT,                                       \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                      \
                   flare::MemoryTraits<flare::Unmanaged>>,                   \
      flare::Tensor<SCALAR[5], LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>,   \
                   flare::MemoryTraits<flare::Unmanaged>>>;


#endif  // FLARE_KERNEL_BLAS_ROTMG_IMPL_H_
