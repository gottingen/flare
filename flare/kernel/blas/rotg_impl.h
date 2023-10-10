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

#ifndef FLARE_KERNEL_BLAS_ROTG_IMPL_H_
#define FLARE_KERNEL_BLAS_ROTG_IMPL_H_

#include <flare/core.h>
#include <flare/kernel/common/inner_product_space_traits.h>
#include <flare/kernel/blas/utility.h>
#include <flare/core/arith_traits.h>
#include <flare/kernel/common/helper.h>

namespace flare::blas::detail {


    template <class Scalar, class Magnitude,
            typename std::enable_if<!flare::ArithTraits<Scalar>::is_complex,
                    bool>::type = true>
    FLARE_INLINE_FUNCTION void rotg_impl(Scalar* a, Scalar* b, Magnitude* c,
                                          Scalar* s) {
        const Scalar one  = flare::ArithTraits<Scalar>::one();
        const Scalar zero = flare::ArithTraits<Scalar>::zero();

        const Scalar numerical_scaling = flare::abs(*a) + flare::abs(*b);
        if (numerical_scaling == zero) {
            *c = one;
            *s = zero;
            *a = zero;
            *b = zero;
        } else {
            const Scalar scaled_a = *a / numerical_scaling;
            const Scalar scaled_b = *b / numerical_scaling;
            Scalar norm = flare::sqrt(scaled_a * scaled_a + scaled_b * scaled_b) *
                          numerical_scaling;
            Scalar sign = flare::abs(*a) > flare::abs(*b) ? *a : *b;
            norm        = flare::copysign(norm, sign);
            *c          = *a / norm;
            *s          = *b / norm;

            Scalar z = one;
            if (flare::abs(*a) > flare::abs(*b)) {
                z = *s;
            }
            if ((flare::abs(*b) >= flare::abs(*a)) && (*c != zero)) {
                z = one / *c;
            }
            *a = norm;
            *b = z;
        }
    }

    template <class Scalar, class Magnitude,
            typename std::enable_if<flare::ArithTraits<Scalar>::is_complex,
                    bool>::type = true>
    FLARE_INLINE_FUNCTION void rotg_impl(Scalar* a, Scalar* b, Magnitude* c,
                                          Scalar* s) {
        using mag_type = typename flare::ArithTraits<Scalar>::mag_type;

        const Scalar one        = flare::ArithTraits<Scalar>::one();
        const Scalar zero       = flare::ArithTraits<Scalar>::zero();
        const mag_type mag_zero = flare::ArithTraits<mag_type>::zero();

        const mag_type numerical_scaling = flare::abs(*a) + flare::abs(*b);
        if (flare::abs(*a) == zero) {
            *c = mag_zero;
            *s = one;
            *a = *b;
        } else {
            const Scalar scaled_a = flare::abs(*a / numerical_scaling);
            const Scalar scaled_b = flare::abs(*b / numerical_scaling);
            mag_type norm =
                    flare::abs(flare::sqrt(scaled_a * scaled_a + scaled_b * scaled_b)) *
                    numerical_scaling;
            Scalar unit_a = *a / flare::abs(*a);
            *c            = flare::abs(*a) / norm;
            *s            = unit_a * flare::conj(*b) / norm;
            *a            = unit_a * norm;
        }
    }

    template <class SViewType, class MViewType>
    struct rotg_functor {
        SViewType a, b;
        MViewType c;
        SViewType s;

        rotg_functor(SViewType const& a_, SViewType const& b_, MViewType const& c_,
                     SViewType const& s_)
                : a(a_), b(b_), c(c_), s(s_) {}

        FLARE_INLINE_FUNCTION
        void operator()(int const) const {
            rotg_impl(a.data(), b.data(), c.data(), s.data());
        }
    };

    /// \brief Compute Givens rotation coefficients.
    template <class ExecutionSpace, class SViewType, class MViewType>
    void Rotg_Invoke(ExecutionSpace const& space, SViewType const& a,
                     SViewType const& b, MViewType const& c, SViewType const& s) {
        flare::RangePolicy<ExecutionSpace> rotg_policy(space, 0, 1);
        rotg_functor rotg_func(a, b, c, s);
        flare::parallel_for("flare::blas::rotg", rotg_policy, rotg_func);
    }

    // Unification layer
    template <class ExecutionSpace, class SViewType, class MViewType>
    struct Rotg{
        static void rotg(ExecutionSpace const& space, SViewType const& a,
                         SViewType const& b, MViewType const& c, SViewType const& s) {
            flare::Profiling::pushRegion("flare::blas::rotg");
            Rotg_Invoke<ExecutionSpace, SViewType, MViewType>(space, a, b, c, s);
            flare::Profiling::popRegion();
        }
    };
}  // namespace flare::blas::detail

//
// Macro for definition of full specialization of
// flare::blas::Impl::Rotg.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define FLARE_BLAS_ROTG_SPEC_INST(SCALAR, LAYOUT, EXECSPACE, MEMSPACE) \
  template struct Rotg<                                                     \
      EXECSPACE,                                                            \
      flare::View<SCALAR, LAYOUT, flare::Device<EXECSPACE, MEMSPACE>,     \
                   flare::MemoryTraits<flare::Unmanaged>>,                \
      flare::View<typename flare::ArithTraits<SCALAR>::mag_type, LAYOUT,  \
                   flare::Device<EXECSPACE, MEMSPACE>,                     \
                   flare::MemoryTraits<flare::Unmanaged>>>;

#endif  // FLARE_KERNEL_BLAS_ROTG_IMPL_H_
