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

#ifndef FLARE_KERNEL_BLAS_ROTM_IMPL_H_
#define FLARE_KERNEL_BLAS_ROTM_IMPL_H_

#include <flare/core.h>
#include <flare/kernel/common/inner_product_space_traits.h>
#include <flare/kernel/blas/utility.h>
#include <flare/kernel/common/arith_traits.h>
#include <flare/kernel/common/helper.h>

namespace flare::blas::detail {

    template <class VectorView, class ParamView>
    struct rotm_functor {
        using Scalar = typename VectorView::non_const_value_type;

        // Dispatch tags
        struct minus_one_tag {};
        struct zero_tag {};
        struct one_tag {};

        VectorView X, Y;
        ParamView param;

        rotm_functor(VectorView const& X_, VectorView const& Y_,
                     ParamView const& param_)
                : X(X_), Y(Y_), param(param_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const minus_one_tag&, const int idx) const {
            Scalar const tmp = X(idx);
            X(idx)           = param(1) * tmp + param(3) * Y(idx);
            Y(idx)           = param(2) * tmp + param(4) * Y(idx);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const zero_tag&, const int idx) const {
            Scalar const tmp = X(idx);
            X(idx)           = tmp + param(3) * Y(idx);
            Y(idx)           = param(2) * tmp + Y(idx);
        }

        FLARE_INLINE_FUNCTION
        void operator()(const one_tag&, const int idx) const {
            Scalar const tmp = X(idx);
            X(idx)           = param(1) * tmp + Y(idx);
            Y(idx)           = -tmp + param(4) * Y(idx);
        }
    };

    template <class execution_space, class VectorView, class ParamView>
    void Rotm_Invoke(execution_space const& space, VectorView const& X,
                     VectorView const& Y, ParamView const& param) {
        using Scalar = typename VectorView::value_type;
        static_assert(!flare::ArithTraits<Scalar>::is_complex,
                      "rotm is not defined for complex types!");

        Scalar const zero = flare::ArithTraits<Scalar>::zero();
        Scalar const one  = flare::ArithTraits<Scalar>::one();
        Scalar const two  = one + one;

        rotm_functor myFunc(X, Y, param);

        typename ParamView::HostMirror param_h = flare::create_mirror_view(param);
        flare::deep_copy(param_h, param);
        Scalar const flag = param_h(0);

        if (flag == -two) {
            return;
        } else if (flag == -one) {
            flare::RangePolicy<
                    execution_space,
                    typename rotm_functor<VectorView, ParamView>::minus_one_tag>
                    rotm_policy(space, 0, X.extent(0));
            flare::parallel_for("flare::blas::rotm_minus_one", rotm_policy, myFunc);
        } else if (flag == zero) {
            flare::RangePolicy<execution_space,
                    typename rotm_functor<VectorView, ParamView>::zero_tag>
                    rotm_policy(space, 0, X.extent(0));
            flare::parallel_for("flare::blas::rotm_zero", rotm_policy, myFunc);
        } else if (flag == one) {
            flare::RangePolicy<execution_space,
                    typename rotm_functor<VectorView, ParamView>::one_tag>
                    rotm_policy(space, 0, X.extent(0));
            flare::parallel_for("flare::blas::rotm_one", rotm_policy, myFunc);
        } else {
            throw std::runtime_error(
                    "flare::blas::rotm: param(0) is not -2, -1, 0 or 1!");
        }
    }

    // Unification layer
    template <class execution_space, class VectorView, class ParamView>
    struct Rotm {
        static void rotm(execution_space const& space, VectorView const& X,
                         VectorView const& Y, ParamView const& param) {
            flare::Profiling::pushRegion("flare::blas::rotm");
            Rotm_Invoke<execution_space, VectorView, ParamView>(space, X, Y, param);
            flare::Profiling::popRegion();
        }
    };
}  // namespace flare::blas::detail
//
// Macro for definition of full specialization of
// flare::blas::Impl::Rotm.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define FLARE_BLAS_ROTM_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  template struct Rotm<                                                       \
      EXEC_SPACE,                                                             \
      flare::View<SCALAR*, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>,    \
                   flare::MemoryTraits<flare::Unmanaged>>,                  \
      flare::View<const SCALAR[5], LAYOUT,                                   \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                     \
                   flare::MemoryTraits<flare::Unmanaged>>>;


#endif // FLARE_KERNEL_BLAS_ROTM_IMPL_H_
