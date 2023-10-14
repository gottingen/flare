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
#include <flare/core/arith_traits.h>
#include <flare/core/layout_utility.h>

namespace flare::blas::detail {

    template <class VectorTensor, class ParamTensor>
    struct rotm_functor {
        using Scalar = typename VectorTensor::non_const_value_type;

        // Dispatch tags
        struct minus_one_tag {};
        struct zero_tag {};
        struct one_tag {};

        VectorTensor X, Y;
        ParamTensor param;

        rotm_functor(VectorTensor const& X_, VectorTensor const& Y_,
                     ParamTensor const& param_)
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

    template <class execution_space, class VectorTensor, class ParamTensor>
    void Rotm_Invoke(execution_space const& space, VectorTensor const& X,
                     VectorTensor const& Y, ParamTensor const& param) {
        using Scalar = typename VectorTensor::value_type;
        static_assert(!flare::ArithTraits<Scalar>::is_complex,
                      "rotm is not defined for complex types!");

        Scalar const zero = flare::ArithTraits<Scalar>::zero();
        Scalar const one  = flare::ArithTraits<Scalar>::one();
        Scalar const two  = one + one;

        rotm_functor myFunc(X, Y, param);

        typename ParamTensor::HostMirror param_h = flare::create_mirror_tensor(param);
        flare::deep_copy(param_h, param);
        Scalar const flag = param_h(0);

        if (flag == -two) {
            return;
        } else if (flag == -one) {
            flare::RangePolicy<
                    execution_space,
                    typename rotm_functor<VectorTensor, ParamTensor>::minus_one_tag>
                    rotm_policy(space, 0, X.extent(0));
            flare::parallel_for("flare::blas::rotm_minus_one", rotm_policy, myFunc);
        } else if (flag == zero) {
            flare::RangePolicy<execution_space,
                    typename rotm_functor<VectorTensor, ParamTensor>::zero_tag>
                    rotm_policy(space, 0, X.extent(0));
            flare::parallel_for("flare::blas::rotm_zero", rotm_policy, myFunc);
        } else if (flag == one) {
            flare::RangePolicy<execution_space,
                    typename rotm_functor<VectorTensor, ParamTensor>::one_tag>
                    rotm_policy(space, 0, X.extent(0));
            flare::parallel_for("flare::blas::rotm_one", rotm_policy, myFunc);
        } else {
            throw std::runtime_error(
                    "flare::blas::rotm: param(0) is not -2, -1, 0 or 1!");
        }
    }

    // Unification layer
    template <class execution_space, class VectorTensor, class ParamTensor>
    struct Rotm {
        static void rotm(execution_space const& space, VectorTensor const& X,
                         VectorTensor const& Y, ParamTensor const& param) {
            flare::Profiling::pushRegion("flare::blas::rotm");
            Rotm_Invoke<execution_space, VectorTensor, ParamTensor>(space, X, Y, param);
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
      flare::Tensor<SCALAR*, LAYOUT, flare::Device<EXEC_SPACE, MEM_SPACE>,    \
                   flare::MemoryTraits<flare::Unmanaged>>,                  \
      flare::Tensor<const SCALAR[5], LAYOUT,                                   \
                   flare::Device<EXEC_SPACE, MEM_SPACE>,                     \
                   flare::MemoryTraits<flare::Unmanaged>>>;


#endif // FLARE_KERNEL_BLAS_ROTM_IMPL_H_
