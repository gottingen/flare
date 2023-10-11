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


#ifndef FLARE_KERNEL_BLAS_ROT_IMPL_H_
#define FLARE_KERNEL_BLAS_ROT_IMPL_H_

#include <flare/core.h>
#include <flare/kernel/common/inner_product_space_traits.h>
#include <flare/kernel/blas/utility.h>
#include <flare/core/arith_traits.h>
#include <flare/core/layout_utility.h>

namespace flare::blas::detail {

    template <class VectorView, class ScalarView>
    struct rot_functor {
        using scalar_type = typename VectorView::non_const_value_type;

        VectorView X, Y;
        ScalarView c, s;

        rot_functor(VectorView const& X_, VectorView const& Y_, ScalarView const& c_,
                    ScalarView const& s_)
                : X(X_), Y(Y_), c(c_), s(s_) {}

        FLARE_INLINE_FUNCTION
        void operator()(const int entryIdx) const {
            const scalar_type temp = c() * X(entryIdx) + s() * Y(entryIdx);
            Y(entryIdx)            = c() * Y(entryIdx) - s() * X(entryIdx);
            X(entryIdx)            = temp;
        }
    };

    template <class ExecutionSpace, class VectorView, class ScalarView>
    void Rot_Invoke(ExecutionSpace const& space, VectorView const& X,
                    VectorView const& Y, ScalarView const& c, ScalarView const& s) {
        flare::RangePolicy<ExecutionSpace> rot_policy(space, 0, X.extent(0));
        rot_functor rot_func(X, Y, c, s);
        flare::parallel_for("flare::blas::rot", rot_policy, rot_func);
    }


    // Unification layer
    template <class ExecutionSpace, class VectorView, class ScalarView>
    struct Rot {
        static void rot(ExecutionSpace const& space, VectorView const& X,
                        VectorView const& Y, ScalarView const& c,
                        ScalarView const& s) {
            flare::Profiling::pushRegion("flare::blas::rot");
            Rot_Invoke<ExecutionSpace, VectorView, ScalarView>(space, X, Y, c, s);
            flare::Profiling::popRegion();
        }
    };

}  // namespace flare::blas::detail

//
// Macro for definition of full specialization of
// flare::blas::Impl::Rot.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define FLARE_BLAS_ROT_SPEC_INST(SCALAR, LAYOUT, EXECSPACE, MEMSPACE) \
  template struct Rot<                                                     \
      EXECSPACE,                                                           \
      flare::View<SCALAR*, LAYOUT, flare::Device<EXECSPACE, MEMSPACE>,   \
                   flare::MemoryTraits<flare::Unmanaged>>,               \
      flare::View<typename flare::ArithTraits<SCALAR>::mag_type, LAYOUT, \
                   flare::Device<EXECSPACE, MEMSPACE>,                    \
                   flare::MemoryTraits<flare::Unmanaged>>>;

#endif  // FLARE_KERNEL_BLAS_ROT_IMPL_H_
