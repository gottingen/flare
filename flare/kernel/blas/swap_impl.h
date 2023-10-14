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

#ifndef FLARE_KERNEL_BLAS_SWAP_IMPL_H_
#define FLARE_KERNEL_BLAS_SWAP_IMPL_H_

#include <flare/core.h>

namespace flare::blas::detail {

    template <class XVector, class YVector>
    struct swap_functor {
        using scalar_type = typename XVector::non_const_value_type;

        XVector X;
        YVector Y;

        swap_functor(XVector const& X_, YVector const& Y_) : X(X_), Y(Y_) {}

        FLARE_INLINE_FUNCTION void operator()(int const entryIdx) const {
            scalar_type const temp = Y(entryIdx);
            Y(entryIdx)            = X(entryIdx);
            X(entryIdx)            = temp;
        }
    };

    template <class ExecutionSpace, class XVector, class YVector>
    void Swap_Invoke(ExecutionSpace const& space, XVector const& X,
                     YVector const& Y) {
        flare::RangePolicy<ExecutionSpace> swap_policy(space, 0, X.extent(0));
        swap_functor swap_func(X, Y);
        flare::parallel_for("flare::blas::swap", swap_policy, swap_func);
    }


    // Unification layer
    template <class ExecutionSpace, class XVector, class YVector>
    struct Swap {
        static void swap(ExecutionSpace const& space, XVector const& X,
                         YVector const& Y) {
            flare::Profiling::pushRegion("flare::blas::swap");
            Swap_Invoke<ExecutionSpace, XVector, YVector>(space, X, Y);
            flare::Profiling::popRegion();
        }
    };
}  // namespace flare::blas::detail

//
// Macro for definition of full specialization of
// flare::blas::Impl::Swap.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define FLARE_BLAS_SWAP_SPEC_INST(SCALAR, LAYOUT, EXECSPACE, MEMSPACE) \
  template struct Swap<                                                     \
      EXECSPACE,                                                            \
      flare::Tensor<SCALAR*, LAYOUT, flare::Device<EXECSPACE, MEMSPACE>,    \
                   flare::MemoryTraits<flare::Unmanaged>>,                \
      flare::Tensor<SCALAR*, LAYOUT, flare::Device<EXECSPACE, MEMSPACE>,    \
                   flare::MemoryTraits<flare::Unmanaged>>>;

#endif  // FLARE_KERNEL_BLAS_SWAP_IMPL_H_
