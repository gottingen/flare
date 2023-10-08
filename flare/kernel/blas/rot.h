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

#ifndef FLARE_KERNEL_BLAS_ROT_H_
#define FLARE_KERNEL_BLAS_ROT_H_
#include <flare/kernel/blas/rot_impl.h>
namespace flare::blas {

    template <class execution_space, class VectorView, class ScalarView>
    void rot(execution_space const& space, VectorView const& X, VectorView const& Y,
             ScalarView const& c, ScalarView const& s) {
        static_assert(flare::is_execution_space<execution_space>::value,
                      "rot: execution_space template parameter is not a flare "
                      "execution space.");
        static_assert(VectorView::rank == 1,
                      "rot: VectorView template parameter needs to be a rank 1 view");
        static_assert(ScalarView::rank == 0,
                      "rot: ScalarView template parameter needs to be a rank 0 view");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename VectorView::memory_space>::accessible,
                "rot: VectorView template parameter memory space needs to be accessible "
                "from "
                "execution_space template parameter");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename ScalarView::memory_space>::accessible,
                "rot: VectorView template parameter memory space needs to be accessible "
                "from "
                "execution_space template parameter");
        static_assert(
                std::is_same<typename VectorView::non_const_value_type,
                        typename VectorView::value_type>::value,
                "rot: VectorView template parameter needs to store non-const values");

        using VectorView_Internal = flare::View<
                typename VectorView::non_const_value_type*,
                typename flare::detail::GetUnifiedLayout<VectorView>::array_layout,
                flare::Device<execution_space, typename VectorView::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;

        using ScalarView_Internal = flare::View<
                typename ScalarView::non_const_value_type,
                typename flare::detail::GetUnifiedLayout<ScalarView>::array_layout,
                flare::Device<execution_space, typename ScalarView::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;

        VectorView_Internal X_(X), Y_(Y);
        ScalarView_Internal c_(c), s_(s);

        flare::Profiling::pushRegion("flare::blas::rot");
        flare::blas::detail::Rot<execution_space, VectorView_Internal, ScalarView_Internal>::rot(
                space, X_, Y_, c_, s_);
        flare::Profiling::popRegion();
    }

    template <class VectorView, class ScalarView>
    void rot(VectorView const& X, VectorView const& Y, ScalarView const& c,
             ScalarView const& s) {
        const typename VectorView::execution_space space =
                typename VectorView::execution_space();
        rot(space, X, Y, c, s);
    }

}  // namespace flare::blas
#endif  // FLARE_KERNEL_BLAS_ROT_H_
