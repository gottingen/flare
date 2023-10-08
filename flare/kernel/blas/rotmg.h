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

#ifndef FLARE_KERNEL_BLAS_ROTMG_H_
#define FLARE_KERNEL_BLAS_ROTMG_H_
#include <flare/kernel/blas/rotmg_impl.h>

namespace flare::blas {
    /// \brief Compute the coefficients to apply a modified Givens rotation.
    ///
    /// \tparam execution_space the execution space where the kernel will be
    ///         executed
    /// \tparam DXView a rank0 view type that hold non const data
    /// \tparam YView a rank0 view type that holds const data
    /// \tparam PView a rank1 view of
    ///         static extent 5 that holds non const data
    ///
    /// \param space [in] execution space used for parallel loops
    /// \param d1 [in/out]
    /// \param d2 [in/out]
    /// \param x1 [in/out]
    /// \param y1 [in]
    /// \param param [out]
    ///
    template <class execution_space, class DXView, class YView, class PView>
    void rotmg(execution_space const& space, DXView const& d1, DXView const& d2,
               DXView const& x1, YView const& y1, PView const& param) {
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename DXView::memory_space>::accessible,
                "rotmg: execution_space cannot access data in DXView");

        using DXView_Internal = flare::View<
                typename DXView::value_type,
                typename flare::detail::GetUnifiedLayout<DXView>::array_layout,
                flare::Device<execution_space, typename DXView::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;

        using YView_Internal = flare::View<
                typename YView::value_type,
                typename flare::detail::GetUnifiedLayout<YView>::array_layout,
                flare::Device<execution_space, typename YView::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;

        using PView_Internal = flare::View<
                typename PView::value_type[5],
                typename flare::detail::GetUnifiedLayout<PView>::array_layout,
                flare::Device<execution_space, typename PView::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;

        DXView_Internal d1_(d1), d2_(d2), x1_(x1);
        YView_Internal y1_(y1);
        PView_Internal param_(param);

        flare::Profiling::pushRegion("flare::blas::rotmg");
        flare::blas::detail::Rotmg<execution_space, DXView_Internal, YView_Internal,
                PView_Internal>::rotmg(space, d1_, d2_, x1_, y1_, param_);
        flare::Profiling::popRegion();
    }

    template <class DXView, class YView, class PView>
    void rotmg(DXView const& d1, DXView const& d2, DXView const& x1,
               YView const& y1, PView const& param) {
        const typename PView::execution_space space =
                typename PView::execution_space();
        rotmg(space, d1, d2, x1, y1, param);
    }

}  // namespace  flare::blas

#endif  // FLARE_KERNEL_BLAS_ROTMG_H_
