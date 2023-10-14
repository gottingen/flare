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
    /// \tparam DXTensor a rank0 tensor type that hold non const data
    /// \tparam YTensor a rank0 tensor type that holds const data
    /// \tparam PTensor a rank1 tensor of
    ///         static extent 5 that holds non const data
    ///
    /// \param space [in] execution space used for parallel loops
    /// \param d1 [in/out]
    /// \param d2 [in/out]
    /// \param x1 [in/out]
    /// \param y1 [in]
    /// \param param [out]
    ///
    template <class execution_space, class DXTensor, class YTensor, class PTensor>
    void rotmg(execution_space const& space, DXTensor const& d1, DXTensor const& d2,
               DXTensor const& x1, YTensor const& y1, PTensor const& param) {
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename DXTensor::memory_space>::accessible,
                "rotmg: execution_space cannot access data in DXTensor");

        using DXTensor_Internal = flare::Tensor<
                typename DXTensor::value_type,
                typename flare::detail::GetUnifiedLayout<DXTensor>::array_layout,
                flare::Device<execution_space, typename DXTensor::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;

        using YTensor_Internal = flare::Tensor<
                typename YTensor::value_type,
                typename flare::detail::GetUnifiedLayout<YTensor>::array_layout,
                flare::Device<execution_space, typename YTensor::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;

        using PTensor_Internal = flare::Tensor<
                typename PTensor::value_type[5],
                typename flare::detail::GetUnifiedLayout<PTensor>::array_layout,
                flare::Device<execution_space, typename PTensor::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;

        DXTensor_Internal d1_(d1), d2_(d2), x1_(x1);
        YTensor_Internal y1_(y1);
        PTensor_Internal param_(param);

        flare::Profiling::pushRegion("flare::blas::rotmg");
        flare::blas::detail::Rotmg<execution_space, DXTensor_Internal, YTensor_Internal,
                PTensor_Internal>::rotmg(space, d1_, d2_, x1_, y1_, param_);
        flare::Profiling::popRegion();
    }

    template <class DXTensor, class YTensor, class PTensor>
    void rotmg(DXTensor const& d1, DXTensor const& d2, DXTensor const& x1,
               YTensor const& y1, PTensor const& param) {
        const typename PTensor::execution_space space =
                typename PTensor::execution_space();
        rotmg(space, d1, d2, x1, y1, param);
    }

}  // namespace  flare::blas

#endif  // FLARE_KERNEL_BLAS_ROTMG_H_
