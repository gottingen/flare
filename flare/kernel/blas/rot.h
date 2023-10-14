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

    template <class execution_space, class VectorTensor, class ScalarTensor>
    void rot(execution_space const& space, VectorTensor const& X, VectorTensor const& Y,
             ScalarTensor const& c, ScalarTensor const& s) {
        static_assert(flare::is_execution_space<execution_space>::value,
                      "rot: execution_space template parameter is not a flare "
                      "execution space.");
        static_assert(VectorTensor::rank == 1,
                      "rot: VectorTensor template parameter needs to be a rank 1 tensor");
        static_assert(ScalarTensor::rank == 0,
                      "rot: ScalarTensor template parameter needs to be a rank 0 tensor");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename VectorTensor::memory_space>::accessible,
                "rot: VectorTensor template parameter memory space needs to be accessible "
                "from "
                "execution_space template parameter");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename ScalarTensor::memory_space>::accessible,
                "rot: VectorTensor template parameter memory space needs to be accessible "
                "from "
                "execution_space template parameter");
        static_assert(
                std::is_same<typename VectorTensor::non_const_value_type,
                        typename VectorTensor::value_type>::value,
                "rot: VectorTensor template parameter needs to store non-const values");

        using VectorTensor_Internal = flare::Tensor<
                typename VectorTensor::non_const_value_type*,
                typename flare::detail::GetUnifiedLayout<VectorTensor>::array_layout,
                flare::Device<execution_space, typename VectorTensor::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;

        using ScalarTensor_Internal = flare::Tensor<
                typename ScalarTensor::non_const_value_type,
                typename flare::detail::GetUnifiedLayout<ScalarTensor>::array_layout,
                flare::Device<execution_space, typename ScalarTensor::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;

        VectorTensor_Internal X_(X), Y_(Y);
        ScalarTensor_Internal c_(c), s_(s);

        flare::Profiling::pushRegion("flare::blas::rot");
        flare::blas::detail::Rot<execution_space, VectorTensor_Internal, ScalarTensor_Internal>::rot(
                space, X_, Y_, c_, s_);
        flare::Profiling::popRegion();
    }

    template <class VectorTensor, class ScalarTensor>
    void rot(VectorTensor const& X, VectorTensor const& Y, ScalarTensor const& c,
             ScalarTensor const& s) {
        const typename VectorTensor::execution_space space =
                typename VectorTensor::execution_space();
        rot(space, X, Y, c, s);
    }

}  // namespace flare::blas
#endif  // FLARE_KERNEL_BLAS_ROT_H_
