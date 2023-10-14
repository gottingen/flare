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

#ifndef FLARE_KERNEL_BLAS_ROTM_H_
#define FLARE_KERNEL_BLAS_ROTM_H_

#include <flare/kernel/blas/rotm_impl.h>

namespace flare::blas {


    /// \brief Applies modified Givens rotation coefficients to vectors x and y.
    ///
    /// \tparam execution_space the execution space where the kernel will be
    ///         executed, it can be used to specify a stream too.
    /// \tparam VectorTensor a rank1 tensor type that hold non const data
    /// \tparam ParamTensor a rank1 tensor of static extent [5] type that
    ///         holds const data
    ///
    /// \param space [in]  execution space used for parallel loops in this kernel
    /// \param X [in/out] vector to be rotated with param coefficients
    /// \param Y [in/out] vector to be rotated with param coefficients
    /// \param param [in]  output of rotmg contains rotation coefficients
    ///
    template <class execution_space, class VectorTensor, class ParamTensor>
    void rotm(execution_space const& space, VectorTensor const& X,
              VectorTensor const& Y, ParamTensor const& param) {
        static_assert(flare::is_execution_space<execution_space>::value,
                      "rotm: execution_space template parameter is not a flare "
                      "execution space.");
        static_assert(
                VectorTensor::rank == 1,
                "rotm: VectorTensor template parameter needs to be a rank 1 tensor");
        static_assert(ParamTensor::rank == 1,
                      "rotm: ParamTensor template parameter needs to be a rank 1 tensor");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename VectorTensor::memory_space>::accessible,
                "rotm: VectorTensor template parameter memory space needs to be accessible "
                "from execution_space template parameter");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename ParamTensor::memory_space>::accessible,
                "rotm: ScalarTensor template parameter memory space needs to be accessible "
                "from execution_space template parameter");
        static_assert(
                std::is_same<typename VectorTensor::non_const_value_type,
                        typename VectorTensor::value_type>::value,
                "rotm: VectorTensor template parameter needs to store non-const values");
        static_assert(
                !flare::ArithTraits<typename VectorTensor::value_type>::is_complex,
                "rotm: VectorTensor template parameter cannot use complex value_type");
        static_assert(
                !flare::ArithTraits<typename ParamTensor::value_type>::is_complex,
                "rotm: ParamTensor template parameter cannot use complex value_type");

        using VectorTensor_Internal = flare::Tensor<
                typename VectorTensor::non_const_value_type*,
                typename flare::detail::GetUnifiedLayout<VectorTensor>::array_layout,
                flare::Device<execution_space, typename VectorTensor::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;

        using ParamTensor_Internal = flare::Tensor<
                typename ParamTensor::const_value_type[5],
                typename flare::detail::GetUnifiedLayout<ParamTensor>::array_layout,
                flare::Device<execution_space, typename ParamTensor::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;

        VectorTensor_Internal X_(X), Y_(Y);
        ParamTensor_Internal param_(param);

        flare::Profiling::pushRegion("flare::blas::rotm");
        flare::blas::detail::Rotm<execution_space, VectorTensor_Internal, ParamTensor_Internal>::rotm(
                space, X_, Y_, param_);
        flare::Profiling::popRegion();
    }

    template <class VectorTensor, class ParamTensor>
    void rotm(VectorTensor const& X, VectorTensor const& Y, ParamTensor const& param) {
        const typename VectorTensor::execution_space space =
                typename VectorTensor::execution_space();
        rotm(space, X, Y, param);
    }
}  // namespace flare::blas
#endif  // FLARE_KERNEL_BLAS_ROTM_H_
