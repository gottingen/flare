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
    /// \tparam VectorView a rank1 view type that hold non const data
    /// \tparam ParamView a rank1 view of static extent [5] type that
    ///         holds const data
    ///
    /// \param space [in]  execution space used for parallel loops in this kernel
    /// \param X [in/out] vector to be rotated with param coefficients
    /// \param Y [in/out] vector to be rotated with param coefficients
    /// \param param [in]  output of rotmg contains rotation coefficients
    ///
    template <class execution_space, class VectorView, class ParamView>
    void rotm(execution_space const& space, VectorView const& X,
              VectorView const& Y, ParamView const& param) {
        static_assert(flare::is_execution_space<execution_space>::value,
                      "rotm: execution_space template parameter is not a flare "
                      "execution space.");
        static_assert(
                VectorView::rank == 1,
                "rotm: VectorView template parameter needs to be a rank 1 view");
        static_assert(ParamView::rank == 1,
                      "rotm: ParamView template parameter needs to be a rank 1 view");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename VectorView::memory_space>::accessible,
                "rotm: VectorView template parameter memory space needs to be accessible "
                "from execution_space template parameter");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename ParamView::memory_space>::accessible,
                "rotm: ScalarView template parameter memory space needs to be accessible "
                "from execution_space template parameter");
        static_assert(
                std::is_same<typename VectorView::non_const_value_type,
                        typename VectorView::value_type>::value,
                "rotm: VectorView template parameter needs to store non-const values");
        static_assert(
                !flare::ArithTraits<typename VectorView::value_type>::is_complex,
                "rotm: VectorView template parameter cannot use complex value_type");
        static_assert(
                !flare::ArithTraits<typename ParamView::value_type>::is_complex,
                "rotm: ParamView template parameter cannot use complex value_type");

        using VectorView_Internal = flare::View<
                typename VectorView::non_const_value_type*,
                typename flare::detail::GetUnifiedLayout<VectorView>::array_layout,
                flare::Device<execution_space, typename VectorView::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;

        using ParamView_Internal = flare::View<
                typename ParamView::const_value_type[5],
                typename flare::detail::GetUnifiedLayout<ParamView>::array_layout,
                flare::Device<execution_space, typename ParamView::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;

        VectorView_Internal X_(X), Y_(Y);
        ParamView_Internal param_(param);

        flare::Profiling::pushRegion("flare::blas::rotm");
        flare::blas::detail::Rotm<execution_space, VectorView_Internal, ParamView_Internal>::rotm(
                space, X_, Y_, param_);
        flare::Profiling::popRegion();
    }

    template <class VectorView, class ParamView>
    void rotm(VectorView const& X, VectorView const& Y, ParamView const& param) {
        const typename VectorView::execution_space space =
                typename VectorView::execution_space();
        rotm(space, X, Y, param);
    }
}  // namespace flare::blas
#endif  // FLARE_KERNEL_BLAS_ROTM_H_
