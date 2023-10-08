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

#ifndef FLARE_KERNEL_BLAS_ROTG_H_
#define FLARE_KERNEL_BLAS_ROTG_H_

#include <flare/kernel/blas/rotg_impl.h>

namespace flare::blas {

    /// \brief Compute the coefficients to apply a Givens rotation.
    ///
    /// \tparam Scalar data type of inputs and outputs
    ///
    /// \param space [in] the execution space
    /// \param a [in/out] on input one of the values to rotate, on output the
    ///          rotated value
    /// \param b [in/out] on input one of the values to rotate, on
    ///          output the rotated value
    /// \param c [out] cosine value associated with the
    ///          rotation
    /// \param s [out] sine value associated with the rotation
    template <class execution_space, class SViewType, class MViewType>
    void rotg(execution_space const& space, SViewType const& a, SViewType const& b,
              MViewType const& c, SViewType const& s) {
        static_assert(SViewType::rank == 0,
                      "rotg: the inputs need to be rank 0 views");
        static_assert(MViewType::rank == 0,
                      "rotg: the inputs need to be rank 0 views");
        static_assert(
                !flare::ArithTraits<typename MViewType::value_type>::is_complex);
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename SViewType::memory_space>::accessible,
                "rotg: execution_space cannot access data in SViewType");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename MViewType::memory_space>::accessible,
                "rotg: execution_space cannot access data in MViewType");

        using SView_Internal = flare::View<
                typename SViewType::value_type,
                typename flare::detail::GetUnifiedLayout<SViewType>::array_layout,
                flare::Device<execution_space, typename SViewType::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;
        using MView_Internal = flare::View<
                typename MViewType::value_type,
                typename flare::detail::GetUnifiedLayout<MViewType>::array_layout,
                flare::Device<execution_space, typename MViewType::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;

        SView_Internal a_(a), b_(b), s_(s);
        MView_Internal c_(c);

        flare::Profiling::pushRegion("flare::blas::rotg");
        flare::blas::detail::Rotg<execution_space, SView_Internal, MView_Internal>::rotg(space, a, b,
                                                                          c, s);
        flare::Profiling::popRegion();
    }

}  // namespace flare::blas
#endif  // FLARE_KERNEL_BLAS_ROTG_H_
