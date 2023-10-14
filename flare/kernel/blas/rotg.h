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
    template <class execution_space, class STensorType, class MTensorType>
    void rotg(execution_space const& space, STensorType const& a, STensorType const& b,
              MTensorType const& c, STensorType const& s) {
        static_assert(STensorType::rank == 0,
                      "rotg: the inputs need to be rank 0 tensors");
        static_assert(MTensorType::rank == 0,
                      "rotg: the inputs need to be rank 0 tensors");
        static_assert(
                !flare::ArithTraits<typename MTensorType::value_type>::is_complex);
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename STensorType::memory_space>::accessible,
                "rotg: execution_space cannot access data in STensorType");
        static_assert(
                flare::SpaceAccessibility<execution_space,
                        typename MTensorType::memory_space>::accessible,
                "rotg: execution_space cannot access data in MTensorType");

        using STensor_Internal = flare::Tensor<
                typename STensorType::value_type,
                typename flare::detail::GetUnifiedLayout<STensorType>::array_layout,
                flare::Device<execution_space, typename STensorType::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;
        using MTensor_Internal = flare::Tensor<
                typename MTensorType::value_type,
                typename flare::detail::GetUnifiedLayout<MTensorType>::array_layout,
                flare::Device<execution_space, typename MTensorType::memory_space>,
                flare::MemoryTraits<flare::Unmanaged>>;

        STensor_Internal a_(a), b_(b), s_(s);
        MTensor_Internal c_(c);

        flare::Profiling::pushRegion("flare::blas::rotg");
        flare::blas::detail::Rotg<execution_space, STensor_Internal, MTensor_Internal>::rotg(space, a, b,
                                                                          c, s);
        flare::Profiling::popRegion();
    }

}  // namespace flare::blas
#endif  // FLARE_KERNEL_BLAS_ROTG_H_
