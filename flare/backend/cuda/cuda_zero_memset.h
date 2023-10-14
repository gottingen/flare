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
#ifndef FLARE_BACKEND_CUDA_CUDA_ZERO_MEMSET_H_
#define FLARE_BACKEND_CUDA_CUDA_ZERO_MEMSET_H_

#include <flare/core/defines.h>
#include <flare/backend/cuda/cuda.h>
#include <flare/core/common/zero_memset_fwd.h>

namespace flare::detail {

    template<class T, class... P>
    struct ZeroMemset<flare::Cuda, Tensor<T, P...>> {
        ZeroMemset(const flare::Cuda &exec_space_instance, const Tensor<T, P...> &dst,
                   typename Tensor<T, P...>::const_value_type &) {
            FLARE_IMPL_CUDA_SAFE_CALL(
                    (exec_space_instance.impl_internal_space_instance()
                            ->cuda_memset_async_wrapper(
                                    dst.data(), 0,
                                    dst.size() * sizeof(typename Tensor<T, P...>::value_type))));
        }

        ZeroMemset(const Tensor<T, P...> &dst,
                   typename Tensor<T, P...>::const_value_type &) {
            // FIXME_CUDA_MULTIPLE_DEVICES
            FLARE_IMPL_CUDA_SAFE_CALL(
                    (flare::detail::CudaInternal::singleton().cuda_memset_wrapper(
                            dst.data(), 0,
                            dst.size() * sizeof(typename Tensor<T, P...>::value_type))));
        }
    };

}  // namespace flare::detail

#endif  // FLARE_BACKEND_CUDA_CUDA_ZERO_MEMSET_H_
