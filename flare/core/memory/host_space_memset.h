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

#ifndef FLARE_CORE_MEMORY_HOST_SPACE_MEMSET_H_
#define FLARE_CORE_MEMORY_HOST_SPACE_MEMSET_H_

#include <flare/core/defines.h>
#include <flare/core/memory/host_space.h>
#include <flare/core/common/memset_fwd.h>

#include <iostream>

namespace flare::detail {

    template<class T, class... P>
    struct ZeroMemset<HostSpace::execution_space, Tensor<T, P...>> {
        ZeroMemset(const HostSpace::execution_space &exec, const Tensor<T, P...> &dst,
                   typename Tensor<T, P...>::const_value_type &) {
            // We can't use exec.fence() directly since we don't have a full definition
            // of HostSpace here.
            hostspace_fence(exec);
            using ValueType = typename Tensor<T, P...>::value_type;
            std::memset(dst.data(), 0, sizeof(ValueType) * dst.size());
        }

        ZeroMemset(const Tensor<T, P...> &dst,
                   typename Tensor<T, P...>::const_value_type &) {
            using ValueType = typename Tensor<T, P...>::value_type;
            std::memset(dst.data(), 0, sizeof(ValueType) * dst.size());
        }
    };

}  // end namespace flare::detail

#endif  // FLARE_CORE_MEMORY_HOST_SPACE_MEMSET_H_
