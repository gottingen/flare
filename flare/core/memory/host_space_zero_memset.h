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

#ifndef FLARE_CORE_MEMORY_HOST_SPACE_ZERO_MEMSET_H_
#define FLARE_CORE_MEMORY_HOST_SPACE_ZERO_MEMSET_H_

#include <flare/core/defines.h>
#include <flare/core/memory/host_space.h>
#include <flare/core/common/zero_memset_fwd.h>

#include <iostream>

namespace flare::detail {

    template<class T, class... P>
    struct ZeroMemset<HostSpace::execution_space, View<T, P...>> {
        ZeroMemset(const HostSpace::execution_space &exec, const View<T, P...> &dst,
                   typename View<T, P...>::const_value_type &) {
            // We can't use exec.fence() directly since we don't have a full definition
            // of HostSpace here.
            hostspace_fence(exec);
            using ValueType = typename View<T, P...>::value_type;
            std::memset(dst.data(), 0, sizeof(ValueType) * dst.size());
        }

        ZeroMemset(const View<T, P...> &dst,
                   typename View<T, P...>::const_value_type &) {
            using ValueType = typename View<T, P...>::value_type;
            std::memset(dst.data(), 0, sizeof(ValueType) * dst.size());
        }
    };

}  // end namespace flare::detail

#endif  // FLARE_CORE_MEMORY_HOST_SPACE_ZERO_MEMSET_H_
