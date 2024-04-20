// Copyright 2023 The EA Authors.
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

#include <jit_test_api.h>

#include <backend.hpp>
#include <common/err_common.hpp>
#include <platform.hpp>

fly_err fly_get_max_jit_len(int *jitLen) {
    *jitLen = detail::getMaxJitSize();
    return FLY_SUCCESS;
}

fly_err fly_set_max_jit_len(const int maxJitLen) {
    try {
        ARG_ASSERT(1, maxJitLen > 0);
        detail::getMaxJitSize() = maxJitLen;
    }
    CATCHALL;
    return FLY_SUCCESS;
}
