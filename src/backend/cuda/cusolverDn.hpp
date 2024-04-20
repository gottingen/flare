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

#pragma once

#include <common/unique_handle.hpp>
#include <cusolverDn.h>

DEFINE_HANDLER(cusolverDnHandle_t, cusolverDnCreate, cusolverDnDestroy);

namespace flare {
namespace cuda {

const char* errorString(cusolverStatus_t err);

#define CUSOLVER_CHECK(fn)                                                    \
    do {                                                                      \
        cusolverStatus_t _error = fn;                                         \
        if (_error != CUSOLVER_STATUS_SUCCESS) {                              \
            char _err_msg[1024];                                              \
            snprintf(_err_msg, sizeof(_err_msg), "CUSOLVER Error (%d): %s\n", \
                     (int)(_error), flare::cuda::errorString(_error));    \
                                                                              \
            FLY_ERROR(_err_msg, FLY_ERR_INTERNAL);                              \
        }                                                                     \
    } while (0)

}  // namespace cuda
}  // namespace flare
