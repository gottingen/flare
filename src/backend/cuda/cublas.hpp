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

#include <common/defines.hpp>
#include <common/unique_handle.hpp>
#include <cublas_v2.h>

DEFINE_HANDLER(cublasHandle_t, cublasCreate, cublasDestroy);

namespace flare {
namespace cuda {

const char* errorString(cublasStatus_t err);

#define CUBLAS_CHECK(fn)                                                    \
    do {                                                                    \
        cublasStatus_t _error = fn;                                         \
        if (_error != CUBLAS_STATUS_SUCCESS) {                              \
            char _err_msg[1024];                                            \
            snprintf(_err_msg, sizeof(_err_msg), "CUBLAS Error (%d): %s\n", \
                     (int)(_error), flare::cuda::errorString(_error));  \
            FLY_ERROR(_err_msg, FLY_ERR_INTERNAL);                            \
        }                                                                   \
    } while (0)

}  // namespace cuda
}  // namespace flare
