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
#include <common/err_common.hpp>
#include <stdio.h>

#define CUDA_NOT_SUPPORTED(message)                                         \
    do {                                                                    \
        throw SupportError(__FLY_FUNC__, __FLY_FILENAME__, __LINE__, message, \
                           boost::stacktrace::stacktrace());                \
    } while (0)

#define CU_CHECK(fn)                                                          \
    do {                                                                      \
        CUresult res = fn;                                                    \
        if (res == CUDA_SUCCESS) break;                                       \
        char cu_err_msg[1024];                                                \
        const char* cu_err_name;                                              \
        const char* cu_err_string;                                            \
        CUresult nameErr, strErr;                                             \
        nameErr = cuGetErrorName(res, &cu_err_name);                          \
        strErr  = cuGetErrorString(res, &cu_err_string);                      \
        if (nameErr == CUDA_SUCCESS && strErr == CUDA_SUCCESS) {              \
            snprintf(cu_err_msg, sizeof(cu_err_msg), "CU Error %s(%d): %s\n", \
                     cu_err_name, (int)(res), cu_err_string);                 \
            FLY_ERROR(cu_err_msg, FLY_ERR_INTERNAL);                            \
        } else {                                                              \
            FLY_ERROR("CU Unknown error.\n", FLY_ERR_INTERNAL);                 \
        }                                                                     \
    } while (0)

#define CUDA_CHECK(fn)                                               \
    do {                                                             \
        cudaError_t _cuda_error = fn;                                \
        if (_cuda_error != cudaSuccess) {                            \
            char cuda_err_msg[1024];                                 \
            snprintf(cuda_err_msg, sizeof(cuda_err_msg),             \
                     "CUDA Error (%d): %s\n", (int)(_cuda_error),    \
                     cudaGetErrorString(cudaGetLastError()));        \
                                                                     \
            if (_cuda_error == cudaErrorMemoryAllocation) {          \
                FLY_ERROR(cuda_err_msg, FLY_ERR_NO_MEM);               \
            } else if (_cuda_error == cudaErrorDevicesUnavailable) { \
                FLY_ERROR(cuda_err_msg, FLY_ERR_DRIVER);               \
            } else {                                                 \
                FLY_ERROR(cuda_err_msg, FLY_ERR_INTERNAL);             \
            }                                                        \
        }                                                            \
    } while (0)
