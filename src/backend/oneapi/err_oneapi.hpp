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

#include <common/err_common.hpp>

#define ONEAPI_NOT_SUPPORTED(message)                                       \
    do {                                                                    \
        throw SupportError(__FLY_FUNC__, __FLY_FILENAME__, __LINE__, message, \
                           boost::stacktrace::stacktrace());                \
    } while (0)

#define CL_CHECK(call)                                                      \
    do {                                                                    \
        if (cl_int err = (call)) {                                          \
            char cl_err_msg[2048];                                          \
            const char* cl_err_call = #call;                                \
            snprintf(cl_err_msg, sizeof(cl_err_msg),                        \
                     "CL Error %s(%d): %d = %s\n", __FILE__, __LINE__, err, \
                     cl_err_call);                                          \
            FLY_ERROR(cl_err_msg, FLY_ERR_INTERNAL);                          \
        }                                                                   \
    } while (0)

#define CL_CHECK_BUILD(call)                                                  \
    do {                                                                      \
        if (cl_int err = (call)) {                                            \
            char log[8192];                                                   \
            char cl_err_msg[8192];                                            \
            const char* cl_err_call = #call;                                  \
            size_t log_ret;                                                   \
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 8192, log, \
                                  &log_ret);                                  \
            snprintf(cl_err_msg, sizeof(cl_err_msg),                          \
                     "OpenCL Error building %s(%d): %d = %s\nLog:\n%s",       \
                     __FILE__, __LINE__, err, cl_err_call, log);              \
            FLY_ERROR(cl_err_msg, FLY_ERR_INTERNAL);                            \
        }                                                                     \
    } while (0)
