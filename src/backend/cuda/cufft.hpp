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

#include <common/FFTPlanCache.hpp>
#include <common/err_common.hpp>
#include <common/unique_handle.hpp>
#include <cufft.h>
#include <cstdio>

DEFINE_HANDLER(cufftHandle, cufftCreate, cufftDestroy);

namespace flare {
namespace cuda {

typedef cufftHandle PlanType;
typedef std::shared_ptr<PlanType> SharedPlan;

const char *_cufftGetResultString(cufftResult res);

SharedPlan findPlan(int rank, int *n, int *inembed, int istride, int idist,
                    int *onembed, int ostride, int odist, cufftType type,
                    int batch);

class PlanCache : public common::FFTPlanCache<PlanCache, PlanType> {
    friend SharedPlan findPlan(int rank, int *n, int *inembed, int istride,
                               int idist, int *onembed, int ostride, int odist,
                               cufftType type, int batch);
};

}  // namespace cuda
}  // namespace flare

#define CUFFT_CHECK(fn)                                                   \
    do {                                                                  \
        cufftResult _cufft_res = fn;                                      \
        if (_cufft_res != CUFFT_SUCCESS) {                                \
            char cufft_res_msg[1024];                                     \
            snprintf(cufft_res_msg, sizeof(cufft_res_msg),                \
                     "cuFFT Error (%d): %s\n", (int)(_cufft_res),         \
                     flare::cuda::_cufftGetResultString(_cufft_res)); \
                                                                          \
            FLY_ERROR(cufft_res_msg, FLY_ERR_INTERNAL);                     \
        }                                                                 \
    } while (0)
