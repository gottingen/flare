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

#include <clFFT.h>
#include <common/FFTPlanCache.hpp>
#include <memory.hpp>

#include <cstdio>

namespace flare {
namespace opencl {
typedef clfftPlanHandle PlanType;
typedef std::shared_ptr<PlanType> SharedPlan;

const char *_clfftGetResultString(clfftStatus st);

SharedPlan findPlan(clfftLayout iLayout, clfftLayout oLayout, clfftDim rank,
                    size_t *clLengths, size_t *istrides, size_t idist,
                    size_t *ostrides, size_t odist, clfftPrecision precision,
                    size_t batch);

class PlanCache : public common::FFTPlanCache<PlanCache, PlanType> {
    friend SharedPlan findPlan(clfftLayout iLayout, clfftLayout oLayout,
                               clfftDim rank, size_t *clLengths,
                               size_t *istrides, size_t idist, size_t *ostrides,
                               size_t odist, clfftPrecision precision,
                               size_t batch);
};
}  // namespace opencl
}  // namespace flare

#define CLFFT_CHECK(fn)                                          \
    do {                                                         \
        clfftStatus _clfft_st = fn;                              \
        if (_clfft_st != CLFFT_SUCCESS) {                        \
            opencl::signalMemoryCleanup();                       \
            _clfft_st = (fn);                                    \
        }                                                        \
        if (_clfft_st != CLFFT_SUCCESS) {                        \
            char clfft_st_msg[1024];                             \
            snprintf(clfft_st_msg, sizeof(clfft_st_msg),         \
                     "clFFT Error (%d): %s\n", (int)(_clfft_st), \
                     opencl::_clfftGetResultString(_clfft_st));  \
                                                                 \
            FLY_ERROR(clfft_st_msg, FLY_ERR_INTERNAL);             \
        }                                                        \
    } while (0)
