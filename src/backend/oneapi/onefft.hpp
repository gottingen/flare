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
#include <memory.hpp>
#include <oneapi/mkl/dfti.hpp>

#include <cstdint>

namespace flare {
namespace oneapi {

using ::oneapi::mkl::dft::domain;
using ::oneapi::mkl::dft::precision;

using PlanType   = std::shared_ptr<void>;
using SharedPlan = std::shared_ptr<PlanType>;

template<precision p, domain d>
PlanType findPlan(int rank, const bool isInPlace, int *n,
                  std::int64_t *istrides, int ibatch, std::int64_t *ostrides,
                  int obatch, int nbatch);

class PlanCache : public common::FFTPlanCache<PlanCache, PlanType> {
    template<precision p, domain d>
    friend PlanType findPlan(int rank, const bool isInPlace, int *n,
                             std::int64_t *istrides, int ibatch,
                             std::int64_t *ostrides, int obatch, int nbatch);
};

}  // namespace oneapi
}  // namespace flare
