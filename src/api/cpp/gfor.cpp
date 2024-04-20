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

#include <fly/array.h>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/gfor.h>
#include <fly/seq.h>
#include "error.hpp"

namespace fly {

thread_local bool gforStatus;

bool gforGet() { return gforStatus; }
void gforSet(bool val) { gforStatus = val; }

bool gforToggle() {
    bool status = gforGet();
    status ^= 1U;
    gforSet(status);
    return status;
}

array batchFunc(const array &lhs, const array &rhs, batchFunc_t func) {
    if (gforGet()) {
        FLY_THROW_ERR("batchFunc can not be used inside GFOR", FLY_ERR_ARG);
    }
    gforSet(true);
    array res = func(lhs, rhs);
    gforSet(false);
    return res;
}

}  // namespace fly
