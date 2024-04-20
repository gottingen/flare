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
#include <fly/internal.h>
#include "error.hpp"

namespace fly {
array createStridedArray(
    const void *data, const dim_t offset,
    const dim4 dims,     // NOLINT(performance-unnecessary-value-param)
    const dim4 strides,  // NOLINT(performance-unnecessary-value-param)
    const fly::dtype ty, const fly::source location) {
    fly_array res;
    FLY_THROW(fly_create_strided_array(&res, data, offset, dims.ndims(),
                                     dims.get(), strides.get(), ty, location));
    return array(res);
}

dim4 getStrides(const array &in) {
    dim_t s0, s1, s2, s3;
    FLY_THROW(fly_get_strides(&s0, &s1, &s2, &s3, in.get()));
    return dim4(s0, s1, s2, s3);
}

dim_t getOffset(const array &in) {
    dim_t offset;
    FLY_THROW(fly_get_offset(&offset, in.get()));
    return offset;
}

void *getRawPtr(const array &in) {
    void *ptr = NULL;
    FLY_THROW(fly_get_raw_ptr(&ptr, in.get()));
    return ptr;
}

bool isLinear(const array &in) {
    bool is_linear = false;
    FLY_THROW(fly_is_linear(&is_linear, in.get()));
    return is_linear;
}

bool isOwner(const array &in) {
    bool is_owner = false;
    FLY_THROW(fly_is_owner(&is_owner, in.get()));
    return is_owner;
}

}  // namespace fly
