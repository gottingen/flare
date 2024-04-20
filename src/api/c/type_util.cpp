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

#include <type_util.hpp>

#include <common/err_common.hpp>
#include <fly/half.h>
#include <fly/util.h>

size_t size_of(fly_dtype type) {
    try {
        switch (type) {
            case f32: return sizeof(float);
            case f64: return sizeof(double);
            case s32: return sizeof(int);
            case u32: return sizeof(unsigned);
            case u8: return sizeof(unsigned char);
            case b8: return sizeof(unsigned char);
            case c32: return sizeof(float) * 2;
            case c64: return sizeof(double) * 2;
            case s16: return sizeof(short);
            case u16: return sizeof(unsigned short);
            case s64: return sizeof(long long);
            case u64: return sizeof(unsigned long long);
            case f16: return sizeof(fly_half);
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_get_size_of(size_t *size, fly_dtype type) {
    try {
        *size = size_of(type);
        return FLY_SUCCESS;
    }
    CATCHALL;
}
