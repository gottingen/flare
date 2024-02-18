/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

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
