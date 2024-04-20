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

#include <backend.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <optypes.hpp>
#include <scan.hpp>
#include <scan_by_key.hpp>
#include <fly/algorithm.h>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <complex>

using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<fly_op_t op, typename Ti, typename To>
static inline fly_array scan(const fly_array in, const int dim,
                            bool inclusive_scan = true) {
    return getHandle(scan<op, Ti, To>(getArray<Ti>(in), dim, inclusive_scan));
}

template<fly_op_t op, typename Ti, typename To>
static inline fly_array scan_key(const fly_array key, const fly_array in,
                                const int dim, bool inclusive_scan = true) {
    const ArrayInfo& key_info = getInfo(key);
    fly_dtype type             = key_info.getType();
    fly_array out;

    switch (type) {
        case s32:
            out = getHandle(scan<op, Ti, int, To>(
                getArray<int>(key), castArray<Ti>(in), dim, inclusive_scan));
            break;
        case u32:
            out = getHandle(scan<op, Ti, uint, To>(
                getArray<uint>(key), castArray<Ti>(in), dim, inclusive_scan));
            break;
        case s64:
            out = getHandle(scan<op, Ti, intl, To>(
                getArray<intl>(key), castArray<Ti>(in), dim, inclusive_scan));
            break;
        case u64:
            out = getHandle(scan<op, Ti, uintl, To>(
                getArray<uintl>(key), castArray<Ti>(in), dim, inclusive_scan));
            break;
        default: TYPE_ERROR(1, type);
    }
    return out;
}

template<typename Ti, typename To>
static inline fly_array scan_op(const fly_array key, const fly_array in,
                               const int dim, fly_binary_op op,
                               bool inclusive_scan = true) {
    fly_array out;

    switch (op) {
        case FLY_BINARY_ADD:
            out = scan_key<fly_add_t, Ti, To>(key, in, dim, inclusive_scan);
            break;
        case FLY_BINARY_MUL:
            out = scan_key<fly_mul_t, Ti, To>(key, in, dim, inclusive_scan);
            break;
        case FLY_BINARY_MIN:
            out = scan_key<fly_min_t, Ti, To>(key, in, dim, inclusive_scan);
            break;
        case FLY_BINARY_MAX:
            out = scan_key<fly_max_t, Ti, To>(key, in, dim, inclusive_scan);
            break;
        default:
            FLY_ERROR("Incorrect binary operation enum for argument number 3",
                     FLY_ERR_ARG);
            break;
    }
    return out;
}

template<typename Ti, typename To>
static inline fly_array scan_op(const fly_array in, const int dim,
                               fly_binary_op op, bool inclusive_scan) {
    fly_array out;

    switch (op) {
        case FLY_BINARY_ADD:
            out = scan<fly_add_t, Ti, To>(in, dim, inclusive_scan);
            break;
        case FLY_BINARY_MUL:
            out = scan<fly_mul_t, Ti, To>(in, dim, inclusive_scan);
            break;
        case FLY_BINARY_MIN:
            out = scan<fly_min_t, Ti, To>(in, dim, inclusive_scan);
            break;
        case FLY_BINARY_MAX:
            out = scan<fly_max_t, Ti, To>(in, dim, inclusive_scan);
            break;
        default:
            FLY_ERROR("Incorrect binary operation enum for argument number 2",
                     FLY_ERR_ARG);
            break;
    }
    return out;
}

fly_err fly_accum(fly_array* out, const fly_array in, const int dim) {
    try {
        ARG_ASSERT(2, dim >= 0);
        ARG_ASSERT(2, dim < 4);

        const ArrayInfo& in_info = getInfo(in);

        if (dim >= static_cast<int>(in_info.ndims())) {
            *out = retain(in);
            return FLY_SUCCESS;
        }

        fly_dtype type = in_info.getType();
        fly_array res;

        switch (type) {
            case f32: res = scan<fly_add_t, float, float>(in, dim); break;
            case f64: res = scan<fly_add_t, double, double>(in, dim); break;
            case c32: res = scan<fly_add_t, cfloat, cfloat>(in, dim); break;
            case c64: res = scan<fly_add_t, cdouble, cdouble>(in, dim); break;
            case u32: res = scan<fly_add_t, uint, uint>(in, dim); break;
            case s32: res = scan<fly_add_t, int, int>(in, dim); break;
            case u64: res = scan<fly_add_t, uintl, uintl>(in, dim); break;
            case s64: res = scan<fly_add_t, intl, intl>(in, dim); break;
            case u16: res = scan<fly_add_t, ushort, uint>(in, dim); break;
            case s16: res = scan<fly_add_t, short, int>(in, dim); break;
            case u8: res = scan<fly_add_t, uchar, uint>(in, dim); break;
            // Make sure you are adding only "1" for every non zero value, even
            // if op == fly_add_t
            case b8: res = scan<fly_notzero_t, char, uint>(in, dim); break;
            default: TYPE_ERROR(1, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_scan(fly_array* out, const fly_array in, const int dim, fly_binary_op op,
               bool inclusive_scan) {
    try {
        ARG_ASSERT(2, dim >= 0);
        ARG_ASSERT(2, dim < 4);

        const ArrayInfo& in_info = getInfo(in);

        if (dim >= static_cast<int>(in_info.ndims())) {
            *out = retain(in);
            return FLY_SUCCESS;
        }

        fly_dtype type = in_info.getType();
        fly_array res;

        switch (type) {
            case f32:
                res = scan_op<float, float>(in, dim, op, inclusive_scan);
                break;
            case f64:
                res = scan_op<double, double>(in, dim, op, inclusive_scan);
                break;
            case c32:
                res = scan_op<cfloat, cfloat>(in, dim, op, inclusive_scan);
                break;
            case c64:
                res = scan_op<cdouble, cdouble>(in, dim, op, inclusive_scan);
                break;
            case u32:
                res = scan_op<uint, uint>(in, dim, op, inclusive_scan);
                break;
            case s32:
                res = scan_op<int, int>(in, dim, op, inclusive_scan);
                break;
            case u64:
                res = scan_op<uintl, uintl>(in, dim, op, inclusive_scan);
                break;
            case s64:
                res = scan_op<intl, intl>(in, dim, op, inclusive_scan);
                break;
            case u16:
                res = scan_op<ushort, uint>(in, dim, op, inclusive_scan);
                break;
            case s16:
                res = scan_op<short, int>(in, dim, op, inclusive_scan);
                break;
            case u8:
                res = scan_op<uchar, uint>(in, dim, op, inclusive_scan);
                break;
            case b8:
                res = scan_op<char, uint>(in, dim, op, inclusive_scan);
                break;
            default: TYPE_ERROR(1, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_scan_by_key(fly_array* out, const fly_array key, const fly_array in,
                      const int dim, fly_binary_op op, bool inclusive_scan) {
    try {
        ARG_ASSERT(2, dim >= 0);
        ARG_ASSERT(2, dim < 4);

        const ArrayInfo& in_info  = getInfo(in);
        const ArrayInfo& key_info = getInfo(key);

        if (dim >= static_cast<int>(in_info.ndims())) {
            *out = retain(in);
            return FLY_SUCCESS;
        }

        ARG_ASSERT(2, in_info.dims() == key_info.dims());

        fly_dtype type = in_info.getType();
        fly_array res;

        switch (type) {
            case f32:
                res = scan_op<float, float>(key, in, dim, op, inclusive_scan);
                break;
            case f64:
                res = scan_op<double, double>(key, in, dim, op, inclusive_scan);
                break;
            case c32:
                res = scan_op<cfloat, cfloat>(key, in, dim, op, inclusive_scan);
                break;
            case c64:
                res =
                    scan_op<cdouble, cdouble>(key, in, dim, op, inclusive_scan);
                break;
            case s16:
            case s32:
                res = scan_op<int, int>(key, in, dim, op, inclusive_scan);
                break;
            case u64:
                res = scan_op<uintl, uintl>(key, in, dim, op, inclusive_scan);
                break;
            case s64:
                res = scan_op<intl, intl>(key, in, dim, op, inclusive_scan);
                break;
            case u16:
            case u32:
            case u8:
            case b8:
                res = scan_op<uint, uint>(key, in, dim, op, inclusive_scan);
                break;
            default: TYPE_ERROR(1, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
