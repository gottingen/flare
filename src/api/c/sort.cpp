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
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <copy.hpp>
#include <handle.hpp>
#include <sort.hpp>
#include <sort_by_key.hpp>
#include <sort_index.hpp>
#include <fly/algorithm.h>
#include <fly/array.h>
#include <fly/defines.h>

#include <cstdio>

using fly::dim4;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
static inline fly_array sort(const fly_array in, const unsigned dim,
                            const bool isAscending) {
    const Array<T> &inArray = getArray<T>(in);
    return getHandle(sort<T>(inArray, dim, isAscending));
}

fly_err fly_sort(fly_array *out, const fly_array in, const unsigned dim,
               const bool isAscending) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();

        if (info.elements() == 0) { return fly_retain_array(out, in); }
        DIM_ASSERT(1, info.elements() > 0);

        fly_array val;

        switch (type) {
            case f32: val = sort<float>(in, dim, isAscending); break;
            case f64: val = sort<double>(in, dim, isAscending); break;
            case s32: val = sort<int>(in, dim, isAscending); break;
            case u32: val = sort<uint>(in, dim, isAscending); break;
            case s16: val = sort<short>(in, dim, isAscending); break;
            case u16: val = sort<ushort>(in, dim, isAscending); break;
            case s64: val = sort<intl>(in, dim, isAscending); break;
            case u64: val = sort<uintl>(in, dim, isAscending); break;
            case u8: val = sort<uchar>(in, dim, isAscending); break;
            case b8: val = sort<char>(in, dim, isAscending); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, val);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

template<typename T>
static inline void sort_index(fly_array *val, fly_array *idx, const fly_array in,
                              const unsigned dim, const bool isAscending) {
    const Array<T> &inArray = getArray<T>(in);

    // Initialize Dummy Arrays
    Array<T> valArray    = createEmptyArray<T>(fly::dim4());
    Array<uint> idxArray = createEmptyArray<uint>(fly::dim4());

    sort_index<T>(valArray, idxArray, inArray, dim, isAscending);
    *val = getHandle(valArray);
    *idx = getHandle(idxArray);
}

fly_err fly_sort_index(fly_array *out, fly_array *indices, const fly_array in,
                     const unsigned dim, const bool isAscending) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();

        if (info.elements() <= 0) {
            FLY_CHECK(fly_create_handle(out, 0, nullptr, type));
            FLY_CHECK(fly_create_handle(indices, 0, nullptr, type));
            return FLY_SUCCESS;
        }

        fly_array val;
        fly_array idx;

        switch (type) {
            case f32:
                sort_index<float>(&val, &idx, in, dim, isAscending);
                break;
            case f64:
                sort_index<double>(&val, &idx, in, dim, isAscending);
                break;
            case s32: sort_index<int>(&val, &idx, in, dim, isAscending); break;
            case u32: sort_index<uint>(&val, &idx, in, dim, isAscending); break;
            case s16:
                sort_index<short>(&val, &idx, in, dim, isAscending);
                break;
            case u16:
                sort_index<ushort>(&val, &idx, in, dim, isAscending);
                break;
            case s64: sort_index<intl>(&val, &idx, in, dim, isAscending); break;
            case u64:
                sort_index<uintl>(&val, &idx, in, dim, isAscending);
                break;
            case u8: sort_index<uchar>(&val, &idx, in, dim, isAscending); break;
            case b8: sort_index<char>(&val, &idx, in, dim, isAscending); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, val);
        std::swap(*indices, idx);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

template<typename Tk, typename Tv>
static inline void sort_by_key(fly_array *okey, fly_array *oval,
                               const fly_array ikey, const fly_array ival,
                               const unsigned dim, const bool isAscending) {
    const Array<Tk> &ikeyArray = getArray<Tk>(ikey);
    const Array<Tv> &ivalArray = getArray<Tv>(ival);

    // Initialize Dummy Arrays
    Array<Tk> okeyArray = createEmptyArray<Tk>(fly::dim4());
    Array<Tv> ovalArray = createEmptyArray<Tv>(fly::dim4());

    sort_by_key<Tk, Tv>(okeyArray, ovalArray, ikeyArray, ivalArray, dim,
                        isAscending);
    *okey = getHandle(okeyArray);
    *oval = getHandle(ovalArray);
}

template<typename Tk>
void sort_by_key_tmplt(fly_array *okey, fly_array *oval, const fly_array ikey,
                       const fly_array ival, const unsigned dim,
                       const bool isAscending) {
    const ArrayInfo &info = getInfo(ival);
    fly_dtype vtype        = info.getType();

    switch (vtype) {
        case f32:
            sort_by_key<Tk, float>(okey, oval, ikey, ival, dim, isAscending);
            break;
        case f64:
            sort_by_key<Tk, double>(okey, oval, ikey, ival, dim, isAscending);
            break;
        case c32:
            sort_by_key<Tk, cfloat>(okey, oval, ikey, ival, dim, isAscending);
            break;
        case c64:
            sort_by_key<Tk, cdouble>(okey, oval, ikey, ival, dim, isAscending);
            break;
        case s32:
            sort_by_key<Tk, int>(okey, oval, ikey, ival, dim, isAscending);
            break;
        case u32:
            sort_by_key<Tk, uint>(okey, oval, ikey, ival, dim, isAscending);
            break;
        case s16:
            sort_by_key<Tk, short>(okey, oval, ikey, ival, dim, isAscending);
            break;
        case u16:
            sort_by_key<Tk, ushort>(okey, oval, ikey, ival, dim, isAscending);
            break;
        case s64:
            sort_by_key<Tk, intl>(okey, oval, ikey, ival, dim, isAscending);
            break;
        case u64:
            sort_by_key<Tk, uintl>(okey, oval, ikey, ival, dim, isAscending);
            break;
        case u8:
            sort_by_key<Tk, uchar>(okey, oval, ikey, ival, dim, isAscending);
            break;
        case b8:
            sort_by_key<Tk, char>(okey, oval, ikey, ival, dim, isAscending);
            break;
        default: TYPE_ERROR(1, vtype);
    }
}

fly_err fly_sort_by_key(fly_array *out_keys, fly_array *out_values,
                      const fly_array keys, const fly_array values,
                      const unsigned dim, const bool isAscending) {
    try {
        const ArrayInfo &kinfo = getInfo(keys);
        fly_dtype ktype         = kinfo.getType();

        const ArrayInfo &vinfo = getInfo(values);

        DIM_ASSERT(4, kinfo.dims() == vinfo.dims());
        if (kinfo.elements() == 0) {
            FLY_CHECK(fly_create_handle(out_keys, 0, nullptr, ktype));
            FLY_CHECK(fly_create_handle(out_values, 0, nullptr, ktype));
            return FLY_SUCCESS;
        }

        TYPE_ASSERT(kinfo.isReal());

        fly_array oKey;
        fly_array oVal;

        switch (ktype) {
            case f32:
                sort_by_key_tmplt<float>(&oKey, &oVal, keys, values, dim,
                                         isAscending);
                break;
            case f64:
                sort_by_key_tmplt<double>(&oKey, &oVal, keys, values, dim,
                                          isAscending);
                break;
            case s32:
                sort_by_key_tmplt<int>(&oKey, &oVal, keys, values, dim,
                                       isAscending);
                break;
            case u32:
                sort_by_key_tmplt<uint>(&oKey, &oVal, keys, values, dim,
                                        isAscending);
                break;
            case s16:
                sort_by_key_tmplt<short>(&oKey, &oVal, keys, values, dim,
                                         isAscending);
                break;
            case u16:
                sort_by_key_tmplt<ushort>(&oKey, &oVal, keys, values, dim,
                                          isAscending);
                break;
            case s64:
                sort_by_key_tmplt<intl>(&oKey, &oVal, keys, values, dim,
                                        isAscending);
                break;
            case u64:
                sort_by_key_tmplt<uintl>(&oKey, &oVal, keys, values, dim,
                                         isAscending);
                break;
            case u8:
                sort_by_key_tmplt<uchar>(&oKey, &oVal, keys, values, dim,
                                         isAscending);
                break;
            case b8:
                sort_by_key_tmplt<char>(&oKey, &oVal, keys, values, dim,
                                        isAscending);
                break;
            default: TYPE_ERROR(1, ktype);
        }
        std::swap(*out_keys, oKey);
        std::swap(*out_values, oVal);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
