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
#include <common/half.hpp>
#include <copy.hpp>
#include <handle.hpp>
#include <platform.hpp>
#include <sparse.hpp>
#include <sparse_handle.hpp>
#include <fly/sparse.h>

using fly::dim4;
using flare::copyData;
using flare::copySparseArray;
using flare::getSparseArrayBase;
using flare::getUseCount;
using flare::releaseHandle;
using flare::releaseSparseHandle;
using flare::retainSparseHandle;
using flare::common::half;
using flare::common::SparseArrayBase;
using detail::cdouble;
using detail::cfloat;
using detail::createDeviceDataArray;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

fly_err fly_get_data_ptr(void *data, const fly_array arr) {
    try {
        fly_dtype type = getInfo(arr).getType();
        // clang-format off
        switch (type) {
            case f32: copyData(static_cast<float*   >(data), arr); break;
            case c32: copyData(static_cast<cfloat*  >(data), arr); break;
            case f64: copyData(static_cast<double*  >(data), arr); break;
            case c64: copyData(static_cast<cdouble* >(data), arr); break;
            case b8:  copyData(static_cast<char*    >(data), arr); break;
            case s32: copyData(static_cast<int*     >(data), arr); break;
            case u32: copyData(static_cast<unsigned*>(data), arr); break;
            case u8:  copyData(static_cast<uchar*   >(data), arr); break;
            case s64: copyData(static_cast<intl*    >(data), arr); break;
            case u64: copyData(static_cast<uintl*   >(data), arr); break;
            case s16: copyData(static_cast<short*   >(data), arr); break;
            case u16: copyData(static_cast<ushort*  >(data), arr); break;
            case f16: copyData(static_cast<half*    >(data), arr); break;
            default: TYPE_ERROR(1, type);
        }
        // clang-format on
    }
    CATCHALL;
    return FLY_SUCCESS;
}

// Strong Exception Guarantee
fly_err fly_create_array(fly_array *result, const void *const data,
                       const unsigned ndims, const dim_t *const dims,
                       const fly_dtype type) {
    try {
        fly_array out;
        FLY_CHECK(fly_init());

        dim4 d = verifyDims(ndims, dims);

        switch (type) {
            case f32:
                out = createHandleFromData(d, static_cast<const float *>(data));
                break;
            case c32:
                out =
                    createHandleFromData(d, static_cast<const cfloat *>(data));
                break;
            case f64:
                out =
                    createHandleFromData(d, static_cast<const double *>(data));
                break;
            case c64:
                out =
                    createHandleFromData(d, static_cast<const cdouble *>(data));
                break;
            case b8:
                out = createHandleFromData(d, static_cast<const char *>(data));
                break;
            case s32:
                out = createHandleFromData(d, static_cast<const int *>(data));
                break;
            case u32:
                out = createHandleFromData(d, static_cast<const uint *>(data));
                break;
            case u8:
                out = createHandleFromData(d, static_cast<const uchar *>(data));
                break;
            case s64:
                out = createHandleFromData(d, static_cast<const intl *>(data));
                break;
            case u64:
                out = createHandleFromData(d, static_cast<const uintl *>(data));
                break;
            case s16:
                out = createHandleFromData(d, static_cast<const short *>(data));
                break;
            case u16:
                out =
                    createHandleFromData(d, static_cast<const ushort *>(data));
                break;
            case f16:
                out = createHandleFromData(d, static_cast<const half *>(data));
                break;
            default: TYPE_ERROR(4, type);
        }
        std::swap(*result, out);
    }
    CATCHALL
    return FLY_SUCCESS;
}

// Strong Exception Guarantee
fly_err fly_create_handle(fly_array *result, const unsigned ndims,
                        const dim_t *const dims, const fly_dtype type) {
    try {
        FLY_CHECK(fly_init());

        if (ndims > 0) { ARG_ASSERT(2, ndims > 0 && dims != NULL); }

        dim4 d(0);
        for (unsigned i = 0; i < ndims; i++) { d[i] = dims[i]; }

        fly_array out = createHandle(d, type);
        std::swap(*result, out);
    }
    CATCHALL
    return FLY_SUCCESS;
}

// Strong Exception Guarantee
fly_err fly_copy_array(fly_array *out, const fly_array in) {
    try {
        const ArrayInfo &info = getInfo(in, false);
        const fly_dtype type   = info.getType();

        fly_array res = 0;
        if (info.isSparse()) {
            const SparseArrayBase sbase = getSparseArrayBase(in);
            if (info.ndims() == 0) {
                return fly_create_sparse_array_from_ptr(
                    out, info.dims()[0], info.dims()[1], 0, nullptr, nullptr,
                    nullptr, type, sbase.getStorage(), flyDevice);
            }
            switch (type) {
                case f32: res = copySparseArray<float>(in); break;
                case f64: res = copySparseArray<double>(in); break;
                case c32: res = copySparseArray<cfloat>(in); break;
                case c64: res = copySparseArray<cdouble>(in); break;
                default: TYPE_ERROR(0, type);
            }

        } else {
            if (info.ndims() == 0) {
                return fly_create_handle(out, 0, nullptr, type);
            }
            switch (type) {
                case f32: res = copyArray<float>(in); break;
                case c32: res = copyArray<cfloat>(in); break;
                case f64: res = copyArray<double>(in); break;
                case c64: res = copyArray<cdouble>(in); break;
                case b8: res = copyArray<char>(in); break;
                case s32: res = copyArray<int>(in); break;
                case u32: res = copyArray<uint>(in); break;
                case u8: res = copyArray<uchar>(in); break;
                case s64: res = copyArray<intl>(in); break;
                case u64: res = copyArray<uintl>(in); break;
                case s16: res = copyArray<short>(in); break;
                case u16: res = copyArray<ushort>(in); break;
                case f16: res = copyArray<half>(in); break;
                default: TYPE_ERROR(1, type);
            }
        }
        std::swap(*out, res);
    }
    CATCHALL
    return FLY_SUCCESS;
}

// Strong Exception Guarantee
fly_err fly_get_data_ref_count(int *use_count, const fly_array in) {
    try {
        const ArrayInfo &info = getInfo(in, false);
        const fly_dtype type   = info.getType();

        int res;
        switch (type) {
            case f32: res = getUseCount<float>(in); break;
            case c32: res = getUseCount<cfloat>(in); break;
            case f64: res = getUseCount<double>(in); break;
            case c64: res = getUseCount<cdouble>(in); break;
            case b8: res = getUseCount<char>(in); break;
            case s32: res = getUseCount<int>(in); break;
            case u32: res = getUseCount<uint>(in); break;
            case u8: res = getUseCount<uchar>(in); break;
            case s64: res = getUseCount<intl>(in); break;
            case u64: res = getUseCount<uintl>(in); break;
            case s16: res = getUseCount<short>(in); break;
            case u16: res = getUseCount<ushort>(in); break;
            case f16: res = getUseCount<half>(in); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*use_count, res);
    }
    CATCHALL
    return FLY_SUCCESS;
}

fly_err fly_release_array(fly_array arr) {
    try {
        if (arr == 0) { return FLY_SUCCESS; }
        const ArrayInfo &info = getInfo(arr, false);
        fly_dtype type         = info.getType();

        if (info.isSparse()) {
            switch (type) {
                case f32: releaseSparseHandle<float>(arr); break;
                case f64: releaseSparseHandle<double>(arr); break;
                case c32: releaseSparseHandle<cfloat>(arr); break;
                case c64: releaseSparseHandle<cdouble>(arr); break;
                default: TYPE_ERROR(0, type);
            }
        } else {
            switch (type) {
                case f32: releaseHandle<float>(arr); break;
                case c32: releaseHandle<cfloat>(arr); break;
                case f64: releaseHandle<double>(arr); break;
                case c64: releaseHandle<cdouble>(arr); break;
                case b8: releaseHandle<char>(arr); break;
                case s32: releaseHandle<int>(arr); break;
                case u32: releaseHandle<uint>(arr); break;
                case u8: releaseHandle<uchar>(arr); break;
                case s64: releaseHandle<intl>(arr); break;
                case u64: releaseHandle<uintl>(arr); break;
                case s16: releaseHandle<short>(arr); break;
                case u16: releaseHandle<ushort>(arr); break;
                case f16: releaseHandle<half>(arr); break;
                default: TYPE_ERROR(0, type);
            }
        }
    }
    CATCHALL

    return FLY_SUCCESS;
}

fly_err fly_retain_array(fly_array *out, const fly_array in) {
    try {
        *out = retain(in);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

template<typename T>
void write_array(fly_array arr, const T *const data, const size_t bytes,
                 fly_source src) {
    if (src == flyHost) {
        writeHostDataArray(getArray<T>(arr), data, bytes);
    } else {
        writeDeviceDataArray(getArray<T>(arr), data, bytes);
    }
}

fly_err fly_write_array(fly_array arr, const void *data, const size_t bytes,
                      fly_source src) {
    if (bytes == 0) { return FLY_SUCCESS; }
    try {
        fly_dtype type = getInfo(arr).getType();
        ARG_ASSERT(1, (data != nullptr));
        ARG_ASSERT(3, (src == flyHost || src == flyDevice));
        // FIXME ArrayInfo class no bytes method, hence commented
        // DIM_ASSERT(2, bytes <= getInfo(arr).bytes());

        switch (type) {
            case f32:
                write_array(arr, static_cast<const float *>(data), bytes, src);
                break;
            case c32:
                write_array(arr, static_cast<const cfloat *>(data), bytes, src);
                break;
            case f64:
                write_array(arr, static_cast<const double *>(data), bytes, src);
                break;
            case c64:
                write_array(arr, static_cast<const cdouble *>(data), bytes,
                            src);
                break;
            case b8:
                write_array(arr, static_cast<const char *>(data), bytes, src);
                break;
            case s32:
                write_array(arr, static_cast<const int *>(data), bytes, src);
                break;
            case u32:
                write_array(arr, static_cast<const uint *>(data), bytes, src);
                break;
            case u8:
                write_array(arr, static_cast<const uchar *>(data), bytes, src);
                break;
            case s64:
                write_array(arr, static_cast<const intl *>(data), bytes, src);
                break;
            case u64:
                write_array(arr, static_cast<const uintl *>(data), bytes, src);
                break;
            case s16:
                write_array(arr, static_cast<const short *>(data), bytes, src);
                break;
            case u16:
                write_array(arr, static_cast<const ushort *>(data), bytes, src);
                break;
            case f16:
                write_array(arr, static_cast<const half *>(data), bytes, src);
                break;
            default: TYPE_ERROR(4, type);
        }
    }
    CATCHALL
    return FLY_SUCCESS;
}

fly_err fly_get_elements(dim_t *elems, const fly_array arr) {
    try {
        // Do not check for device mismatch
        *elems = getInfo(arr, false).elements();
    }
    CATCHALL
    return FLY_SUCCESS;
}

fly_err fly_get_type(fly_dtype *type, const fly_array arr) {
    try {
        // Do not check for device mismatch
        *type = getInfo(arr, false).getType();
    }
    CATCHALL
    return FLY_SUCCESS;
}

fly_err fly_get_dims(dim_t *d0, dim_t *d1, dim_t *d2, dim_t *d3,
                   const fly_array in) {
    try {
        // Do not check for device mismatch
        const ArrayInfo &info = getInfo(in, false);
        *d0                   = info.dims()[0];
        *d1                   = info.dims()[1];
        *d2                   = info.dims()[2];
        *d3                   = info.dims()[3];
    }
    CATCHALL
    return FLY_SUCCESS;
}

fly_err fly_get_numdims(unsigned *nd, const fly_array in) {
    try {
        // Do not check for device mismatch
        const ArrayInfo &info = getInfo(in, false);
        *nd                   = info.ndims();
    }
    CATCHALL
    return FLY_SUCCESS;
}

#undef INSTANTIATE
#define INSTANTIATE(fn1, fn2)                           \
    fly_err fn1(bool *result, const fly_array in) {       \
        try {                                           \
            const ArrayInfo &info = getInfo(in, false); \
            *result               = info.fn2();         \
        }                                               \
        CATCHALL                                        \
        return FLY_SUCCESS;                              \
    }

INSTANTIATE(fly_is_empty, isEmpty)
INSTANTIATE(fly_is_scalar, isScalar)
INSTANTIATE(fly_is_row, isRow)
INSTANTIATE(fly_is_column, isColumn)
INSTANTIATE(fly_is_vector, isVector)
INSTANTIATE(fly_is_complex, isComplex)
INSTANTIATE(fly_is_real, isReal)
INSTANTIATE(fly_is_double, isDouble)
INSTANTIATE(fly_is_single, isSingle)
INSTANTIATE(fly_is_half, isHalf)
INSTANTIATE(fly_is_realfloating, isRealFloating)
INSTANTIATE(fly_is_floating, isFloating)
INSTANTIATE(fly_is_integer, isInteger)
INSTANTIATE(fly_is_bool, isBool)
INSTANTIATE(fly_is_sparse, isSparse)

#undef INSTANTIATE

template<typename T>
inline void getScalar(T *out, const fly_array &arr) {
    out[0] = getScalar<T>(getArray<T>(arr));
}

fly_err fly_get_scalar(void *output_value, const fly_array arr) {
    try {
        ARG_ASSERT(0, (output_value != NULL));

        const ArrayInfo &info = getInfo(arr);
        const fly_dtype type   = info.getType();

        switch (type) {
            case f32:
                getScalar<float>(reinterpret_cast<float *>(output_value), arr);
                break;
            case f64:
                getScalar<double>(reinterpret_cast<double *>(output_value),
                                  arr);
                break;
            case b8:
                getScalar<char>(reinterpret_cast<char *>(output_value), arr);
                break;
            case s32:
                getScalar<int>(reinterpret_cast<int *>(output_value), arr);
                break;
            case u32:
                getScalar<uint>(reinterpret_cast<uint *>(output_value), arr);
                break;
            case u8:
                getScalar<uchar>(reinterpret_cast<uchar *>(output_value), arr);
                break;
            case s64:
                getScalar<intl>(reinterpret_cast<intl *>(output_value), arr);
                break;
            case u64:
                getScalar<uintl>(reinterpret_cast<uintl *>(output_value), arr);
                break;
            case s16:
                getScalar<short>(reinterpret_cast<short *>(output_value), arr);
                break;
            case u16:
                getScalar<ushort>(reinterpret_cast<ushort *>(output_value),
                                  arr);
                break;
            case c32:
                getScalar<cfloat>(reinterpret_cast<cfloat *>(output_value),
                                  arr);
                break;
            case c64:
                getScalar<cdouble>(reinterpret_cast<cdouble *>(output_value),
                                   arr);
                break;
            case f16:
                getScalar<half>(static_cast<half *>(output_value), arr);
                break;
            default: TYPE_ERROR(4, type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}
