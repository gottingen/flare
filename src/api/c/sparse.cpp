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

#include <arith.hpp>
#include <backend.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <lookup.hpp>
#include <platform.hpp>
#include <sparse.hpp>
#include <sparse_handle.hpp>
#include <fly/algorithm.h>
#include <fly/array.h>
#include <fly/sparse.h>

using fly::dim4;
using flare::getSparseArray;
using flare::retainSparseHandle;
using flare::common::createArrayDataSparseArray;
using flare::common::createDeviceDataSparseArray;
using flare::common::createEmptySparseArray;
using flare::common::createHostDataSparseArray;
using flare::common::SparseArray;
using flare::common::SparseArrayBase;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::sparseConvertDenseToStorage;

namespace flare {

const SparseArrayBase &getSparseArrayBase(const fly_array in,
                                          bool device_check) {
    const SparseArrayBase *base =
        static_cast<SparseArrayBase *>(static_cast<void *>(in));

    if (!base->isSparse()) {
        FLY_ERROR(
            "Input is not a SparseArray and cannot be used in Sparse functions",
            FLY_ERR_ARG);
    }

    if (device_check &&
        base->getDevId() != static_cast<int>(detail::getActiveDeviceId())) {
        FLY_ERROR("Input Array not created on current device", FLY_ERR_DEVICE);
    }

    return *base;
}

////////////////////////////////////////////////////////////////////////////////
// Sparse Creation
////////////////////////////////////////////////////////////////////////////////
template<typename T>
fly_array createSparseArrayFromData(const dim4 &dims, const fly_array values,
                                   const fly_array rowIdx, const fly_array colIdx,
                                   const fly::storage stype) {
    SparseArray<T> sparse = createArrayDataSparseArray(
        dims, getArray<T>(values), getArray<int>(rowIdx), getArray<int>(colIdx),
        stype);
    return getHandle(sparse);
}

template<typename T>
fly_array createSparseArrayFromPtr(const fly::dim4 &dims, const dim_t nNZ,
                                  const T *const values,
                                  const int *const rowIdx,
                                  const int *const colIdx,
                                  const fly::storage stype,
                                  const fly::source source) {
    if (nNZ) {
        switch (source) {
            case flyHost:
                return getHandle(createHostDataSparseArray(
                    dims, nNZ, values, rowIdx, colIdx, stype));
                break;
            case flyDevice:
                return getHandle(createDeviceDataSparseArray(
                    dims, nNZ, const_cast<T *>(values),
                    const_cast<int *>(rowIdx), const_cast<int *>(colIdx),
                    stype));
                break;
        }
    }

    return getHandle(createEmptySparseArray<T>(dims, nNZ, stype));
}

template<typename T>
fly_array createSparseArrayFromDense(const fly_array _in,
                                    const fly_storage stype) {
    const Array<T> in = getArray<T>(_in);

    switch (stype) {
        case FLY_STORAGE_CSR:
            return getHandle(
                sparseConvertDenseToStorage<T, FLY_STORAGE_CSR>(in));
        case FLY_STORAGE_COO:
            return getHandle(
                sparseConvertDenseToStorage<T, FLY_STORAGE_COO>(in));
        case FLY_STORAGE_CSC:
            // return getHandle(sparseConvertDenseToStorage<T,
            // FLY_STORAGE_CSC>(in));
        default:
            FLY_ERROR("Storage type is out of range/unsupported", FLY_ERR_ARG);
    }
}

template<typename T>
fly_array sparseConvertStorage(const fly_array in_,
                              const fly_storage destStorage) {
    const SparseArray<T> in = getSparseArray<T>(in_);

    if (destStorage == FLY_STORAGE_DENSE) {
        // Returns a regular fly_array, not sparse
        switch (in.getStorage()) {
            case FLY_STORAGE_CSR:
                return getHandle(
                    detail::sparseConvertStorageToDense<T, FLY_STORAGE_CSR>(in));
            case FLY_STORAGE_COO:
                return getHandle(
                    detail::sparseConvertStorageToDense<T, FLY_STORAGE_COO>(in));
            default:
                FLY_ERROR("Invalid storage type of input array", FLY_ERR_ARG);
        }
    } else if (destStorage == FLY_STORAGE_CSR) {
        // Returns a sparse fly_array
        switch (in.getStorage()) {
            case FLY_STORAGE_CSR: return retainSparseHandle<T>(in_);
            case FLY_STORAGE_COO:
                return getHandle(
                    detail::sparseConvertStorageToStorage<T, FLY_STORAGE_CSR,
                                                          FLY_STORAGE_COO>(in));
            default:
                FLY_ERROR("Invalid storage type of input array", FLY_ERR_ARG);
        }
    } else if (destStorage == FLY_STORAGE_COO) {
        // Returns a sparse fly_array
        switch (in.getStorage()) {
            case FLY_STORAGE_CSR:
                return getHandle(
                    detail::sparseConvertStorageToStorage<T, FLY_STORAGE_COO,
                                                          FLY_STORAGE_CSR>(in));
            case FLY_STORAGE_COO: return retainSparseHandle<T>(in_);
            default:
                FLY_ERROR("Invalid storage type of input array", FLY_ERR_ARG);
        }
    }

    // Shoud never come here
    return NULL;
}

////////////////////////////////////////////////////////////////////////////////
// Get Functions
////////////////////////////////////////////////////////////////////////////////
template<typename T>
fly_array getSparseValues(const fly_array in) {
    return getHandle(getSparseArray<T>(in).getValues());
}

}  // namespace flare

using flare::createSparseArrayFromData;
using flare::createSparseArrayFromDense;
using flare::createSparseArrayFromPtr;
using flare::getSparseArrayBase;
using flare::getSparseValues;
using flare::sparseConvertStorage;

fly_err fly_create_sparse_array(fly_array *out, const dim_t nRows,
                              const dim_t nCols, const fly_array values,
                              const fly_array rowIdx, const fly_array colIdx,
                              const fly_storage stype) {
    try {
        // Checks:
        // rowIdx and colIdx arrays are of s32 type
        // values is of floating point type
        // if COO, rowIdx, colIdx and values should have same dims
        // if CRS, colIdx and values should have same dims, rowIdx.dims = nRows
        // if CRC, rowIdx and values should have same dims, colIdx.dims = nCols
        // stype is within acceptable range
        // type is floating type

        if (!(stype == FLY_STORAGE_CSR || stype == FLY_STORAGE_CSC ||
              stype == FLY_STORAGE_COO)) {
            FLY_ERROR("Storage type is out of range/unsupported", FLY_ERR_ARG);
        }

        const ArrayInfo &vInfo = getInfo(values);
        const ArrayInfo &rInfo = getInfo(rowIdx);
        const ArrayInfo &cInfo = getInfo(colIdx);

        TYPE_ASSERT(vInfo.isFloating());
        DIM_ASSERT(3, vInfo.isLinear());
        ARG_ASSERT(4, rInfo.getType() == s32);
        DIM_ASSERT(4, rInfo.isLinear());
        ARG_ASSERT(5, cInfo.getType() == s32);
        DIM_ASSERT(5, cInfo.isLinear());

        const dim_t nNZ = vInfo.elements();
        if (stype == FLY_STORAGE_COO) {
            DIM_ASSERT(4, rInfo.elements() == nNZ);
            DIM_ASSERT(5, cInfo.elements() == nNZ);
        } else if (stype == FLY_STORAGE_CSR) {
            DIM_ASSERT(4, (dim_t)rInfo.elements() == nRows + 1);
            DIM_ASSERT(5, cInfo.elements() == nNZ);
        } else if (stype == FLY_STORAGE_CSC) {
            DIM_ASSERT(4, rInfo.elements() == nNZ);
            DIM_ASSERT(5, (dim_t)cInfo.elements() == nCols + 1);
        }

        fly_array output = nullptr;

        dim4 dims(nRows, nCols);

        switch (vInfo.getType()) {
            case f32:
                output = createSparseArrayFromData<float>(dims, values, rowIdx,
                                                          colIdx, stype);
                break;
            case f64:
                output = createSparseArrayFromData<double>(dims, values, rowIdx,
                                                           colIdx, stype);
                break;
            case c32:
                output = createSparseArrayFromData<cfloat>(dims, values, rowIdx,
                                                           colIdx, stype);
                break;
            case c64:
                output = createSparseArrayFromData<cdouble>(
                    dims, values, rowIdx, colIdx, stype);
                break;
            default: TYPE_ERROR(1, vInfo.getType());
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_create_sparse_array_from_ptr(
    fly_array *out, const dim_t nRows, const dim_t nCols, const dim_t nNZ,
    const void *const values, const int *const rowIdx, const int *const colIdx,
    const fly_dtype type, const fly_storage stype, const fly_source source) {
    try {
        // Checks:
        // rowIdx and colIdx arrays are of s32 type
        // values is of floating point type
        // if COO, rowIdx, colIdx and values should have same dims
        // if CRS, colIdx and values should have same dims, rowIdx.dims = nRows
        // if CRC, rowIdx and values should have same dims, colIdx.dims = nCols
        // stype is within acceptable range
        // type is floating type
        if (!(stype == FLY_STORAGE_CSR || stype == FLY_STORAGE_CSC ||
              stype == FLY_STORAGE_COO)) {
            FLY_ERROR("Storage type is out of range/unsupported", FLY_ERR_ARG);
        }

        TYPE_ASSERT(type == f32 || type == f64 || type == c32 || type == c64);

        fly_array output = nullptr;

        dim4 dims(nRows, nCols);

        switch (type) {
            case f32:
                output = createSparseArrayFromPtr<float>(
                    dims, nNZ, static_cast<const float *>(values), rowIdx,
                    colIdx, stype, source);
                break;
            case f64:
                output = createSparseArrayFromPtr<double>(
                    dims, nNZ, static_cast<const double *>(values), rowIdx,
                    colIdx, stype, source);
                break;
            case c32:
                output = createSparseArrayFromPtr<cfloat>(
                    dims, nNZ, static_cast<const cfloat *>(values), rowIdx,
                    colIdx, stype, source);
                break;
            case c64:
                output = createSparseArrayFromPtr<cdouble>(
                    dims, nNZ, static_cast<const cdouble *>(values), rowIdx,
                    colIdx, stype, source);
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_create_sparse_array_from_dense(fly_array *out, const fly_array in,
                                         const fly_storage stype) {
    try {
        // Checks:
        // stype is within acceptable range
        // values is of floating point type

        const ArrayInfo &info = getInfo(in);

        if (!(stype == FLY_STORAGE_CSR || stype == FLY_STORAGE_CSC ||
              stype == FLY_STORAGE_COO)) {
            FLY_ERROR("Storage type is out of range/unsupported", FLY_ERR_ARG);
        }

        // Only matrices allowed
        DIM_ASSERT(1, info.ndims() == 2);

        TYPE_ASSERT(info.isFloating());

        fly_array output = 0;

        switch (info.getType()) {
            case f32:
                output = createSparseArrayFromDense<float>(in, stype);
                break;
            case f64:
                output = createSparseArrayFromDense<double>(in, stype);
                break;
            case c32:
                output = createSparseArrayFromDense<cfloat>(in, stype);
                break;
            case c64:
                output = createSparseArrayFromDense<cdouble>(in, stype);
                break;
            default: TYPE_ERROR(1, info.getType());
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_sparse_convert_to(fly_array *out, const fly_array in,
                            const fly_storage destStorage) {
    try {
        // Handle dense case
        const ArrayInfo &info = getInfo(in, false);
        if (!info.isSparse()) {  // If input is dense
            return fly_create_sparse_array_from_dense(out, in, destStorage);
        }

        fly_array output = nullptr;

        const SparseArrayBase &base = getSparseArrayBase(in);

        // Dense not allowed as input -> Should never happen with
        // SparseArrayBase CSC is currently not supported
        ARG_ASSERT(1, base.getStorage() != FLY_STORAGE_DENSE &&
                          base.getStorage() != FLY_STORAGE_CSC);

        // Conversion to and from CSC is not supported
        ARG_ASSERT(2, destStorage != FLY_STORAGE_CSC);

        if (base.getStorage() == destStorage) {
            // Return a reference
            FLY_CHECK(fly_retain_array(out, in));
            return FLY_SUCCESS;
        }

        switch (base.getType()) {
            case f32:
                output = sparseConvertStorage<float>(in, destStorage);
                break;
            case f64:
                output = sparseConvertStorage<double>(in, destStorage);
                break;
            case c32:
                output = sparseConvertStorage<cfloat>(in, destStorage);
                break;
            case c64:
                output = sparseConvertStorage<cdouble>(in, destStorage);
                break;
            default: FLY_ERROR("Output storage type is not valid", FLY_ERR_ARG);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_sparse_to_dense(fly_array *out, const fly_array in) {
    try {
        fly_array output = nullptr;

        const SparseArrayBase &base = getSparseArrayBase(in);

        // Dense not allowed as input -> Should never happen
        // To convert from dense to type, use the create* functions
        ARG_ASSERT(1, base.getStorage() != FLY_STORAGE_DENSE);

        switch (base.getType()) {
            case f32:
                output = sparseConvertStorage<float>(in, FLY_STORAGE_DENSE);
                break;
            case f64:
                output = sparseConvertStorage<double>(in, FLY_STORAGE_DENSE);
                break;
            case c32:
                output = sparseConvertStorage<cfloat>(in, FLY_STORAGE_DENSE);
                break;
            case c64:
                output = sparseConvertStorage<cdouble>(in, FLY_STORAGE_DENSE);
                break;
            default: FLY_ERROR("Output storage type is not valid", FLY_ERR_ARG);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_sparse_get_info(fly_array *values, fly_array *rows, fly_array *cols,
                          fly_storage *stype, const fly_array in) {
    try {
        if (values != NULL) { FLY_CHECK(fly_sparse_get_values(values, in)); }
        if (rows != NULL) { FLY_CHECK(fly_sparse_get_row_idx(rows, in)); }
        if (cols != NULL) { FLY_CHECK(fly_sparse_get_col_idx(cols, in)); }
        if (stype != NULL) { FLY_CHECK(fly_sparse_get_storage(stype, in)); }
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_sparse_get_values(fly_array *out, const fly_array in) {
    try {
        const SparseArrayBase base = getSparseArrayBase(in);

        fly_array output = nullptr;

        switch (base.getType()) {
            case f32: output = getSparseValues<float>(in); break;
            case f64: output = getSparseValues<double>(in); break;
            case c32: output = getSparseValues<cfloat>(in); break;
            case c64: output = getSparseValues<cdouble>(in); break;
            default: TYPE_ERROR(1, base.getType());
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_sparse_get_row_idx(fly_array *out, const fly_array in) {
    try {
        const SparseArrayBase base = getSparseArrayBase(in);
        *out                       = getHandle(base.getRowIdx());
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_sparse_get_col_idx(fly_array *out, const fly_array in) {
    try {
        const SparseArrayBase base = getSparseArrayBase(in);
        *out                       = getHandle(base.getColIdx());
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_sparse_get_nnz(dim_t *out, const fly_array in) {
    try {
        const SparseArrayBase base = getSparseArrayBase(in);
        *out                       = base.getNNZ();
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_sparse_get_storage(fly_storage *out, const fly_array in) {
    try {
        const SparseArrayBase base = getSparseArrayBase(in);
        *out                       = base.getStorage();
    }
    CATCHALL;
    return FLY_SUCCESS;
}
