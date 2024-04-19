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
#include <fly/defines.h>

#ifdef __cplusplus
namespace fly
{
    class array;

    /**
       This function converts \ref fly::array of values, row indices and column
       indices into a sparse array.

       \note This function only create references of these arrays into the
             sparse data structure and does not do deep copies.

       \param[in] nRows is the number of rows in the dense matrix
       \param[in] nCols is the number of columns in the dense matrix
       \param[in] values is the \ref fly::array containing the non-zero elements
                  of the matrix
       \param[in] rowIdx is the row indices for the sparse array
       \param[in] colIdx is the column indices for the sparse array
       \param[in] stype is the storage format of the sparse array
       \return \ref fly::array for the sparse array

       \snippet test/sparse.cpp ex_sparse_fly_arrays

       \ingroup sparse_func_create
     */
    FLY_API array sparse(const dim_t nRows, const dim_t nCols,
                       const array values, const array rowIdx, const array colIdx,
                       const fly::storage stype = FLY_STORAGE_CSR);
    /**
       This function converts host or device arrays of values, row indices and
       column indices into a sparse array on the device.

       \note The rules for deep copy/shallow copy/reference are the same as for
             creating a regular \ref fly::array.

       \param[in] nRows is the number of rows in the dense matrix
       \param[in] nCols is the number of columns in the dense matrix
       \param[in] nNZ is the number of non zero elements in the dense matrix
       \param[in] values is the host array containing the non-zero elements
                  of the matrix
       \param[in] rowIdx is the row indices for the sparse array
       \param[in] colIdx is the column indices for the sparse array
       \param[in] type is the data type for the matrix
       \param[in] stype is the storage format of the sparse array
       \param[in] src is \ref flyHost if inputs are host arrays and \ref flyDevice
                  if the arrays are device arrays.
       \return \ref fly::array for the sparse array

       \snippet test/sparse.cpp ex_sparse_host_arrays

       \ingroup sparse_func_create
     */
    FLY_API array sparse(const dim_t nRows, const dim_t nCols, const dim_t nNZ,
                       const void* const values,
                       const int * const rowIdx, const int * const colIdx,
                       const dtype type = f32, const fly::storage stype = FLY_STORAGE_CSR,
                       const fly::source src = flyHost);

    /**
       This function converts a dense \ref fly::array into a sparse array.

       \param[in] dense is the source dense matrix
       \param[in] stype is the storage format of the sparse array
       \return \ref fly::array for the sparse array with the given storage type

       \snippet test/sparse.cpp ex_sparse_from_dense

       \ingroup sparse_func_create
     */
    FLY_API array sparse(const array dense, const fly::storage stype = FLY_STORAGE_CSR);

    /**
       \param[in] in is the source sparse matrix to be converted
       \param[in] destStrorage is the storage format of the output sparse array
       \return \ref fly::array for the sparse array with the given storage type

       \ingroup sparse_func_convert_to
     */
    FLY_API array sparseConvertTo(const array in, const fly::storage destStrorage);

    /**
       \param[in] sparse is the source sparse matrix
       \return dense \ref fly::array from sparse

       \snippet test/sparse.cpp ex_dense_from_sparse

       \ingroup sparse_func_dense
     */
    FLY_API array dense(const array sparse);


    /**
       \param[out] values stores the non-zero elements component of the sparse array
       \param[out] rowIdx stores the row indices component of the sparse array
       \param[out] colIdx stores the column indices component of the sparse array
       \param[out] stype stores the storage type of the sparse array
       \param[in] in is the input sparse matrix

       \ingroup sparse_func_info
     */
    FLY_API void sparseGetInfo(array &values, array &rowIdx, array &colIdx, fly::storage &stype,
                             const array in);

    /**
       \param[in] in is the input sparse matrix
       \return \ref fly::array for the non-zero elements component of the sparse array

       \ingroup sparse_func_values
     */
    FLY_API array sparseGetValues(const array in);

    /**
       \param[in] in is the input sparse matrix
       \return \ref fly::array for the row indices component of the sparse array

       \ingroup sparse_func_row_idx
     */
    FLY_API array sparseGetRowIdx(const array in);

    /**
       \param[in] in is the input sparse matrix
       \return \ref fly::array for the column indices component of the sparse array

       \ingroup sparse_func_col_idx
     */
    FLY_API array sparseGetColIdx(const array in);

    /**
       \param[in] in is the input sparse matrix
       \return the number of non-zero elements of the sparse array

       \ingroup sparse_func_nnz
     */
    FLY_API dim_t sparseGetNNZ(const array in);

    /**
       \param[in] in is the input sparse matrix
       \return \ref fly::storage for the storage type of the sparse array

       \ingroup sparse_func_storage
     */
    FLY_API fly::storage sparseGetStorage(const array in);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       This function converts \ref fly::array of values, row indices and column
       indices into a sparse array.

       \note This function only create references of these arrays into the
             sparse data structure and does not do deep copies.

       \param[out] out \ref fly::array for the sparse array
       \param[in] nRows is the number of rows in the dense matrix
       \param[in] nCols is the number of columns in the dense matrix
       \param[in] values is the \ref fly_array containing the non-zero elements
                  of the matrix
       \param[in] rowIdx is the row indices for the sparse array
       \param[in] colIdx is the column indices for the sparse array
       \param[in] stype is the storage format of the sparse array

       \return \ref FLY_SUCCESS if the execution completes properly

       \ingroup sparse_func_create
     */
    FLY_API fly_err fly_create_sparse_array(
                 fly_array *out,
                 const dim_t nRows, const dim_t nCols,
                 const fly_array values, const fly_array rowIdx, const fly_array colIdx,
                 const fly_storage stype);

    /**
       This function converts host or device arrays of values, row indices and
       column indices into a sparse array on the device.

       \note The rules for deep copy/shallow copy/reference are the same as for
             creating a regular \ref fly::array.

       \param[out] out \ref fly::array for the sparse array
       \param[in] nRows is the number of rows in the dense matrix
       \param[in] nCols is the number of columns in the dense matrix
       \param[in] nNZ is the number of non zero elements in the dense matrix
       \param[in] values is the host array containing the non-zero elements
                  of the matrix
       \param[in] rowIdx is the row indices for the sparse array
       \param[in] colIdx is the column indices for the sparse array
       \param[in] type is the data type for the matrix
       \param[in] stype is the storage format of the sparse array
       \param[in] src is \ref flyHost if inputs are host arrays and \ref flyDevice
                  if the arrays are device arrays.

       \return \ref FLY_SUCCESS if the execution completes properly

       \ingroup sparse_func_create
     */
    FLY_API fly_err fly_create_sparse_array_from_ptr(
                 fly_array *out,
                 const dim_t nRows, const dim_t nCols, const dim_t nNZ,
                 const void * const values,
                 const int * const rowIdx, const int * const colIdx,
                 const fly_dtype type, const fly_storage stype,
                 const fly_source src);

    /**
       This function converts a dense \ref fly_array into a sparse array.

       \param[out] out \ref fly_array for the sparse array with the given storage type
       \param[in] dense is the source dense matrix
       \param[in] stype is the storage format of the sparse array

       \return \ref FLY_SUCCESS if the execution completes properly

       \ingroup sparse_func_create
     */
    FLY_API fly_err fly_create_sparse_array_from_dense(
                 fly_array *out, const fly_array dense,
                 const fly_storage stype);

    /**
       \param[out] out \ref fly_array for the sparse array with the given storage type
       \param[in] in is the source sparse matrix to be converted
       \param[in] destStorage is the storage format of the output sparse array

       \return \ref FLY_SUCCESS if the execution completes properly

       \ingroup sparse_func_convert_to
     */
    FLY_API fly_err fly_sparse_convert_to(fly_array *out, const fly_array in,
                                      const fly_storage destStorage);

    /**
       \param[out] out dense \ref fly_array from sparse
       \param[in] sparse is the source sparse matrix

       \return \ref FLY_SUCCESS if the execution completes properly

       \ingroup sparse_func_dense
     */
    FLY_API fly_err fly_sparse_to_dense(fly_array *out, const fly_array sparse);

    /**
       \param[out] values stores the non-zero elements component of the sparse array
       \param[out] rowIdx stores the row indices component of the sparse array
       \param[out] colIdx stores the column indices component of the sparse array
       \param[out] stype stores the storage type of the sparse array
       \param[in] in is the input sparse matrix

       \return \ref FLY_SUCCESS if the execution completes properly

       \ingroup sparse_func_info
     */
    FLY_API fly_err fly_sparse_get_info(fly_array *values, fly_array *rowIdx, fly_array *colIdx, fly_storage *stype,
                                    const fly_array in);

    /**
       \param[out] out \ref fly_array for the non-zero elements component of the sparse array
       \param[in] in is the input sparse matrix

       \return \ref FLY_SUCCESS if the execution completes properly

       \ingroup sparse_func_values
     */
    FLY_API fly_err fly_sparse_get_values(fly_array *out, const fly_array in);

    /**
       \param[out] out \ref fly_array for the row indices component of the sparse array
       \param[in] in is the input sparse matrix

       \return \ref FLY_SUCCESS if the execution completes properly

       \ingroup sparse_func_row_idx
     */
    FLY_API fly_err fly_sparse_get_row_idx(fly_array *out, const fly_array in);

    /**
       \param[out] out \ref fly_array for the column indices component of the sparse array
       \param[in] in is the input sparse matrix

       \return \ref FLY_SUCCESS if the execution completes properly

       \ingroup sparse_func_col_idx
     */
    FLY_API fly_err fly_sparse_get_col_idx(fly_array *out, const fly_array in);

    /**
       \param[out] out the number of non-zero elements of the sparse array
       \param[in] in is the input sparse matrix

       \return \ref FLY_SUCCESS if the execution completes properly

       \ingroup sparse_func_nnz
     */
    FLY_API fly_err fly_sparse_get_nnz(dim_t *out, const fly_array in);

    /**
       \param[out] out contains \ref fly_storage for the storage type of the sparse array
       \param[in] in is the input sparse matrix

       \return \ref FLY_SUCCESS if the execution completes properly

       \ingroup sparse_func_storage
     */
    FLY_API fly_err fly_sparse_get_storage(fly_storage *out, const fly_array in);

#ifdef __cplusplus
}
#endif
