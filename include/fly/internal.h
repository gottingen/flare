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
#include <fly/dim4.hpp>

#ifdef __cplusplus
namespace fly
{
    class array;

    /**
       \param[in] data is the raw data pointer.
       \param[in] offset specifies the number of elements to skip.
       \param[in] dims specifies the dimensions for the region of interest.
       \param[in] strides specifies the distance between each element of a given dimension.
       \param[in] ty specifies the data type of \p data.
       \param[in] location specifies if the data is on host or the device.

       \note: If \p location is `flyHost`, a memory copy is performed.

       \returns an fly::array() with specified offset, dimensions and strides.

       \ingroup internal_func_create
    */
    FLY_API array createStridedArray(const void *data, const dim_t offset,
                                   const dim4 dims, const dim4 strides,
                                   const fly::dtype ty,
                                   const fly::source location);

    /**
       \param[in] in An multi dimensional array.
       \returns fly::dim4() containing distance between consecutive elements in each dimension.

       \ingroup internal_func_strides
    */
    FLY_API dim4 getStrides(const array &in);

    /**
       \param[in] in An multi dimensional array.
       \returns offset from the starting location of data pointer specified in number of elements.

       \ingroup internal_func_offset
    */
    FLY_API dim_t getOffset(const array &in);


    /**
       \param[in] in An multi dimensional array.
       \returns Returns the raw pointer location to the array.

       \note This pointer may be shared with other arrays. Use this function with caution.

       \ingroup internal_func_rawptr
    */
    FLY_API void *getRawPtr(const array &in);

    /**
       \param[in] in An multi dimensional array.
       \returns a boolean specifying if all elements in the array are contiguous.

       \ingroup internal_func_linear
    */
    FLY_API bool isLinear(const array &in);

    /**
       \param[in] in An multi dimensional array.
       \returns a boolean specifying if the array owns the raw pointer. It is false if it is a sub array.

       \ingroup internal_func_owner
    */
    FLY_API bool isOwner(const array &in);
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /**
       \param[out] arr an fly_array with specified offset, dimensions and strides.
       \param[in] data is the raw data pointer.
       \param[in] offset specifies the number of elements to skip.
       \param[in] ndims specifies the number of array dimensions.
       \param[in] dims specifies the dimensions for the region of interest.
       \param[in] strides specifies the distance between each element of a given dimension.
       \param[in] ty specifies the data type of \p data.
       \param[in] location specifies if the data is on host or the device.

       \note If \p location is `flyHost`, a memory copy is performed.

       \ingroup internal_func_create
    */
    FLY_API fly_err fly_create_strided_array(fly_array *arr,
                                         const void *data,
                                         const dim_t offset,
                                         const unsigned ndims,
                                         const dim_t *const dims,
                                         const dim_t *const strides,
                                         const fly_dtype ty,
                                         const fly_source location);

    /**
       \param[in] arr An multi dimensional array.
       \param[out] s0 distance between each consecutive element along first  dimension.
       \param[out] s1 distance between each consecutive element along second dimension.
       \param[out] s2 distance between each consecutive element along third  dimension.
       \param[out] s3 distance between each consecutive element along fourth dimension.

       \ingroup internal_func_strides
    */
    FLY_API fly_err fly_get_strides(dim_t *s0, dim_t *s1, dim_t *s2, dim_t *s3, const fly_array arr);

    /**
       \param[in] arr An multi dimensional array.
       \param[out] offset: Offset from the starting location of data pointer specified in number of elements. distance between each consecutive element along first  dimension.

       \ingroup internal_func_offset
    */
    FLY_API fly_err fly_get_offset(dim_t *offset, const fly_array arr);

    /**
       \param[in] arr An multi dimensional array.
       \param[out] ptr the raw pointer location to the array.

       \note This pointer may be shared with other arrays. Use this function with caution.

       \ingroup internal_func_rawptr
    */
    FLY_API fly_err fly_get_raw_ptr(void **ptr, const fly_array arr);

    /**
       \param[in] arr An multi dimensional array.
       \param[out] result: a boolean specifying if all elements in the array are contiguous.

       \ingroup internal_func_linear
    */
    FLY_API fly_err fly_is_linear(bool *result, const fly_array arr);

    /**
       \param[in] arr An multi dimensional array.
       \param[out] result: a boolean specifying if the array owns the raw pointer. It is false if it is a sub array.

       \ingroup internal_func_owner
    */
    FLY_API fly_err fly_is_owner(bool *result, const fly_array arr);

    /**
       \param[out] bytes the size of the physical allocated bytes. This will return the size
       of the parent/owner if the \p arr is an indexed array.
       \param[in] arr the input array.

       \ingroup internal_func_allocatedbytes
    */
    FLY_API fly_err fly_get_allocated_bytes(size_t *bytes, const fly_array arr);

#ifdef __cplusplus
}
#endif
