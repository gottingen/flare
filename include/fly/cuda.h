/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <fly/defines.h>
#include <fly/exception.h>

/// This file contain functions that apply only to the CUDA backend. It will
/// include cuda headers when it is built with NVCC. Otherwise the you can
/// define the FLY_DEFINE_CUDA_TYPES before including this file and it will
/// define the cuda types used in this header.

#ifdef __NVCC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#else
#ifdef FLY_DEFINE_CUDA_TYPES
typedef struct CUstream_st *cudaStream_t;

/*Enum for default math mode/tensor operation*/
typedef enum {
  CUBLAS_DEFAULT_MATH = 0,
  CUBLAS_TENSOR_OP_MATH = 1
} cublasMath_t;
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif


#if FLY_API_VERSION >= 31
/**
   Get the stream for the CUDA device with \p id in Flare context

   \param[out] stream CUDA Stream of device with \p id in Flare context
   \param[in] id Flare device id
   \returns \ref fly_err error code

   \ingroup cuda_mat
 */
FLY_API fly_err flycu_get_stream(cudaStream_t* stream, int id);
#endif

#if FLY_API_VERSION >= 31
/**
   Get the native device id of the CUDA device with \p id in Flare context

   \param[out] nativeid native device id of the CUDA device with \p id in Flare context
   \param[in] id Flare device id
   \returns \ref fly_err error code

   \ingroup cuda_mat
 */
FLY_API fly_err flycu_get_native_id(int* nativeid, int id);
#endif

#if FLY_API_VERSION >= 32
/**
   Set the CUDA device with given native id as the active device for Flare

   \param[in] nativeid native device id of the CUDA device
   \returns \ref fly_err error code

   \ingroup cuda_mat
 */
FLY_API fly_err flycu_set_native_id(int nativeid);
#endif

#if FLY_API_VERSION >= 37
/**
    Sets the cuBLAS math mode for the internal handle

    See the cuBLAS documentation for additional details

    \param[in] mode The cublasMath_t type to set
    \returns \ref fly_err error code

    \ingroup cuda_mat
*/
FLY_API fly_err flycu_cublasSetMathMode(cublasMath_t mode);
#endif

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

namespace flycu
{

#if FLY_API_VERSION >= 31
/**
   Get the stream for the CUDA device with \p id in Flare context

   \param[in] id Flare device id
   \returns cuda stream used by CUDA device

   \ingroup cuda_mat
 */
static inline cudaStream_t getStream(int id)
{
    cudaStream_t retVal;
    fly_err err = flycu_get_stream(&retVal, id);
    if (err!=FLY_SUCCESS)
        throw fly::exception("Failed to get CUDA stream from Flare");
    return retVal;
}
#endif

#if FLY_API_VERSION >= 31
/**
   Get the native device id of the CUDA device with \p id in Flare context

   \param[in] id Flare device id
   \returns cuda native id of device

   \ingroup cuda_mat
 */
static inline int getNativeId(int id)
{
    int retVal;
    fly_err err = flycu_get_native_id(&retVal, id);
    if (err!=FLY_SUCCESS)
        throw fly::exception("Failed to get CUDA device native id from Flare");
    return retVal;
}
#endif

#if FLY_API_VERSION >= 32
/**
   Set the CUDA device with given native id as the active device for Flare

   \param[in] nativeId native device id of the CUDA device

   \ingroup cuda_mat
 */
static inline void setNativeId(int nativeId)
{
    fly_err err = flycu_set_native_id(nativeId);
    if (err!=FLY_SUCCESS)
        throw fly::exception("Failed to change active CUDA device to the device with given native id");
}
#endif

}
#endif
