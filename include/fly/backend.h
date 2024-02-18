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

#ifdef __cplusplus
extern "C" {
#endif

#if FLY_API_VERSION >= 32
/**
   \param[in] bknd takes one of the values of enum \ref fly_backend
   \returns \ref fly_err error code

   \ingroup unified_func_setbackend
 */
FLY_API fly_err fly_set_backend(const fly_backend bknd);
#endif

#if FLY_API_VERSION >= 32
/**
   \param[out] num_backends Number of available backends
   \returns \ref fly_err error code

   \ingroup unified_func_getbackendcount
 */
FLY_API fly_err fly_get_backend_count(unsigned* num_backends);
#endif

#if FLY_API_VERSION >= 32
/**
   Returns a flag of all available backends

   \code{.cpp}
   int backends = 0;
   fly_get_available_backends(&backends);

   if(backends & FLY_BACKEND_CUDA) {
       // The CUDA backend is available
   }
   \endcode

   \param[out] backends A flag of all available backends. Use the &(and)
   operator to check if a particular backend is available

   \returns \ref fly_err error code

   \ingroup unified_func_getavailbackends
 */
FLY_API fly_err fly_get_available_backends(int* backends);
#endif

#if FLY_API_VERSION >= 32
/**
   \param[out] backend takes one of the values of enum \ref fly_backend
   \param[in] in is the array who's backend is to be queried
   \returns \ref fly_err error code

   \ingroup unified_func_getbackendid
 */
FLY_API fly_err fly_get_backend_id(fly_backend *backend, const fly_array in);
#endif

#if FLY_API_VERSION >= 33
/**
   \param[out] backend takes one of the values of enum \ref fly_backend
   from the backend that is currently set to active
   \returns \ref fly_err error code

   \ingroup unified_func_getactivebackend
 */
FLY_API fly_err fly_get_active_backend(fly_backend *backend);
#endif

#if FLY_API_VERSION >= 33
/**
   \param[out] device contains the device on which \p in was created.
   \param[in] in is the array who's device is to be queried.
   \returns \ref fly_err error code

   \ingroup unified_func_getdeviceid
 */
FLY_API fly_err fly_get_device_id(int *device, const fly_array in);
#endif


#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace fly
{
class array;

#if FLY_API_VERSION >= 32
/**
   \param[in] bknd takes one of the values of enum \ref fly_backend

   \ingroup unified_func_setbackend
 */
FLY_API void setBackend(const Backend bknd);
#endif

#if FLY_API_VERSION >= 32
/**
   \returns Number of available backends

   \ingroup unified_func_getbackendcount
 */
FLY_API unsigned getBackendCount();
#endif

#if FLY_API_VERSION >= 32
/**
   Returns a flag of all available backends

   \code{.cpp}
   int backends = getAvailableBackends();

   if(backends & FLY_BACKEND_CUDA) {
   // The CUDA backend is available
   }
   \endcode

   \returns A flag of available backends

   \ingroup unified_func_getavailbackends
 */
FLY_API int getAvailableBackends();
#endif

#if FLY_API_VERSION >= 32
/**
   \param[in] in is the array who's backend is to be queried
   \returns \ref fly_backend which is the backend on which the array is created

   \ingroup unified_func_getbackendid
 */
FLY_API fly::Backend getBackendId(const array &in);
#endif

#if FLY_API_VERSION >= 33
/**
   \returns \ref fly_backend which is the backend is currently active

   \ingroup unified_func_getctivebackend
 */
FLY_API fly::Backend getActiveBackend();
#endif

#if FLY_API_VERSION >= 33
/**
   \param[in] in is the array who's device is to be queried.
   \returns The id of the device on which this array was created.

   \note Device ID can be the same for arrays belonging to different backends.

   \ingroup unified_func_getdeviceid
 */
FLY_API int getDeviceId(const array &in);
#endif

}
#endif
