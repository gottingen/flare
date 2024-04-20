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
extern "C" {
#endif

/**
   \param[in] bknd takes one of the values of enum \ref fly_backend
   \returns \ref fly_err error code

   \ingroup unified_func_setbackend
 */
FLY_API fly_err fly_set_backend(const fly_backend bknd);

/**
   \param[out] num_backends Number of available backends
   \returns \ref fly_err error code

   \ingroup unified_func_getbackendcount
 */
FLY_API fly_err fly_get_backend_count(unsigned* num_backends);


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


/**
   \param[out] backend takes one of the values of enum \ref fly_backend
   \param[in] in is the array who's backend is to be queried
   \returns \ref fly_err error code

   \ingroup unified_func_getbackendid
 */
FLY_API fly_err fly_get_backend_id(fly_backend *backend, const fly_array in);

/**
   \param[out] backend takes one of the values of enum \ref fly_backend
   from the backend that is currently set to active
   \returns \ref fly_err error code

   \ingroup unified_func_getactivebackend
 */
FLY_API fly_err fly_get_active_backend(fly_backend *backend);

/**
   \param[out] device contains the device on which \p in was created.
   \param[in] in is the array who's device is to be queried.
   \returns \ref fly_err error code

   \ingroup unified_func_getdeviceid
 */
FLY_API fly_err fly_get_device_id(int *device, const fly_array in);


#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace fly
{
class array;

/**
   \param[in] bknd takes one of the values of enum \ref fly_backend

   \ingroup unified_func_setbackend
 */
FLY_API void setBackend(const Backend bknd);

/**
   \returns Number of available backends

   \ingroup unified_func_getbackendcount
 */
FLY_API unsigned getBackendCount();

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


/**
   \param[in] in is the array who's backend is to be queried
   \returns \ref fly_backend which is the backend on which the array is created

   \ingroup unified_func_getbackendid
 */
FLY_API fly::Backend getBackendId(const array &in);

/**
   \returns \ref fly_backend which is the backend is currently active

   \ingroup unified_func_getctivebackend
 */
FLY_API fly::Backend getActiveBackend();


/**
   \param[in] in is the array who's device is to be queried.
   \returns The id of the device on which this array was created.

   \note Device ID can be the same for arrays belonging to different backends.

   \ingroup unified_func_getdeviceid
 */
FLY_API int getDeviceId(const array &in);

}
#endif
