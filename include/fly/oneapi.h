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

typedef enum
{
    //TODO: update? are these relevant in sycl
    FLY_ONEAPI_PLATFORM_AMD     = 0,
    FLY_ONEAPI_PLATFORM_APPLE   = 1,
    FLY_ONEAPI_PLATFORM_INTEL   = 2,
    FLY_ONEAPI_PLATFORM_NVIDIA  = 3,
    FLY_ONEAPI_PLATFORM_BEIGNET = 4,
    FLY_ONEAPI_PLATFORM_POCL    = 5,
    FLY_ONEAPI_PLATFORM_UNKNOWN = -1
} fly_oneapi_platform;

#if 0
/**
    \ingroup opencl_mat
    @{
*/
/**
  Get a handle to Flare's OpenCL context

  \param[out] ctx the current context being used by Flare
  \param[in] retain if true calls clRetainContext prior to returning the context
  \returns \ref fly_err error code

  \note Set \p retain to true if this value will be passed to a cl::Context constructor
*/
FLY_API fly_err flycl_get_context(cl_context *ctx, const bool retain);

/**
  Get a handle to Flare's OpenCL command queue

  \param[out] queue the current command queue being used by Flare
  \param[in] retain if true calls clRetainCommandQueue prior to returning the context
  \returns \ref fly_err error code

  \note Set \p retain to true if this value will be passed to a cl::CommandQueue constructor
*/
FLY_API fly_err flycl_get_queue(cl_command_queue *queue, const bool retain);

/**
   Get the device ID for Flare's current active device

   \param[out] id the cl_device_id of the current device
   \returns \ref fly_err error code
*/
FLY_API fly_err flycl_get_device_id(cl_device_id *id);

/**
   Set Flare's active device based on \p id of type cl_device_id

   \param[in] id the cl_device_id of the device to be set as active device
   \returns \ref fly_err error code
*/
FLY_API fly_err flycl_set_device_id(cl_device_id id);

/**
   Push user provided device control constructs into the Flare device manager pool

   This function should be used only when the user would like Flare to use an
   user generated OpenCL context and related objects for Flare operations.

   \param[in] dev is the OpenCL device for which user provided context will be used by Flare
   \param[in] ctx is the user provided OpenCL cl_context to be used by Flare
   \param[in] que is the user provided OpenCL cl_command_queue to be used by Flare. If this
                  parameter is NULL, then we create a command queue for the user using the OpenCL
                  context they provided us.

   \note Flare does not take control of releasing the objects passed to it. The user needs to release them appropriately.
*/
FLY_API fly_err flycl_add_device_context(cl_device_id dev, cl_context ctx, cl_command_queue que);

/**
   Set active device using cl_context and cl_device_id

   \param[in] dev is the OpenCL device id that is to be set as Active device inside Flare
   \param[in] ctx is the OpenCL cl_context being used by Flare
*/
FLY_API fly_err flycl_set_device_context(cl_device_id dev, cl_context ctx);

/**
   Remove the user provided device control constructs from the Flare device manager pool

   This function should be used only when the user would like Flare to remove an already
   pushed user generated OpenCL context and related objects.

   \param[in] dev is the OpenCL device id that has to be popped
   \param[in] ctx is the cl_context object to be removed from Flare pool

   \note Flare does not take control of releasing the objects passed to it. The user needs to release them appropriately.
*/
FLY_API fly_err flycl_delete_device_context(cl_device_id dev, cl_context ctx);

/*
 Get the type of the current device
*/
FLY_API fly_err flycl_get_device_type(flycl_device_type *res);

/**
   Get the platform of the current device
*/
FLY_API fly_err flycl_get_platform(flycl_platform *res);

/**
  @}
*/
#endif //if 0 comment

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <fly/array.h>
#include <fly/dim4.hpp>
#include <fly/exception.h>
#include <fly/device.h>
#include <stdio.h>

namespace flyoneapi
{

#if 0
 /**
     \addtogroup opencl_mat
     @{
 */

 /**
 Get a handle to Flare's OpenCL context

 \param[in] retain if true calls clRetainContext prior to returning the context
 \returns the current context being used by Flare

 \note Set \p retain to true if this value will be passed to a cl::Context constructor
 */
 static inline cl_context getContext(bool retain = false)
 {
     cl_context ctx;
     fly_err err = flycl_get_context(&ctx, retain);
     if (err != FLY_SUCCESS) throw fly::exception("Failed to get OpenCL context from flare");
     return ctx;
 }

 /**
 Get a handle to Flare's OpenCL command queue

 \param[in] retain if true calls clRetainCommandQueue prior to returning the context
 \returns the current command queue being used by Flare

 \note Set \p retain to true if this value will be passed to a cl::CommandQueue constructor
 */
 static inline cl_command_queue getQueue(bool retain = false)
 {
     cl_command_queue queue;
     fly_err err = flycl_get_queue(&queue, retain);
     if (err != FLY_SUCCESS) throw fly::exception("Failed to get OpenCL command queue from flare");
     return queue;
 }

 /**
    Get the device ID for Flare's current active device
    \returns the cl_device_id of the current device
 */
 static inline cl_device_id getDeviceId()
 {
     cl_device_id id;
     fly_err err = flycl_get_device_id(&id);
     if (err != FLY_SUCCESS) throw fly::exception("Failed to get OpenCL device ID");

     return id;
 }

 /**
   Set Flare's active device based on \p id of type cl_device_id

   \param[in] id the cl_device_id of the device to be set as active device
 */
 static inline void setDeviceId(cl_device_id id)
 {
     fly_err err = flycl_set_device_id(id);
     if (err != FLY_SUCCESS) throw fly::exception("Failed to set OpenCL device as active device");
 }

/**
   Push user provided device control constructs into the Flare device manager pool

   This function should be used only when the user would like Flare to use an
   user generated OpenCL context and related objects for Flare operations.

   \param[in] dev is the OpenCL device for which user provided context will be used by Flare
   \param[in] ctx is the user provided OpenCL cl_context to be used by Flare
   \param[in] que is the user provided OpenCL cl_command_queue to be used by Flare. If this
                  parameter is NULL, then we create a command queue for the user using the OpenCL
                  context they provided us.

   \note Flare does not take control of releasing the objects passed to it. The user needs to release them appropriately.
*/
static inline void addDevice(cl_device_id dev, cl_context ctx, cl_command_queue que)
{
    fly_err err = flycl_add_device_context(dev, ctx, que);
    if (err!=FLY_SUCCESS) throw fly::exception("Failed to push user provided device/context to Flare pool");
}

/**
   Set active device using cl_context and cl_device_id

   \param[in] dev is the OpenCL device id that is to be set as Active device inside Flare
   \param[in] ctx is the OpenCL cl_context being used by Flare
*/
static inline void setDevice(cl_device_id dev, cl_context ctx)
{
    fly_err err = flycl_set_device_context(dev, ctx);
    if (err!=FLY_SUCCESS) throw fly::exception("Failed to set device based on cl_device_id & cl_context");
}

/**
   Remove the user provided device control constructs from the Flare device manager pool

   This function should be used only when the user would like Flare to remove an already
   pushed user generated OpenCL context and related objects.

   \param[in] dev is the OpenCL device id that has to be popped
   \param[in] ctx is the cl_context object to be removed from Flare pool

   \note Flare does not take control of releasing the objects passed to it. The user needs to release them appropriately.
*/
static inline void deleteDevice(cl_device_id dev, cl_context ctx)
{
    fly_err err = flycl_delete_device_context(dev, ctx);
    if (err!=FLY_SUCCESS) throw fly::exception("Failed to remove the requested device from Flare device pool");
}
#endif


 typedef flycl_device_type deviceType;
 typedef flycl_platform platform;

/**
   Get the type of the current device
*/
static inline deviceType getDeviceType()
{
    flycl_device_type res = FLYCL_DEVICE_TYPE_UNKNOWN;
    fly_err err = flycl_get_device_type(&res);
    if (err!=FLY_SUCCESS) throw fly::exception("Failed to get OpenCL device type");
    return res;
}

/**
   Get a vendor enumeration for the current platform
*/
static inline platform getPlatform()
{
    flycl_platform res = FLYCL_PLATFORM_UNKNOWN;
    fly_err err = flycl_get_platform(&res);
    if (err!=FLY_SUCCESS) throw fly::exception("Failed to get OpenCL platform");
    return res;
}

 /**
 Create an fly::array object from an OpenCL cl_mem buffer

 \param[in] idims the dimensions of the buffer
 \param[in] buf the OpenCL memory object
 \param[in] type the data type contained in the buffer
 \param[in] retain if true, instructs Flare to retain the memory object
 \returns an array object created from the OpenCL buffer

 \note Set \p retain to true if the memory originates from a cl::Buffer object
  */
 static inline fly::array array(fly::dim4 idims, cl_mem buf, fly::dtype type, bool retain=false)
 {
     const unsigned ndims = (unsigned)idims.ndims();
     const dim_t *dims = idims.get();

     cl_context context;
     cl_int clerr = clGetMemObjectInfo(buf, CL_MEM_CONTEXT, sizeof(cl_context), &context, NULL);
     if (clerr != CL_SUCCESS) {
         throw fly::exception("Failed to get context from cl_mem object \"buf\" ");
     }

     if (context != getContext()) {
         throw(fly::exception("Context mismatch between input \"buf\" and flare"));
     }


     if (retain) clerr = clRetainMemObject(buf);

     fly_array out;
     fly_err err = fly_device_array(&out, buf, ndims, dims, type);

     if (err != FLY_SUCCESS || clerr != CL_SUCCESS) {
         if (retain && clerr == CL_SUCCESS) clReleaseMemObject(buf);
         throw fly::exception("Failed to create device array");
     }

     return fly::array(out);
 }

 /**
 Create an fly::array object from an OpenCL cl_mem buffer

 \param[in] dim0 the length of the first dimension of the buffer
 \param[in] buf the OpenCL memory object
 \param[in] type the data type contained in the buffer
 \param[in] retain if true, instructs Flare to retain the memory object
 \returns an array object created from the OpenCL buffer

 \note Set \p retain to true if the memory originates from a cl::Buffer object
  */
 static inline fly::array array(dim_t dim0,
                               cl_mem buf, fly::dtype type, bool retain=false)
 {
     return flycl::array(fly::dim4(dim0), buf, type, retain);
 }

 /**
 Create an fly::array object from an OpenCL cl_mem buffer

 \param[in] dim0 the length of the first dimension of the buffer
 \param[in] dim1 the length of the second dimension of the buffer
 \param[in] buf the OpenCL memory object
 \param[in] type the data type contained in the buffer
 \param[in] retain if true, instructs Flare to retain the memory object
 \returns an array object created from the OpenCL buffer

 \note Set \p retain to true if the memory originates from a cl::Buffer object
  */
 static inline fly::array array(dim_t dim0, dim_t dim1,
                               cl_mem buf, fly::dtype type, bool retain=false)
 {
     return flycl::array(fly::dim4(dim0, dim1), buf, type, retain);
 }

 /**
 Create an fly::array object from an OpenCL cl_mem buffer

 \param[in] dim0 the length of the first dimension of the buffer
 \param[in] dim1 the length of the second dimension of the buffer
 \param[in] dim2 the length of the third dimension of the buffer
 \param[in] buf the OpenCL memory object
 \param[in] type the data type contained in the buffer
 \param[in] retain if true, instructs Flare to retain the memory object
 \returns an array object created from the OpenCL buffer

 \note Set \p retain to true if the memory originates from a cl::Buffer object
  */
 static inline fly::array array(dim_t dim0, dim_t dim1,
                               dim_t dim2,
                               cl_mem buf, fly::dtype type, bool retain=false)
 {
     return flycl::array(fly::dim4(dim0, dim1, dim2), buf, type, retain);
 }

 /**
 Create an fly::array object from an OpenCL cl_mem buffer

 \param[in] dim0 the length of the first dimension of the buffer
 \param[in] dim1 the length of the second dimension of the buffer
 \param[in] dim2 the length of the third dimension of the buffer
 \param[in] dim3 the length of the fourth dimension of the buffer
 \param[in] buf the OpenCL memory object
 \param[in] type the data type contained in the buffer
 \param[in] retain if true, instructs Flare to retain the memory object
 \returns an array object created from the OpenCL buffer

 \note Set \p retain to true if the memory originates from a cl::Buffer object
  */
 static inline fly::array array(dim_t dim0, dim_t dim1,
                               dim_t dim2, dim_t dim3,
                               cl_mem buf, fly::dtype type, bool retain=false)
 {
     return flycl::array(fly::dim4(dim0, dim1, dim2, dim3), buf, type, retain);
 }

/**
   @}
*/
#endif //#IF 0 tmp comment

}


#endif
