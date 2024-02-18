/*******************************************************
 * Copyright (c) 2020, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/backend.h>
#include "symbol_manager.hpp"

#include <fly/opencl.h>

fly_err flycl_get_device_type(flycl_device_type* res) {
    fly_backend backend;
    fly_get_active_backend(&backend);
    if (backend == FLY_BACKEND_OPENCL) { CALL(flycl_get_device_type, res); }
    return FLY_ERR_NOT_SUPPORTED;
}

fly_err flycl_get_platform(flycl_platform* res) {
    fly_backend backend;
    fly_get_active_backend(&backend);
    if (backend == FLY_BACKEND_OPENCL) { CALL(flycl_get_platform, res); }
    return FLY_ERR_NOT_SUPPORTED;
}

fly_err flycl_get_context(cl_context* ctx, const bool retain) {
    fly_backend backend;
    fly_get_active_backend(&backend);
    if (backend == FLY_BACKEND_OPENCL) { CALL(flycl_get_context, ctx, retain); }
    return FLY_ERR_NOT_SUPPORTED;
}

fly_err flycl_get_queue(cl_command_queue* queue, const bool retain) {
    fly_backend backend;
    fly_get_active_backend(&backend);
    if (backend == FLY_BACKEND_OPENCL) { CALL(flycl_get_queue, queue, retain); }
    return FLY_ERR_NOT_SUPPORTED;
}

fly_err flycl_get_device_id(cl_device_id* id) {
    fly_backend backend;
    fly_get_active_backend(&backend);
    if (backend == FLY_BACKEND_OPENCL) { CALL(flycl_get_device_id, id); }
    return FLY_ERR_NOT_SUPPORTED;
}

fly_err flycl_set_device_id(cl_device_id id) {
    fly_backend backend;
    fly_get_active_backend(&backend);
    if (backend == FLY_BACKEND_OPENCL) { CALL(flycl_set_device_id, id); }
    return FLY_ERR_NOT_SUPPORTED;
}

fly_err flycl_add_device_context(cl_device_id dev, cl_context ctx,
                               cl_command_queue que) {
    fly_backend backend;
    fly_get_active_backend(&backend);
    if (backend == FLY_BACKEND_OPENCL) {
        CALL(flycl_add_device_context, dev, ctx, que);
    }
    return FLY_ERR_NOT_SUPPORTED;
}

fly_err flycl_set_device_context(cl_device_id dev, cl_context ctx) {
    fly_backend backend;
    fly_get_active_backend(&backend);
    if (backend == FLY_BACKEND_OPENCL) {
        CALL(flycl_set_device_context, dev, ctx);
    }
    return FLY_ERR_NOT_SUPPORTED;
}

fly_err flycl_delete_device_context(cl_device_id dev, cl_context ctx) {
    fly_backend backend;
    fly_get_active_backend(&backend);
    if (backend == FLY_BACKEND_OPENCL) {
        CALL(flycl_delete_device_context, dev, ctx);
    }
    return FLY_ERR_NOT_SUPPORTED;
}
