/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/array.h>
#include <fly/opencl.h>
#include <cstring>

namespace fly {
template<>
FLY_API cl_mem *array::device() const {
    auto *mem_ptr = new cl_mem;
    void *dptr    = nullptr;
    fly_err err    = fly_get_device_ptr(&dptr, get());
    memcpy(mem_ptr, &dptr, sizeof(void *));
    if (err != FLY_SUCCESS) {
        throw fly::exception("Failed to get cl_mem from array object");
    }
    return mem_ptr;
}
}  // namespace fly
