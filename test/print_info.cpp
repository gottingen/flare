/*******************************************************
 * Copyright (c) 2018, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <flare.h>

using namespace fly;

int main(int, const char**) {
    int backend = getAvailableBackends();
    if (backend & FLY_BACKEND_OPENCL) {
        setBackend(FLY_BACKEND_OPENCL);
    } else if (backend & FLY_BACKEND_CUDA) {
        setBackend(FLY_BACKEND_CUDA);
    } else if (backend & FLY_BACKEND_CPU) {
        setBackend(FLY_BACKEND_CPU);
    }

    info();
    return 0;
}
