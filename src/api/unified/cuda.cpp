/*******************************************************
 * Copyright (c) 2019, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/backend.h>
#include "symbol_manager.hpp"

#define FLY_DEFINE_CUDA_TYPES
#include <fly/cuda.h>

fly_err flycu_get_stream(cudaStream_t* stream, int id) {
    fly_backend backend;
    fly_get_active_backend(&backend);
    if (backend == FLY_BACKEND_CUDA) { CALL(flycu_get_stream, stream, id); }
    return FLY_ERR_NOT_SUPPORTED;
}

fly_err flycu_get_native_id(int* nativeid, int id) {
    fly_backend backend;
    fly_get_active_backend(&backend);
    if (backend == FLY_BACKEND_CUDA) { CALL(flycu_get_native_id, nativeid, id); }
    return FLY_ERR_NOT_SUPPORTED;
}

fly_err flycu_set_native_id(int nativeid) {
    fly_backend backend;
    fly_get_active_backend(&backend);
    if (backend == FLY_BACKEND_CUDA) { CALL(flycu_set_native_id, nativeid); }
    return FLY_ERR_NOT_SUPPORTED;
}

fly_err flycu_cublasSetMathMode(cublasMath_t mode) {
    fly_backend backend;
    fly_get_active_backend(&backend);
    if (backend == FLY_BACKEND_CUDA) { CALL(flycu_cublasSetMathMode, mode); }
    return FLY_ERR_NOT_SUPPORTED;
}
