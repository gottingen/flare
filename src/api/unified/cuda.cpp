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
