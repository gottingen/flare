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

#include <common/deprecated.hpp>
#include <fly/array.h>
#include <fly/backend.h>
#include <fly/device.h>
#include "symbol_manager.hpp"

fly_err fly_set_backend(const fly_backend bknd) {
    return flare::unified::setBackend(bknd);
}

fly_err fly_get_backend_count(unsigned *num_backends) {
    *num_backends =
        flare::unified::FlySymbolManager::getInstance().getBackendCount();
    return FLY_SUCCESS;
}

fly_err fly_get_available_backends(int *result) {
    *result = flare::unified::FlySymbolManager::getInstance()
                  .getAvailableBackends();
    return FLY_SUCCESS;
}

fly_err fly_get_backend_id(fly_backend *result, const fly_array in) {
    // DO NOT CALL CHECK_ARRAYS HERE.
    // IT WILL RESULT IN AN INFINITE RECURSION
    CALL(fly_get_backend_id, result, in);
}

fly_err fly_get_device_id(int *device, const fly_array in) {
    CHECK_ARRAYS(in);
    CALL(fly_get_device_id, device, in);
}

fly_err fly_get_active_backend(fly_backend *result) {
    *result = flare::unified::getActiveBackend();
    return FLY_SUCCESS;
}

fly_err fly_info() { CALL_NO_PARAMS(fly_info); }

fly_err fly_init() { CALL_NO_PARAMS(fly_init); }

fly_err fly_info_string(char **str, const bool verbose) {
    CALL(fly_info_string, str, verbose);
}

fly_err fly_device_info(char *d_name, char *d_platform, char *d_toolkit,
                      char *d_compute) {
    CALL(fly_device_info, d_name, d_platform, d_toolkit, d_compute);
}

fly_err fly_get_device_count(int *num_of_devices) {
    CALL(fly_get_device_count, num_of_devices);
}

fly_err fly_get_dbl_support(bool *available, const int device) {
    CALL(fly_get_dbl_support, available, device);
}

fly_err fly_get_half_support(bool *available, const int device) {
    CALL(fly_get_half_support, available, device);
}

fly_err fly_set_device(const int device) { CALL(fly_set_device, device); }

fly_err fly_get_device(int *device) { CALL(fly_get_device, device); }

fly_err fly_sync(const int device) { CALL(fly_sync, device); }

fly_err fly_alloc_device(void **ptr, const dim_t bytes) {
    CALL(fly_alloc_device, ptr, bytes);
}


fly_err fly_alloc_pinned(void **ptr, const dim_t bytes) {
    CALL(fly_alloc_pinned, ptr, bytes);
}

fly_err fly_free_device(void *ptr) {
    CALL(fly_free_device, ptr);
}

fly_err fly_free_pinned(void *ptr) { CALL(fly_free_pinned, ptr); }

fly_err fly_alloc_host(void **ptr, const dim_t bytes) {
    *ptr = malloc(bytes);  // NOLINT(hicpp-no-malloc)
    return (*ptr == NULL) ? FLY_ERR_NO_MEM : FLY_SUCCESS;
}

fly_err fly_free_host(void *ptr) {
    free(ptr);  // NOLINT(hicpp-no-malloc)
    return FLY_SUCCESS;
}

fly_err fly_device_array(fly_array *arr, void *data, const unsigned ndims,
                       const dim_t *const dims, const fly_dtype type) {
    CALL(fly_device_array, arr, data, ndims, dims, type);
}

fly_err fly_device_mem_info(size_t *alloc_bytes, size_t *alloc_buffers,
                          size_t *lock_bytes, size_t *lock_buffers) {
    CALL(fly_device_mem_info, alloc_bytes, alloc_buffers, lock_bytes,
         lock_buffers);
}

fly_err fly_print_mem_info(const char *msg, const int device_id) {
    CALL(fly_print_mem_info, msg, device_id);
}

fly_err fly_device_gc() { CALL_NO_PARAMS(fly_device_gc); }

fly_err fly_set_mem_step_size(const size_t step_bytes) {
    CALL(fly_set_mem_step_size, step_bytes);
}

fly_err fly_get_mem_step_size(size_t *step_bytes) {
    CALL(fly_get_mem_step_size, step_bytes);
}

fly_err fly_lock_device_ptr(const fly_array arr) {
    CHECK_ARRAYS(arr);
    FLY_DEPRECATED_WARNINGS_OFF
    CALL(fly_lock_device_ptr, arr);
    FLY_DEPRECATED_WARNINGS_ON
}

fly_err fly_unlock_device_ptr(const fly_array arr) {
    CHECK_ARRAYS(arr);
    FLY_DEPRECATED_WARNINGS_OFF
    CALL(fly_unlock_device_ptr, arr);
    FLY_DEPRECATED_WARNINGS_ON
}

fly_err fly_lock_array(const fly_array arr) {
    CHECK_ARRAYS(arr);
    CALL(fly_lock_array, arr);
}

fly_err fly_unlock_array(const fly_array arr) {
    CHECK_ARRAYS(arr);
    CALL(fly_unlock_array, arr);
}

fly_err fly_is_locked_array(bool *res, const fly_array arr) {
    CHECK_ARRAYS(arr);
    CALL(fly_is_locked_array, res, arr);
}

fly_err fly_get_device_ptr(void **ptr, const fly_array arr) {
    CHECK_ARRAYS(arr);
    CALL(fly_get_device_ptr, ptr, arr);
}

fly_err fly_eval_multiple(const int num, fly_array *arrays) {
    for (int i = 0; i < num; i++) { CHECK_ARRAYS(arrays[i]); }
    CALL(fly_eval_multiple, num, arrays);
}

fly_err fly_set_manual_eval_flag(bool flag) {
    CALL(fly_set_manual_eval_flag, flag);
}

fly_err fly_get_manual_eval_flag(bool *flag) {
    CALL(fly_get_manual_eval_flag, flag);
}

fly_err fly_set_kernel_cache_directory(const char *path, int override_eval) {
    CALL(fly_set_kernel_cache_directory, path, override_eval);
}

fly_err fly_get_kernel_cache_directory(size_t *length, char *path) {
    CALL(fly_get_kernel_cache_directory, length, path);
}
