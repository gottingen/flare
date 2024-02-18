/*******************************************************
 * Copyright (c) 2019, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fly/memory.h>
#include "symbol_manager.hpp"

fly_err fly_create_memory_manager(fly_memory_manager* out) {
    CALL(fly_create_memory_manager, out);
}

fly_err fly_release_memory_manager(fly_memory_manager handle) {
    CALL(fly_release_memory_manager, handle);
}

fly_err fly_set_memory_manager(fly_memory_manager handle) {
    CALL(fly_set_memory_manager, handle);
}

fly_err fly_set_memory_manager_pinned(fly_memory_manager handle) {
    CALL(fly_set_memory_manager_pinned, handle);
}

fly_err fly_unset_memory_manager() { CALL_NO_PARAMS(fly_unset_memory_manager); }

fly_err fly_unset_memory_manager_pinned() {
    CALL_NO_PARAMS(fly_unset_memory_manager_pinned);
}

fly_err fly_memory_manager_get_payload(fly_memory_manager handle, void** payload) {
    CALL(fly_memory_manager_get_payload, handle, payload);
}

fly_err fly_memory_manager_set_payload(fly_memory_manager handle, void* payload) {
    CALL(fly_memory_manager_set_payload, handle, payload);
}

fly_err fly_memory_manager_set_initialize_fn(fly_memory_manager handle,
                                           fly_memory_manager_initialize_fn fn) {
    CALL(fly_memory_manager_set_initialize_fn, handle, fn);
}

fly_err fly_memory_manager_set_shutdown_fn(fly_memory_manager handle,
                                         fly_memory_manager_shutdown_fn fn) {
    CALL(fly_memory_manager_set_shutdown_fn, handle, fn);
}

fly_err fly_memory_manager_set_alloc_fn(fly_memory_manager handle,
                                      fly_memory_manager_alloc_fn fn) {
    CALL(fly_memory_manager_set_alloc_fn, handle, fn);
}

fly_err fly_memory_manager_set_allocated_fn(fly_memory_manager handle,
                                          fly_memory_manager_allocated_fn fn) {
    CALL(fly_memory_manager_set_allocated_fn, handle, fn);
}

fly_err fly_memory_manager_set_unlock_fn(fly_memory_manager handle,
                                       fly_memory_manager_unlock_fn fn) {
    CALL(fly_memory_manager_set_unlock_fn, handle, fn);
}

fly_err fly_memory_manager_set_signal_memory_cleanup_fn(
    fly_memory_manager handle, fly_memory_manager_signal_memory_cleanup_fn fn) {
    CALL(fly_memory_manager_set_signal_memory_cleanup_fn, handle, fn);
}

fly_err fly_memory_manager_set_print_info_fn(fly_memory_manager handle,
                                           fly_memory_manager_print_info_fn fn) {
    CALL(fly_memory_manager_set_print_info_fn, handle, fn);
}

fly_err fly_memory_manager_set_user_lock_fn(fly_memory_manager handle,
                                          fly_memory_manager_user_lock_fn fn) {
    CALL(fly_memory_manager_set_user_lock_fn, handle, fn);
}

fly_err fly_memory_manager_set_user_unlock_fn(
    fly_memory_manager handle, fly_memory_manager_user_unlock_fn fn) {
    CALL(fly_memory_manager_set_user_unlock_fn, handle, fn);
}

fly_err fly_memory_manager_set_is_user_locked_fn(
    fly_memory_manager handle, fly_memory_manager_is_user_locked_fn fn) {
    CALL(fly_memory_manager_set_is_user_locked_fn, handle, fn);
}

fly_err fly_memory_manager_set_get_memory_pressure_fn(
    fly_memory_manager handle, fly_memory_manager_get_memory_pressure_fn fn) {
    CALL(fly_memory_manager_set_get_memory_pressure_fn, handle, fn);
}

fly_err fly_memory_manager_set_jit_tree_exceeds_memory_pressure_fn(
    fly_memory_manager handle,
    fly_memory_manager_jit_tree_exceeds_memory_pressure_fn fn) {
    CALL(fly_memory_manager_set_jit_tree_exceeds_memory_pressure_fn, handle, fn);
}

fly_err fly_memory_manager_set_add_memory_management_fn(
    fly_memory_manager handle, fly_memory_manager_add_memory_management_fn fn) {
    CALL(fly_memory_manager_set_add_memory_management_fn, handle, fn);
}

fly_err fly_memory_manager_set_remove_memory_management_fn(
    fly_memory_manager handle,
    fly_memory_manager_remove_memory_management_fn fn) {
    CALL(fly_memory_manager_set_remove_memory_management_fn, handle, fn);
}

fly_err fly_memory_manager_get_active_device_id(fly_memory_manager handle,
                                              int* id) {
    CALL(fly_memory_manager_get_active_device_id, handle, id);
}

fly_err fly_memory_manager_native_alloc(fly_memory_manager handle, void** ptr,
                                      size_t size) {
    CALL(fly_memory_manager_native_alloc, handle, ptr, size);
}

fly_err fly_memory_manager_native_free(fly_memory_manager handle, void* ptr) {
    CALL(fly_memory_manager_native_free, handle, ptr);
}

fly_err fly_memory_manager_get_max_memory_size(fly_memory_manager handle,
                                             size_t* size, int id) {
    CALL(fly_memory_manager_get_max_memory_size, handle, size, id);
}

fly_err fly_memory_manager_get_memory_pressure_threshold(fly_memory_manager handle,
                                                       float* value) {
    CALL(fly_memory_manager_get_memory_pressure_threshold, handle, value);
}

fly_err fly_memory_manager_set_memory_pressure_threshold(fly_memory_manager handle,
                                                       float value) {
    CALL(fly_memory_manager_set_memory_pressure_threshold, handle, value);
}
