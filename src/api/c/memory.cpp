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

#include <memoryapi.hpp>

#include <Array.hpp>
#include <backend.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <events.hpp>
#include <handle.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <fly/backend.h>
#include <fly/device.h>
#include <fly/dim4.hpp>
#include <fly/memory.h>
#include <fly/version.h>

#include <utility>

using fly::dim4;
using flare::common::half;
using detail::cdouble;
using detail::cfloat;
using detail::createDeviceDataArray;
using detail::deviceMemoryInfo;
using detail::getActiveDeviceId;
using detail::getDeviceCount;
using detail::intl;
using detail::isLocked;
using detail::memAllocUser;
using detail::memFreeUser;
using detail::memLock;
using detail::memUnlock;
using detail::pinnedAlloc;
using detail::pinnedFree;
using detail::printMemInfo;
using detail::signalMemoryCleanup;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;
using std::move;
using std::swap;

fly_err fly_device_array(fly_array *arr, void *data, const unsigned ndims,
                       const dim_t *const dims, const fly_dtype type) {
    try {
        FLY_CHECK(fly_init());

        fly_array res;

        DIM_ASSERT(1, ndims >= 1);
        dim4 d(1, 1, 1, 1);
        for (unsigned i = 0; i < ndims; i++) {
            d[i] = dims[i];
            DIM_ASSERT(3, dims[i] >= 1);
        }

        switch (type) {
            case f32:
                res = getHandle(createDeviceDataArray<float>(d, data));
                break;
            case f64:
                res = getHandle(createDeviceDataArray<double>(d, data));
                break;
            case c32:
                res = getHandle(createDeviceDataArray<cfloat>(d, data));
                break;
            case c64:
                res = getHandle(createDeviceDataArray<cdouble>(d, data));
                break;
            case s32:
                res = getHandle(createDeviceDataArray<int>(d, data));
                break;
            case u32:
                res = getHandle(createDeviceDataArray<uint>(d, data));
                break;
            case s64:
                res = getHandle(createDeviceDataArray<intl>(d, data));
                break;
            case u64:
                res = getHandle(createDeviceDataArray<uintl>(d, data));
                break;
            case s16:
                res = getHandle(createDeviceDataArray<short>(d, data));
                break;
            case u16:
                res = getHandle(createDeviceDataArray<ushort>(d, data));
                break;
            case u8:
                res = getHandle(createDeviceDataArray<uchar>(d, data));
                break;
            case b8:
                res = getHandle(createDeviceDataArray<char>(d, data));
                break;
            case f16:
                res = getHandle(createDeviceDataArray<half>(d, data));
                break;
            default: TYPE_ERROR(4, type);
        }

        swap(*arr, res);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_get_device_ptr(void **data, const fly_array arr) {
    try {
        fly_dtype type = getInfo(arr).getType();

        switch (type) {
            // FIXME: Perform copy if memory not continuous
            case f32: *data = getDevicePtr(getArray<float>(arr)); break;
            case f64: *data = getDevicePtr(getArray<double>(arr)); break;
            case c32: *data = getDevicePtr(getArray<cfloat>(arr)); break;
            case c64: *data = getDevicePtr(getArray<cdouble>(arr)); break;
            case s32: *data = getDevicePtr(getArray<int>(arr)); break;
            case u32: *data = getDevicePtr(getArray<uint>(arr)); break;
            case s64: *data = getDevicePtr(getArray<intl>(arr)); break;
            case u64: *data = getDevicePtr(getArray<uintl>(arr)); break;
            case s16: *data = getDevicePtr(getArray<short>(arr)); break;
            case u16: *data = getDevicePtr(getArray<ushort>(arr)); break;
            case u8: *data = getDevicePtr(getArray<uchar>(arr)); break;
            case b8: *data = getDevicePtr(getArray<char>(arr)); break;
            case f16: *data = getDevicePtr(getArray<half>(arr)); break;

            default: TYPE_ERROR(4, type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}

template<typename T>
inline void lockArray(const fly_array arr) {
    memLock(getArray<T>(arr).get());
}

fly_err fly_lock_device_ptr(const fly_array arr) { return fly_lock_array(arr); }

fly_err fly_lock_array(const fly_array arr) {
    try {
        fly_dtype type = getInfo(arr).getType();

        switch (type) {
            case f32: lockArray<float>(arr); break;
            case f64: lockArray<double>(arr); break;
            case c32: lockArray<cfloat>(arr); break;
            case c64: lockArray<cdouble>(arr); break;
            case s32: lockArray<int>(arr); break;
            case u32: lockArray<uint>(arr); break;
            case s64: lockArray<intl>(arr); break;
            case u64: lockArray<uintl>(arr); break;
            case s16: lockArray<short>(arr); break;
            case u16: lockArray<ushort>(arr); break;
            case u8: lockArray<uchar>(arr); break;
            case b8: lockArray<char>(arr); break;
            case f16: lockArray<half>(arr); break;
            default: TYPE_ERROR(4, type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}

template<typename T>
inline bool checkUserLock(const fly_array arr) {
    detail::Array<T> &out = const_cast<detail::Array<T> &>(getArray<T>(arr));
    return isLocked(static_cast<void *>(out.get()));
}

fly_err fly_is_locked_array(bool *res, const fly_array arr) {
    try {
        fly_dtype type = getInfo(arr).getType();

        switch (type) {
            case f32: *res = checkUserLock<float>(arr); break;
            case f64: *res = checkUserLock<double>(arr); break;
            case c32: *res = checkUserLock<cfloat>(arr); break;
            case c64: *res = checkUserLock<cdouble>(arr); break;
            case s32: *res = checkUserLock<int>(arr); break;
            case u32: *res = checkUserLock<uint>(arr); break;
            case s64: *res = checkUserLock<intl>(arr); break;
            case u64: *res = checkUserLock<uintl>(arr); break;
            case s16: *res = checkUserLock<short>(arr); break;
            case u16: *res = checkUserLock<ushort>(arr); break;
            case u8: *res = checkUserLock<uchar>(arr); break;
            case b8: *res = checkUserLock<char>(arr); break;
            case f16: *res = checkUserLock<half>(arr); break;
            default: TYPE_ERROR(4, type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}

template<typename T>
inline void unlockArray(const fly_array arr) {
    memUnlock(getArray<T>(arr).get());
}

fly_err fly_unlock_device_ptr(const fly_array arr) { return fly_unlock_array(arr); }

fly_err fly_unlock_array(const fly_array arr) {
    try {
        fly_dtype type = getInfo(arr).getType();

        switch (type) {
            case f32: unlockArray<float>(arr); break;
            case f64: unlockArray<double>(arr); break;
            case c32: unlockArray<cfloat>(arr); break;
            case c64: unlockArray<cdouble>(arr); break;
            case s32: unlockArray<int>(arr); break;
            case u32: unlockArray<uint>(arr); break;
            case s64: unlockArray<intl>(arr); break;
            case u64: unlockArray<uintl>(arr); break;
            case s16: unlockArray<short>(arr); break;
            case u16: unlockArray<ushort>(arr); break;
            case u8: unlockArray<uchar>(arr); break;
            case b8: unlockArray<char>(arr); break;
            case f16: unlockArray<half>(arr); break;
            default: TYPE_ERROR(4, type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_alloc_device(void **ptr, const dim_t bytes) {
    try {
        FLY_CHECK(fly_init());
        *ptr = memAllocUser(bytes);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_alloc_pinned(void **ptr, const dim_t bytes) {
    try {
        FLY_CHECK(fly_init());
        *ptr = static_cast<void *>(pinnedAlloc<char>(bytes));
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_free_device(void *ptr) {
    try {
        memFreeUser(ptr);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_free_pinned(void *ptr) {
    try {
        pinnedFree(ptr);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_alloc_host(void **ptr, const dim_t bytes) {
    if ((*ptr = malloc(bytes))) {  // NOLINT(hicpp-no-malloc)
        return FLY_SUCCESS;
    }
    return FLY_ERR_NO_MEM;
}

fly_err fly_free_host(void *ptr) {
    free(ptr);  // NOLINT(hicpp-no-malloc)
    return FLY_SUCCESS;
}

fly_err fly_print_mem_info(const char *msg, const int device_id) {
    try {
        int device = device_id;
        if (device == -1) { device = static_cast<int>(getActiveDeviceId()); }

        if (msg != nullptr) {
            ARG_ASSERT(0, strlen(msg) < 256);  // 256 character limit on msg
        }
        ARG_ASSERT(1, device >= 0 && device < getDeviceCount());

        printMemInfo(msg ? msg : "", device);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_device_gc() {
    try {
        signalMemoryCleanup();
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_device_mem_info(size_t *alloc_bytes, size_t *alloc_buffers,
                          size_t *lock_bytes, size_t *lock_buffers) {
    try {
        deviceMemoryInfo(alloc_bytes, alloc_buffers, lock_bytes, lock_buffers);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_set_mem_step_size(const size_t step_bytes) {
    try {
        detail::setMemStepSize(step_bytes);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_get_mem_step_size(size_t *step_bytes) {
    try {
        *step_bytes = detail::getMemStepSize();
    }
    CATCHALL;
    return FLY_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// Memory Manager API
////////////////////////////////////////////////////////////////////////////////

MemoryManager &getMemoryManager(const fly_memory_manager handle) {
    return *static_cast<MemoryManager *>(handle);
}

fly_memory_manager getHandle(MemoryManager &manager) {
    MemoryManager *handle;
    handle = &manager;
    return static_cast<fly_memory_manager>(handle);
}

fly_err fly_create_memory_manager(fly_memory_manager *manager) {
    try {
        FLY_CHECK(fly_init());
        std::unique_ptr<MemoryManager> m(new MemoryManager());
        *manager = getHandle(*m.release());
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_release_memory_manager(fly_memory_manager handle) {
    try {
        // NB: does NOT reset the internal memory manager to be the default:
        // fly_unset_memory_manager_pinned must be used to fully-reset with a new
        // FLY default memory manager
        delete static_cast<MemoryManager *>(handle);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_set_memory_manager(fly_memory_manager mgr) {
    try {
        std::unique_ptr<MemoryManagerFunctionWrapper> newManager(
            new MemoryManagerFunctionWrapper(mgr));
        // Calls shutdown() on the existing memory manager, but does not free
        // the associated handle, if there is one
        detail::setMemoryManager(std::move(newManager));
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_unset_memory_manager() {
    try {
        detail::resetMemoryManager();
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_set_memory_manager_pinned(fly_memory_manager mgr) {
    try {
        // NB: does NOT free if a non-default implementation is set as the
        // current memory manager - the user is responsible for freeing any
        // controlled memory
        std::unique_ptr<MemoryManagerFunctionWrapper> newManager(
            new MemoryManagerFunctionWrapper(mgr));

        // Calls shutdown() on the existing memory manager
        detail::setMemoryManagerPinned(std::move(newManager));
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_unset_memory_manager_pinned() {
    try {
        detail::resetMemoryManagerPinned();
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_get_payload(fly_memory_manager handle, void **payload) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        *payload               = manager.payload;
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_set_payload(fly_memory_manager handle, void *payload) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.payload        = payload;
    }
    CATCHALL;

    return FLY_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// Native memory interface wrapper implementations

fly_err fly_memory_manager_get_active_device_id(fly_memory_manager handle,
                                              int *id) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        *id                    = manager.wrapper->getActiveDeviceId();
    }

    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_native_alloc(fly_memory_manager handle, void **ptr,
                                      size_t size) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        *ptr                   = manager.wrapper->nativeAlloc(size);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_native_free(fly_memory_manager handle, void *ptr) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.wrapper->nativeFree(ptr);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_get_max_memory_size(fly_memory_manager handle,
                                             size_t *size, int id) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        *size                  = manager.wrapper->getMaxMemorySize(id);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_get_memory_pressure_threshold(fly_memory_manager handle,
                                                       float *value) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        *value                 = manager.wrapper->getMemoryPressureThreshold();
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_set_memory_pressure_threshold(fly_memory_manager handle,
                                                       float value) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.wrapper->setMemoryPressureThreshold(value);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// Function setters

fly_err fly_memory_manager_set_initialize_fn(fly_memory_manager handle,
                                           fly_memory_manager_initialize_fn fn) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.initialize_fn  = fn;
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_set_shutdown_fn(fly_memory_manager handle,
                                         fly_memory_manager_shutdown_fn fn) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.shutdown_fn    = fn;
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_set_alloc_fn(fly_memory_manager handle,
                                      fly_memory_manager_alloc_fn fn) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.alloc_fn       = fn;
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_set_allocated_fn(fly_memory_manager handle,
                                          fly_memory_manager_allocated_fn fn) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.allocated_fn   = fn;
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_set_unlock_fn(fly_memory_manager handle,
                                       fly_memory_manager_unlock_fn fn) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.unlock_fn      = fn;
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_set_signal_memory_cleanup_fn(
    fly_memory_manager handle, fly_memory_manager_signal_memory_cleanup_fn fn) {
    try {
        MemoryManager &manager           = getMemoryManager(handle);
        manager.signal_memory_cleanup_fn = fn;
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_set_print_info_fn(fly_memory_manager handle,
                                           fly_memory_manager_print_info_fn fn) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.print_info_fn  = fn;
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_set_user_lock_fn(fly_memory_manager handle,
                                          fly_memory_manager_user_lock_fn fn) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.user_lock_fn   = fn;
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_set_user_unlock_fn(
    fly_memory_manager handle, fly_memory_manager_user_unlock_fn fn) {
    try {
        MemoryManager &manager = getMemoryManager(handle);
        manager.user_unlock_fn = fn;
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_set_is_user_locked_fn(
    fly_memory_manager handle, fly_memory_manager_is_user_locked_fn fn) {
    try {
        MemoryManager &manager    = getMemoryManager(handle);
        manager.is_user_locked_fn = fn;
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_set_get_memory_pressure_fn(
    fly_memory_manager handle, fly_memory_manager_get_memory_pressure_fn fn) {
    try {
        MemoryManager &manager         = getMemoryManager(handle);
        manager.get_memory_pressure_fn = fn;
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_set_jit_tree_exceeds_memory_pressure_fn(
    fly_memory_manager handle,
    fly_memory_manager_jit_tree_exceeds_memory_pressure_fn fn) {
    try {
        MemoryManager &manager                      = getMemoryManager(handle);
        manager.jit_tree_exceeds_memory_pressure_fn = fn;
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_set_add_memory_management_fn(
    fly_memory_manager handle, fly_memory_manager_add_memory_management_fn fn) {
    try {
        MemoryManager &manager           = getMemoryManager(handle);
        manager.add_memory_management_fn = fn;
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_memory_manager_set_remove_memory_management_fn(
    fly_memory_manager handle,
    fly_memory_manager_remove_memory_management_fn fn) {
    try {
        MemoryManager &manager              = getMemoryManager(handle);
        manager.remove_memory_management_fn = fn;
    }
    CATCHALL;

    return FLY_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// Memory Manager wrapper implementations

MemoryManagerFunctionWrapper::MemoryManagerFunctionWrapper(
    fly_memory_manager handle)
    : handle_(handle) {
    MemoryManager &manager = getMemoryManager(handle_);
    manager.wrapper        = this;
}

MemoryManagerFunctionWrapper::~MemoryManagerFunctionWrapper() {
    MemoryManager &manager = getMemoryManager(handle_);
    manager.wrapper        = 0;
}

void MemoryManagerFunctionWrapper::initialize() {
    FLY_CHECK(getMemoryManager(handle_).initialize_fn(handle_));
}

void MemoryManagerFunctionWrapper::shutdown() {
    FLY_CHECK(getMemoryManager(handle_).shutdown_fn(handle_));
}

void *MemoryManagerFunctionWrapper::alloc(bool user_lock, const unsigned ndims,
                                          dim_t *dims,
                                          const unsigned element_size) {
    void *ptr;
    FLY_CHECK(getMemoryManager(handle_).alloc_fn(handle_, &ptr, (int)user_lock,
                                                ndims, dims, element_size));

    return ptr;
}

size_t MemoryManagerFunctionWrapper::allocated(void *ptr) {
    size_t out;
    FLY_CHECK(getMemoryManager(handle_).allocated_fn(handle_, &out, ptr));
    return out;
}

void MemoryManagerFunctionWrapper::unlock(void *ptr, bool user_unlock) {
    FLY_CHECK(
        getMemoryManager(handle_).unlock_fn(handle_, ptr, (int)user_unlock));
}

void MemoryManagerFunctionWrapper::signalMemoryCleanup() {
    FLY_CHECK(getMemoryManager(handle_).signal_memory_cleanup_fn(handle_));
}

void MemoryManagerFunctionWrapper::printInfo(const char *msg,
                                             const int device) {
    FLY_CHECK(getMemoryManager(handle_).print_info_fn(
        handle_, const_cast<char *>(msg), device));
}

void MemoryManagerFunctionWrapper::userLock(const void *ptr) {
    FLY_CHECK(getMemoryManager(handle_).user_lock_fn(handle_,
                                                    const_cast<void *>(ptr)));
}

void MemoryManagerFunctionWrapper::userUnlock(const void *ptr) {
    FLY_CHECK(getMemoryManager(handle_).user_unlock_fn(handle_,
                                                      const_cast<void *>(ptr)));
}

bool MemoryManagerFunctionWrapper::isUserLocked(const void *ptr) {
    int out;
    FLY_CHECK(getMemoryManager(handle_).is_user_locked_fn(
        handle_, &out, const_cast<void *>(ptr)));
    return static_cast<bool>(out);
}

void MemoryManagerFunctionWrapper::usageInfo(size_t * /*alloc_bytes*/,
                                             size_t * /*alloc_buffers*/,
                                             size_t * /*lock_bytes*/,
                                             size_t * /*lock_buffers*/) {
    // Not implemented in the public memory manager API, but for backward
    // compatibility reasons, needs to be in the common memory manager interface
    // so that it can be used with the default memory manager. Called from
    // deviceMemoryInfo from a backend - throws so as to properly propagate
    FLY_ERROR(
        "Device memory info/usage info not supported "
        "for custom memory manager",
        FLY_ERR_NOT_SUPPORTED);
}

float MemoryManagerFunctionWrapper::getMemoryPressure() {
    float out;
    FLY_CHECK(getMemoryManager(handle_).get_memory_pressure_fn(handle_, &out));
    return out;
}

bool MemoryManagerFunctionWrapper::jitTreeExceedsMemoryPressure(size_t bytes) {
    int out;
    FLY_CHECK(getMemoryManager(handle_).jit_tree_exceeds_memory_pressure_fn(
        handle_, &out, bytes));
    return static_cast<bool>(out);
}

size_t MemoryManagerFunctionWrapper::getMemStepSize() {
    // Not implemented in the public memory manager API, but for backward
    // compatibility reasons, needs to be in the common memory manager interface
    // so that it can be used with the default memory manager. Call into the
    // backend implementation so the exception can be properly propagated
    FLY_ERROR("Memory step size API not implemented for custom memory manager",
             FLY_ERR_NOT_SUPPORTED);
}

void MemoryManagerFunctionWrapper::setMemStepSize(size_t new_step_size) {
    // Not implemented in the public memory manager API, but for backward
    // compatibility reasons, needs to be in the common memory manager interface
    // so that it can be used with the default memory manager.
    UNUSED(new_step_size);
    FLY_ERROR("Memory step size API not implemented for custom memory manager ",
             FLY_ERR_NOT_SUPPORTED);
}

