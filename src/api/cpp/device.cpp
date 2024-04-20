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
#include <fly/compatible.h>
#include <fly/device.h>
#include <fly/traits.hpp>
#include "error.hpp"
#include "type_util.hpp"

namespace fly {
void setBackend(const Backend bknd) { FLY_THROW(fly_set_backend(bknd)); }

unsigned getBackendCount() {
    unsigned temp = 1;
    FLY_THROW(fly_get_backend_count(&temp));
    return temp;
}

int getAvailableBackends() {
    int result = 0;
    FLY_THROW(fly_get_available_backends(&result));
    return result;
}

fly::Backend getBackendId(const array &in) {
    auto result = static_cast<fly::Backend>(0);
    FLY_THROW(fly_get_backend_id(&result, in.get()));
    return result;
}

int getDeviceId(const array &in) {
    int device = getDevice();
    FLY_THROW(fly_get_device_id(&device, in.get()));
    return device;
}

fly::Backend getActiveBackend() {
    auto result = static_cast<fly::Backend>(0);
    FLY_THROW(fly_get_active_backend(&result));
    return result;
}

void info() { FLY_THROW(fly_info()); }

const char *infoString(const bool verbose) {
    char *str = NULL;
    FLY_THROW(fly_info_string(&str, verbose));
    return str;
}

void deviceprop(char *d_name, char *d_platform, char *d_toolkit,
                char *d_compute) {
    deviceInfo(d_name, d_platform, d_toolkit, d_compute);
}
void deviceInfo(char *d_name, char *d_platform, char *d_toolkit,
                char *d_compute) {
    FLY_THROW(fly_device_info(d_name, d_platform, d_toolkit, d_compute));
}

int getDeviceCount() {
    int devices = -1;
    FLY_THROW(fly_get_device_count(&devices));
    return devices;
}

int devicecount() { return getDeviceCount(); }

void setDevice(const int device) { FLY_THROW(fly_set_device(device)); }

void deviceset(const int device) { setDevice(device); }

int getDevice() {
    int device = 0;
    FLY_THROW(fly_get_device(&device));
    return device;
}

bool isDoubleAvailable(const int device) {
    bool temp;
    FLY_THROW(fly_get_dbl_support(&temp, device));
    return temp;
}

bool isHalfAvailable(const int device) {
    bool temp;
    FLY_THROW(fly_get_half_support(&temp, device));
    return temp;
}

int deviceget() { return getDevice(); }

void sync(int device) { FLY_THROW(fly_sync(device)); }

// Alloc device memory
void *alloc(const size_t elements, const fly::dtype type) {
    void *ptr;
    FLY_DEPRECATED_WARNINGS_OFF
    FLY_THROW(fly_alloc_device(&ptr, elements * size_of(type)));
    FLY_DEPRECATED_WARNINGS_ON
    // FIXME: Add to map
    return ptr;
}

// Alloc device memory
void *fly_alloc(const size_t bytes) {
    void *ptr;
    FLY_THROW(fly_alloc_device(&ptr, bytes));
    return ptr;
}

// Alloc pinned memory
void *pinned(const size_t elements, const fly::dtype type) {
    void *ptr;
    FLY_THROW(fly_alloc_pinned(&ptr, elements * size_of(type)));
    // FIXME: Add to map
    return ptr;
}

void free(const void *ptr) {
    // FIXME: look up map and call the right free
    FLY_DEPRECATED_WARNINGS_OFF
    FLY_THROW(fly_free_device(const_cast<void *>(ptr)));
    FLY_DEPRECATED_WARNINGS_ON
}

void fly_free(const void *ptr) {
    FLY_THROW(fly_free_device(const_cast<void *>(ptr)));
}

void freePinned(const void *ptr) {
    // FIXME: look up map and call the right free
    FLY_THROW(fly_free_pinned((void *)ptr));
}

void *allocHost(const size_t elements, const fly::dtype type) {
    void *ptr;
    FLY_THROW(fly_alloc_host(&ptr, elements * size_of(type)));
    return ptr;
}

void freeHost(const void *ptr) { FLY_THROW(fly_free_host((void *)ptr)); }

void printMemInfo(const char *msg, const int device_id) {
    FLY_THROW(fly_print_mem_info(msg, device_id));
}

void deviceGC() { FLY_THROW(fly_device_gc()); }

void deviceMemInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                   size_t *lock_bytes, size_t *lock_buffers) {
    FLY_THROW(fly_device_mem_info(alloc_bytes, alloc_buffers, lock_bytes,
                                lock_buffers));
}

void setMemStepSize(const size_t step_bytes) {
    FLY_THROW(fly_set_mem_step_size(step_bytes));
}

size_t getMemStepSize() {
    size_t size_bytes = 0;
    FLY_THROW(fly_get_mem_step_size(&size_bytes));
    return size_bytes;
}

FLY_DEPRECATED_WARNINGS_OFF
#define INSTANTIATE(T)                                                        \
    template<>                                                                \
    FLY_API T *alloc(const size_t elements) {                                   \
        return (T *)alloc(elements, (fly::dtype)dtype_traits<T>::fly_type);     \
    }                                                                         \
    template<>                                                                \
    FLY_API T *pinned(const size_t elements) {                                  \
        return (T *)pinned(elements, (fly::dtype)dtype_traits<T>::fly_type);    \
    }                                                                         \
    template<>                                                                \
    FLY_API T *allocHost(const size_t elements) {                               \
        return (T *)allocHost(elements, (fly::dtype)dtype_traits<T>::fly_type); \
    }

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(unsigned)
INSTANTIATE(unsigned char)
INSTANTIATE(char)
INSTANTIATE(short)
INSTANTIATE(unsigned short)
INSTANTIATE(long long)
INSTANTIATE(unsigned long long)
FLY_DEPRECATED_WARNINGS_ON

}  // namespace fly
