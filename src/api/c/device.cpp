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

#include <Array.hpp>
#include <backend.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <common/util.hpp>
#include <handle.hpp>
#include <platform.hpp>
#include <sparse_handle.hpp>
#include <fly/backend.h>
#include <fly/device.h>
#include <fly/dim4.hpp>
#include <fly/version.h>

#if defined(USE_MKL)
#include <mkl_service.h>
#endif

#include <cstring>
#include <string>

using fly::dim4;
using flare::getSparseArray;
using flare::common::getCacheDirectory;
using flare::common::getEnvVar;
using flare::common::half;
using flare::common::JIT_KERNEL_CACHE_DIRECTORY_ENV_NAME;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using detail::devprop;
using detail::evalFlag;
using detail::getActiveDeviceId;
using detail::getBackend;
using detail::getDeviceCount;
using detail::getDeviceInfo;
using detail::init;
using detail::intl;
using detail::isDoubleSupported;
using detail::isHalfSupported;
using detail::setDevice;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

fly_err fly_set_backend(const fly_backend bknd) {
    try {
        if (bknd != getBackend() && bknd != FLY_BACKEND_DEFAULT) {
            return FLY_ERR_ARG;
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_get_backend_count(unsigned* num_backends) {
    *num_backends = 1;
    return FLY_SUCCESS;
}

fly_err fly_get_available_backends(int* result) {
    try {
        *result = getBackend();
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_get_backend_id(fly_backend* result, const fly_array in) {
    try {
        if (in) {
            const ArrayInfo& info = getInfo(in, false);
            *result               = info.getBackendId();
        } else {
            return FLY_ERR_ARG;
        }
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_get_device_id(int* device, const fly_array in) {
    try {
        if (in) {
            const ArrayInfo& info = getInfo(in, false);
            *device               = static_cast<int>(info.getDevId());
        } else {
            return FLY_ERR_ARG;
        }
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_get_active_backend(fly_backend* result) {
    *result = static_cast<fly_backend>(getBackend());
    return FLY_SUCCESS;
}

fly_err fly_init() {
    try {
        thread_local std::once_flag flag;
        std::call_once(flag, []() {
            init();
#if defined(USE_MKL) && !defined(USE_STATIC_MKL)
            int errCode = -1;
            // Have used the FLY_MKL_INTERFACE_SIZE as regular if's so that
            // we will know if these are not defined when using MKL when a
            // compilation error is generated.
            if (FLY_MKL_INTERFACE_SIZE == 4) {
                errCode = mkl_set_interface_layer(MKL_INTERFACE_LP64);
            } else if (FLY_MKL_INTERFACE_SIZE == 8) {
                errCode = mkl_set_interface_layer(MKL_INTERFACE_ILP64);
            }
            if (errCode == -1) {
                FLY_ERROR(
                    "Intel MKL Interface layer was not specified prior to the "
                    "call and the input parameter is incorrect.",
                    FLY_ERR_RUNTIME);
            }
            switch (FLY_MKL_THREAD_LAYER) {
                case 0:
                    errCode = mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
                    break;
                case 1:
                    errCode = mkl_set_threading_layer(MKL_THREADING_GNU);
                    break;
                case 2:
                    errCode = mkl_set_threading_layer(MKL_THREADING_INTEL);
                    break;
                case 3:
                    errCode = mkl_set_threading_layer(MKL_THREADING_TBB);
                    break;
            }
            if (errCode == -1) {
                FLY_ERROR(
                    "Intel MKL Thread layer was not specified prior to the "
                    "call and the input parameter is incorrect.",
                    FLY_ERR_RUNTIME);
            }
#endif
        });
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_info() {
    try {
        printf("%s", getDeviceInfo().c_str());  // NOLINT
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_info_string(char** str, const bool verbose) {
    UNUSED(verbose);  // TODO(umar): Add something useful
    try {
        std::string infoStr = getDeviceInfo();
        void* halloc_ptr    = nullptr;
        fly_alloc_host(&halloc_ptr, sizeof(char) * (infoStr.size() + 1));
        memcpy(str, &halloc_ptr, sizeof(void*));

        // Need to do a deep copy
        // str.c_str wont cut it
        infoStr.copy(*str, infoStr.size());
        (*str)[infoStr.size()] = '\0';
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_device_info(char* d_name, char* d_platform, char* d_toolkit,
                      char* d_compute) {
    try {
        devprop(d_name, d_platform, d_toolkit, d_compute);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_get_dbl_support(bool* available, const int device) {
    try {
        *available = isDoubleSupported(device);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_get_half_support(bool* available, const int device) {
    try {
        *available = isHalfSupported(device);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_get_device_count(int* nDevices) {
    try {
        *nDevices = getDeviceCount();
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_get_device(int* device) {
    try {
        *device = static_cast<int>(getActiveDeviceId());
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_set_device(const int device) {
    try {
        ARG_ASSERT(0, device >= 0);
        if (setDevice(device) < 0) {
            int ndevices = getDeviceCount();
            if (ndevices == 0) {
                FLY_ERROR(
                    "No devices were found on this system. Ensure "
                    "you have installed the device driver as well as the "
                    "necessary runtime libraries for your platform.",
                    FLY_ERR_RUNTIME);
            } else {
                char buf[512];
                char err_msg[] =
                    "The device index of %d is out of range. Use a value "
                    "between 0 and %d.";
                snprintf(buf, 512, err_msg, device, ndevices - 1);  // NOLINT
                FLY_ERROR(buf, FLY_ERR_ARG);
            }
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_sync(const int device) {
    try {
        int dev = device == -1 ? static_cast<int>(getActiveDeviceId()) : device;
        detail::sync(dev);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

template<typename T>
static inline void eval(fly_array arr) {
    getArray<T>(arr).eval();
}

template<typename T>
static inline void sparseEval(fly_array arr) {
    getSparseArray<T>(arr).eval();
}

fly_err fly_eval(fly_array arr) {
    try {
        const ArrayInfo& info = getInfo(arr, false);
        fly_dtype type         = info.getType();

        if (info.isSparse()) {
            switch (type) {
                case f32: sparseEval<float>(arr); break;
                case f64: sparseEval<double>(arr); break;
                case c32: sparseEval<cfloat>(arr); break;
                case c64: sparseEval<cdouble>(arr); break;
                default: TYPE_ERROR(0, type);
            }
        } else {
            switch (type) {
                case f32: eval<float>(arr); break;
                case f64: eval<double>(arr); break;
                case c32: eval<cfloat>(arr); break;
                case c64: eval<cdouble>(arr); break;
                case s32: eval<int>(arr); break;
                case u32: eval<uint>(arr); break;
                case u8: eval<uchar>(arr); break;
                case b8: eval<char>(arr); break;
                case s64: eval<intl>(arr); break;
                case u64: eval<uintl>(arr); break;
                case s16: eval<short>(arr); break;
                case u16: eval<ushort>(arr); break;
                case f16: eval<half>(arr); break;
                default: TYPE_ERROR(0, type);
            }
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}

template<typename T>
static inline void evalMultiple(int num, fly_array* arrayPtrs) {
    Array<T> empty = createEmptyArray<T>(dim4());
    std::vector<Array<T>*> arrays(num, &empty);

    for (int i = 0; i < num; i++) {
        arrays[i] = reinterpret_cast<Array<T>*>(arrayPtrs[i]);
    }

    evalMultiple<T>(arrays);
}

fly_err fly_eval_multiple(int num, fly_array* arrays) {
    try {
        const ArrayInfo& info = getInfo(arrays[0]);
        fly_dtype type         = info.getType();
        const dim4& dims      = info.dims();

        for (int i = 1; i < num; i++) {
            const ArrayInfo& currInfo = getInfo(arrays[i]);

            // FIXME: This needs to be removed when new functionality is added
            if (type != currInfo.getType()) {
                FLY_ERROR("All arrays must be of same type", FLY_ERR_TYPE);
            }

            if (dims != currInfo.dims()) {
                FLY_ERROR("All arrays must be of same size", FLY_ERR_SIZE);
            }
        }

        switch (type) {
            case f32: evalMultiple<float>(num, arrays); break;
            case f64: evalMultiple<double>(num, arrays); break;
            case c32: evalMultiple<cfloat>(num, arrays); break;
            case c64: evalMultiple<cdouble>(num, arrays); break;
            case s32: evalMultiple<int>(num, arrays); break;
            case u32: evalMultiple<uint>(num, arrays); break;
            case u8: evalMultiple<uchar>(num, arrays); break;
            case b8: evalMultiple<char>(num, arrays); break;
            case s64: evalMultiple<intl>(num, arrays); break;
            case u64: evalMultiple<uintl>(num, arrays); break;
            case s16: evalMultiple<short>(num, arrays); break;
            case u16: evalMultiple<ushort>(num, arrays); break;
            case f16: evalMultiple<half>(num, arrays); break;
            default: TYPE_ERROR(0, type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_set_manual_eval_flag(bool flag) {
    try {
        bool& backendFlag = evalFlag();
        backendFlag       = !flag;
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_get_manual_eval_flag(bool* flag) {
    try {
        bool backendFlag = evalFlag();
        *flag            = !backendFlag;
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_get_kernel_cache_directory(size_t* length, char* path) {
    try {
        std::string& cache_path = getCacheDirectory();
        if (path == nullptr) {
            ARG_ASSERT(length != nullptr, 1);
            *length = cache_path.size();
        } else {
            size_t min_len = cache_path.size();
            if (length) {
                if (*length < cache_path.size()) {
                    FLY_ERROR("Length not sufficient to store the path",
                             FLY_ERR_SIZE);
                }
                min_len = std::min(*length, cache_path.size());
            }
            memcpy(path, cache_path.c_str(), min_len);
        }
    }
    CATCHALL
    return FLY_SUCCESS;
}

fly_err fly_set_kernel_cache_directory(const char* path, int override_env) {
    try {
        ARG_ASSERT(path != nullptr, 1);
        if (override_env) {
            getCacheDirectory() = std::string(path);
        } else {
            auto env_path = getEnvVar(JIT_KERNEL_CACHE_DIRECTORY_ENV_NAME);
            if (env_path.empty()) { getCacheDirectory() = std::string(path); }
        }
    }
    CATCHALL
    return FLY_SUCCESS;
}
