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

#pragma once
#include <Array.hpp>
#include <backend.hpp>
#include <common/err_common.hpp>
#include <common/traits.hpp>
#include <copy.hpp>
#include <math.hpp>
#include <types.hpp>

#include <fly/array.h>
#include <fly/defines.h>
#include <fly/dim4.hpp>

namespace flare {

fly_array retain(const fly_array in);

fly::dim4 verifyDims(const unsigned ndims, const dim_t *const dims);

fly_array createHandle(const fly::dim4 &d, fly_dtype dtype);

fly_array createHandleFromValue(const fly::dim4 &d, double val, fly_dtype dtype);

/// This function creates an fly_array handle from memory handle on the device.
///
/// \param[in] d The shape of the new fly_array
/// \param[in] dtype The type of the new fly_array
/// \param[in] data The handle to the device memory
/// \returns a new fly_array with a view to the \p data pointer
fly_array createHandleFromDeviceData(const fly::dim4 &d, fly_dtype dtype,
                                    void *data);

namespace common {
const ArrayInfo &getInfo(const fly_array arr, bool sparse_check = true);

template<typename To>
detail::Array<To> castArray(const fly_array &in);

}  // namespace common

template<typename T>
const detail::Array<T> &getArray(const fly_array &arr) {
    const detail::Array<T> *A = static_cast<const detail::Array<T> *>(arr);
    if ((fly_dtype)fly::dtype_traits<T>::fly_type != A->getType())
        FLY_ERROR("Invalid type for input array.", FLY_ERR_INTERNAL);
    checkAndMigrate(*const_cast<detail::Array<T> *>(A));
    return *A;
}

template<typename T>
detail::Array<T> &getArray(fly_array &arr) {
    detail::Array<T> *A = static_cast<detail::Array<T> *>(arr);
    if ((fly_dtype)fly::dtype_traits<T>::fly_type != A->getType())
        FLY_ERROR("Invalid type for input array.", FLY_ERR_INTERNAL);
    checkAndMigrate(*A);
    return *A;
}

/// Returns the use count
///
/// \note This function is called separately because we cannot call getArray in
/// case the data was built on a different context. so we are avoiding the check
/// and migrate function
template<typename T>
int getUseCount(const fly_array &arr) {
    detail::Array<T> *A = static_cast<detail::Array<T> *>(arr);
    return A->useCount();
}

template<typename T>
fly_array getHandle(const detail::Array<T> &A) {
    detail::Array<T> *ret = new detail::Array<T>(A);
    return static_cast<fly_array>(ret);
}

template<typename T>
fly_array retainHandle(const fly_array in) {
    detail::Array<T> *A   = static_cast<detail::Array<T> *>(in);
    detail::Array<T> *out = new detail::Array<T>(*A);
    return static_cast<fly_array>(out);
}

template<typename T>
fly_array createHandle(const fly::dim4 &d) {
    return getHandle(detail::createEmptyArray<T>(d));
}

template<typename T>
fly_array createHandleFromValue(const fly::dim4 &d, double val) {
    return getHandle(detail::createValueArray<T>(d, detail::scalar<T>(val)));
}

template<typename T>
fly_array createHandleFromData(const fly::dim4 &d, const T *const data) {
    return getHandle(detail::createHostDataArray<T>(d, data));
}

template<typename T>
void copyData(T *data, const fly_array &arr) {
    return detail::copyData(data, getArray<T>(arr));
}

template<typename T>
fly_array copyArray(const fly_array in) {
    const detail::Array<T> &inArray = getArray<T>(in);
    return getHandle<T>(detail::copyArray<T>(inArray));
}

template<typename T>
void releaseHandle(const fly_array arr);

template<typename T>
detail::Array<T> &getCopyOnWriteArray(const fly_array &arr);

}  // namespace flare

using flare::copyArray;
using flare::copyData;
using flare::createHandle;
using flare::createHandleFromData;
using flare::createHandleFromValue;
using flare::getArray;
using flare::getHandle;
using flare::releaseHandle;
using flare::retain;
using flare::verifyDims;
using flare::common::castArray;
using flare::common::getInfo;
