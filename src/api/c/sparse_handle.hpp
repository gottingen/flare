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
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <copy.hpp>
#include <handle.hpp>
#include <math.hpp>
#include <fly/array.h>
#include <fly/dim4.hpp>

#include <common/SparseArray.hpp>

namespace flare {

const common::SparseArrayBase &getSparseArrayBase(const fly_array in,
                                                  bool device_check = true);

template<typename T>
const common::SparseArray<T> &getSparseArray(const fly_array &arr) {
    const common::SparseArray<T> *A =
        static_cast<const common::SparseArray<T> *>(arr);
    ARG_ASSERT(0, A->isSparse() == true);
    checkAndMigrate(*A);
    return *A;
}

template<typename T>
common::SparseArray<T> &getSparseArray(fly_array &arr) {
    common::SparseArray<T> *A = static_cast<common::SparseArray<T> *>(arr);
    ARG_ASSERT(0, A->isSparse() == true);
    checkAndMigrate(*A);
    return *A;
}

template<typename T>
static fly_array getHandle(const common::SparseArray<T> &A) {
    common::SparseArray<T> *ret = new common::SparseArray<T>(A);
    return static_cast<fly_array>(ret);
}

template<typename T>
static void releaseSparseHandle(const fly_array arr) {
    common::destroySparseArray(static_cast<common::SparseArray<T> *>(arr));
}

template<typename T>
fly_array retainSparseHandle(const fly_array in) {
    const common::SparseArray<T> *sparse =
        static_cast<const common::SparseArray<T> *>(in);
    common::SparseArray<T> *out = new common::SparseArray<T>(*sparse);
    return static_cast<fly_array>(out);
}

// based on castArray in handle.hpp
template<typename To>
common::SparseArray<To> castSparse(const fly_array &in) {
    const ArrayInfo &info = getInfo(in, false);
    using namespace common;

#define CAST_SPARSE(Ti)                                                          \
    do {                                                                         \
        const SparseArray<Ti> sparse = getSparseArray<Ti>(in);                   \
        detail::Array<To> values     = common::cast<To, Ti>(sparse.getValues()); \
        return createArrayDataSparseArray(                                       \
            sparse.dims(), values, sparse.getRowIdx(), sparse.getColIdx(),       \
            sparse.getStorage());                                                \
    } while (0)

    switch (info.getType()) {
        case f32: CAST_SPARSE(float);
        case f64: CAST_SPARSE(double);
        case c32: CAST_SPARSE(detail::cfloat);
        case c64: CAST_SPARSE(detail::cdouble);
        default: TYPE_ERROR(1, info.getType());
    }
}

template<typename T>
static fly_array copySparseArray(const fly_array in) {
    const common::SparseArray<T> &inArray = getSparseArray<T>(in);
    return getHandle<T>(common::copySparseArray<T>(inArray));
}

}  // namespace flare

using flare::getHandle;
