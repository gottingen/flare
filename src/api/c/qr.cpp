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

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <qr.hpp>
#include <fly/array.h>
#include <fly/defines.h>
#include <fly/lapack.h>

using fly::dim4;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using std::swap;

template<typename T>
static inline void qr(fly_array *q, fly_array *r, fly_array *tau,
                      const fly_array in) {
    Array<T> qArray = createEmptyArray<T>(dim4());
    Array<T> rArray = createEmptyArray<T>(dim4());
    Array<T> tArray = createEmptyArray<T>(dim4());

    qr<T>(qArray, rArray, tArray, getArray<T>(in));

    *q   = getHandle(qArray);
    *r   = getHandle(rArray);
    *tau = getHandle(tArray);
}

template<typename T>
static inline fly_array qr_inplace(fly_array in) {
    return getHandle(qr_inplace<T>(getArray<T>(in)));
}

fly_err fly_qr(fly_array *q, fly_array *r, fly_array *tau, const fly_array in) {
    try {
        const ArrayInfo &i_info = getInfo(in);

        if (i_info.ndims() > 2) {
            FLY_ERROR("qr can not be used in batch mode", FLY_ERR_BATCH);
        }

        fly_dtype type = i_info.getType();

        if (i_info.ndims() == 0) {
            FLY_CHECK(fly_create_handle(q, 0, nullptr, type));
            FLY_CHECK(fly_create_handle(r, 0, nullptr, type));
            FLY_CHECK(fly_create_handle(tau, 0, nullptr, type));
            return FLY_SUCCESS;
        }

        ARG_ASSERT(0, q != nullptr);
        ARG_ASSERT(1, r != nullptr);
        ARG_ASSERT(2, tau != nullptr);
        ARG_ASSERT(3, i_info.isFloating());  // Only floating and complex types

        switch (type) {
            case f32: qr<float>(q, r, tau, in); break;
            case f64: qr<double>(q, r, tau, in); break;
            case c32: qr<cfloat>(q, r, tau, in); break;
            case c64: qr<cdouble>(q, r, tau, in); break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_qr_inplace(fly_array *tau, fly_array in) {
    try {
        const ArrayInfo &i_info = getInfo(in);

        if (i_info.ndims() > 2) {
            FLY_ERROR("qr can not be used in batch mode", FLY_ERR_BATCH);
        }

        fly_dtype type = i_info.getType();

        ARG_ASSERT(1, i_info.isFloating());  // Only floating and complex types
        ARG_ASSERT(0, tau != nullptr);

        if (i_info.ndims() == 0) {
            return fly_create_handle(tau, 0, nullptr, type);
        }

        fly_array out;
        switch (type) {
            case f32: out = qr_inplace<float>(in); break;
            case f64: out = qr_inplace<double>(in); break;
            case c32: out = qr_inplace<cfloat>(in); break;
            case c64: out = qr_inplace<cdouble>(in); break;
            default: TYPE_ERROR(1, type);
        }
        swap(*tau, out);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
