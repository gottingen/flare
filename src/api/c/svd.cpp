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

#include <fly/array.h>
#include <fly/dim4.hpp>
#include <fly/lapack.h>

#include <Array.hpp>
#include <backend.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <svd.hpp>
#include <fly/defines.h>

using fly::dim4;
using fly::dtype_traits;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using std::min;

template<typename T>
static inline void svd(fly_array *s, fly_array *u, fly_array *vt,
                       const fly_array in) {
    const ArrayInfo &info = getInfo(in);  // ArrayInfo is the base class which
    dim4 dims             = info.dims();
    int M                 = dims[0];
    int N                 = dims[1];

    using Tr = typename dtype_traits<T>::base_type;

    // Allocate output arrays
    Array<Tr> sA = createEmptyArray<Tr>(dim4(min(M, N)));
    Array<T> uA  = createEmptyArray<T>(dim4(M, M));
    Array<T> vtA = createEmptyArray<T>(dim4(N, N));

    svd<T, Tr>(sA, uA, vtA, getArray<T>(in));

    *s  = getHandle(sA);
    *u  = getHandle(uA);
    *vt = getHandle(vtA);
}

template<typename T>
static inline void svdInPlace(fly_array *s, fly_array *u, fly_array *vt,
                              fly_array in) {
    const ArrayInfo &info = getInfo(in);  // ArrayInfo is the base class which
    dim4 dims             = info.dims();
    int M                 = dims[0];
    int N                 = dims[1];

    using Tr = typename dtype_traits<T>::base_type;

    // Allocate output arrays
    Array<Tr> sA = createEmptyArray<Tr>(dim4(min(M, N)));
    Array<T> uA  = createEmptyArray<T>(dim4(M, M));
    Array<T> vtA = createEmptyArray<T>(dim4(N, N));

    svdInPlace<T, Tr>(sA, uA, vtA, getArray<T>(in));

    *s  = getHandle(sA);
    *u  = getHandle(uA);
    *vt = getHandle(vtA);
}

fly_err fly_svd(fly_array *u, fly_array *s, fly_array *vt, const fly_array in) {
    try {
        const ArrayInfo &info = getInfo(in);
        dim4 dims             = info.dims();

        ARG_ASSERT(3, (dims.ndims() >= 0 && dims.ndims() <= 2));
        fly_dtype type = info.getType();

        if (dims.ndims() == 0) {
            FLY_CHECK(fly_create_handle(u, 0, nullptr, type));
            FLY_CHECK(fly_create_handle(s, 0, nullptr, type));
            FLY_CHECK(fly_create_handle(vt, 0, nullptr, type));
            return FLY_SUCCESS;
        }

        switch (type) {
            case f64: svd<double>(s, u, vt, in); break;
            case f32: svd<float>(s, u, vt, in); break;
            case c64: svd<cdouble>(s, u, vt, in); break;
            case c32: svd<cfloat>(s, u, vt, in); break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_svd_inplace(fly_array *u, fly_array *s, fly_array *vt, fly_array in) {
    try {
        const ArrayInfo &info = getInfo(in);
        dim4 dims             = info.dims();

        ARG_ASSERT(3, (dims.ndims() >= 0 && dims.ndims() <= 2));
        fly_dtype type = info.getType();

        if (dims.ndims() == 0) {
            FLY_CHECK(fly_create_handle(u, 0, nullptr, type));
            FLY_CHECK(fly_create_handle(s, 0, nullptr, type));
            FLY_CHECK(fly_create_handle(vt, 0, nullptr, type));
            return FLY_SUCCESS;
        }

        DIM_ASSERT(3, dims[0] >= dims[1]);

        switch (type) {
            case f64: svdInPlace<double>(s, u, vt, in); break;
            case f32: svdInPlace<float>(s, u, vt, in); break;
            case c64: svdInPlace<cdouble>(s, u, vt, in); break;
            case c32: svdInPlace<cfloat>(s, u, vt, in); break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return FLY_SUCCESS;
}
