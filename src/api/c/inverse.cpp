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
#include <inverse.hpp>
#include <fly/array.h>
#include <fly/defines.h>
#include <fly/lapack.h>

using detail::cdouble;
using detail::cfloat;

template<typename T>
static inline fly_array inverse(const fly_array in) {
    return getHandle(inverse<T>(getArray<T>(in)));
}

fly_err fly_inverse(fly_array* out, const fly_array in, const fly_mat_prop options) {
    try {
        const ArrayInfo& i_info = getInfo(in);

        if (i_info.ndims() > 2) {
            FLY_ERROR("solve can not be used in batch mode", FLY_ERR_BATCH);
        }

        fly_dtype type = i_info.getType();

        if (options != FLY_MAT_NONE) {
            FLY_ERROR("Using this property is not yet supported in inverse",
                     FLY_ERR_NOT_SUPPORTED);
        }

        DIM_ASSERT(
            1, i_info.dims()[0] == i_info.dims()[1]);  // Only square matrices
        ARG_ASSERT(1, i_info.isFloating());  // Only floating and complex types

        fly_array output;

        if (i_info.ndims() == 0) { return fly_retain_array(out, in); }

        switch (type) {
            case f32: output = inverse<float>(in); break;
            case f64: output = inverse<double>(in); break;
            case c32: output = inverse<cfloat>(in); break;
            case c64: output = inverse<cdouble>(in); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
