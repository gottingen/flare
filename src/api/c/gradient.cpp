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
#include <gradient.hpp>
#include <handle.hpp>
#include <fly/defines.h>
#include <fly/image.h>

using fly::dim4;
using flare::getArray;
using detail::cdouble;
using detail::cfloat;

template<typename T>
static inline void gradient(fly_array *grad0, fly_array *grad1,
                            const fly_array in) {
    gradient<T>(getArray<T>(*grad0), getArray<T>(*grad1), getArray<T>(in));
}

fly_err fly_gradient(fly_array *grows, fly_array *gcols, const fly_array in) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();
        fly::dim4 idims        = info.dims();

        DIM_ASSERT(2, info.elements() > 0);

        fly_array grad0;
        fly_array grad1;
        FLY_CHECK(fly_create_handle(&grad0, idims.ndims(), idims.get(), type));
        FLY_CHECK(fly_create_handle(&grad1, idims.ndims(), idims.get(), type));

        switch (type) {
            case f32: gradient<float>(&grad0, &grad1, in); break;
            case c32: gradient<cfloat>(&grad0, &grad1, in); break;
            case f64: gradient<double>(&grad0, &grad1, in); break;
            case c64: gradient<cdouble>(&grad0, &grad1, in); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*grows, grad0);
        std::swap(*gcols, grad1);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
