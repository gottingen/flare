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
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <math.hpp>
#include <sort.hpp>
#include <fly/arith.h>
#include <fly/data.h>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/index.h>
#include <fly/statistics.h>

using fly::dim4;
using detail::Array;
using detail::division;
using detail::uchar;
using detail::uint;
using detail::ushort;
using std::sort;

template<typename T>
static double median(const fly_array& in) {
    dim_t nElems = getInfo(in).elements();
    dim4 dims(nElems, 1, 1, 1);
    ARG_ASSERT(0, nElems > 0);

    fly_array temp = 0;
    FLY_CHECK(fly_moddims(&temp, in, 1, dims.get()));
    const Array<T>& input = getArray<T>(temp);

    // Shortcut cases for 1 or 2 elements
    if (nElems == 1) {
        T result;
        FLY_CHECK(fly_get_data_ptr((void*)&result, in));
        return result;
    }
    if (nElems == 2) {
        T result[2];
        FLY_CHECK(fly_get_data_ptr((void*)&result, in));
        return division(
            (static_cast<double>(result[0]) + static_cast<double>(result[1])),
            2.0);
    }

    double mid       = static_cast<double>(nElems + 1) / 2.0;
    fly_seq mdSpan[1] = {fly_make_seq(mid - 1, mid, 1)};

    Array<T> sortedArr = sort<T>(input, 0, true);

    fly_array sarrHandle = getHandle<T>(sortedArr);

    double result;
    T resPtr[2];
    fly_array res = 0;
    FLY_CHECK(fly_index(&res, sarrHandle, 1, mdSpan));
    FLY_CHECK(fly_get_data_ptr((void*)&resPtr, res));

    FLY_CHECK(fly_release_array(res));
    FLY_CHECK(fly_release_array(sarrHandle));
    FLY_CHECK(fly_release_array(temp));

    if (nElems % 2 == 1) {
        result = resPtr[0];
    } else {
        result = division(
            static_cast<double>(resPtr[0]) + static_cast<double>(resPtr[1]),
            2.0);
    }

    return result;
}

template<typename T>
static fly_array median(const fly_array& in, const dim_t dim) {
    const Array<T> input = getArray<T>(in);

    // Shortcut cases for 1 element along selected dimension
    if (input.dims()[dim] == 1) {
        Array<T> result = copyArray<T>(input);
        return getHandle<T>(result);
    }

    Array<T> sortedIn = sort<T>(input, dim, true);

    size_t dimLength = input.dims()[dim];
    double mid       = static_cast<double>(dimLength + 1) / 2.0;
    fly_array left    = 0;

    fly_seq slices[4] = {fly_span, fly_span, fly_span, fly_span};
    slices[dim]      = fly_make_seq(mid - 1.0, mid - 1.0, 1.0);

    fly_array sortedIn_handle = getHandle<T>(sortedIn);
    FLY_CHECK(fly_index(&left, sortedIn_handle, input.ndims(), slices));

    fly_array out = nullptr;
    if (dimLength % 2 == 1) {
        // mid-1 is our guy
        if (input.isFloating()) {
            FLY_CHECK(fly_release_array(sortedIn_handle));
            return left;
        }

        // Return as floats for consistency
        fly_array out;
        FLY_CHECK(fly_cast(&out, left, f32));
        FLY_CHECK(fly_release_array(left));
        FLY_CHECK(fly_release_array(sortedIn_handle));
        return out;
    } else {
        // ((mid-1)+mid)/2 is our guy
        dim4 dims      = input.dims();
        fly_array right = 0;
        slices[dim]    = fly_make_seq(mid, mid, 1.0);

        FLY_CHECK(fly_index(&right, sortedIn_handle, dims.ndims(), slices));

        fly_array sumarr = 0;
        fly_array carr   = 0;

        dim4 cdims = dims;
        cdims[dim] = 1;
        FLY_CHECK(fly_constant(&carr, 0.5, cdims.ndims(), cdims.get(),
                             input.isDouble() ? f64 : f32));

        if (!input.isFloating()) {
            fly_array lleft, rright;
            FLY_CHECK(fly_cast(&lleft, left, f32));
            FLY_CHECK(fly_cast(&rright, right, f32));
            FLY_CHECK(fly_release_array(left));
            FLY_CHECK(fly_release_array(right));
            left  = lleft;
            right = rright;
        }

        FLY_CHECK(fly_add(&sumarr, left, right, false));
        FLY_CHECK(fly_mul(&out, sumarr, carr, false));

        FLY_CHECK(fly_release_array(left));
        FLY_CHECK(fly_release_array(right));
        FLY_CHECK(fly_release_array(sumarr));
        FLY_CHECK(fly_release_array(carr));
        FLY_CHECK(fly_release_array(sortedIn_handle));
    }
    return out;
}

fly_err fly_median_all(double* realVal, double* imagVal,  // NOLINT
                     const fly_array in) {
    UNUSED(imagVal);
    try {
        const ArrayInfo& info = getInfo(in);
        fly_dtype type         = info.getType();

        ARG_ASSERT(2, info.ndims() > 0);
        switch (type) {
            case f64: *realVal = median<double>(in); break;
            case f32: *realVal = median<float>(in); break;
            case s32: *realVal = median<int>(in); break;
            case u32: *realVal = median<uint>(in); break;
            case s16: *realVal = median<short>(in); break;
            case u16: *realVal = median<ushort>(in); break;
            case u8: *realVal = median<uchar>(in); break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return FLY_SUCCESS;
}

fly_err fly_median(fly_array* out, const fly_array in, const dim_t dim) {
    try {
        ARG_ASSERT(2, (dim >= 0 && dim <= 4));

        fly_array output       = 0;
        const ArrayInfo& info = getInfo(in);

        ARG_ASSERT(1, info.ndims() > 0);
        fly_dtype type = info.getType();
        switch (type) {
            case f64: output = median<double>(in, dim); break;
            case f32: output = median<float>(in, dim); break;
            case s32: output = median<int>(in, dim); break;
            case u32: output = median<uint>(in, dim); break;
            case s16: output = median<short>(in, dim); break;
            case u16: output = median<ushort>(in, dim); break;
            case u8: output = median<uchar>(in, dim); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return FLY_SUCCESS;
}
