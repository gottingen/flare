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

#include <reorder.hpp>

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <transpose.hpp>

#include <fly/blas.h>
#include <fly/data.h>

using fly::dim4;
using flare::common::half;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;
using std::swap;

template<typename T>
static inline fly_array reorder(const fly_array in, const fly::dim4 &rdims0) {
    Array<T> In = detail::createEmptyArray<T>(fly::dim4(0));
    dim4 rdims  = rdims0;

    if (rdims[0] == 1 && rdims[1] == 0) {
        In = transpose(getArray<T>(in), false);
        std::swap(rdims[0], rdims[1]);
    } else {
        In = getArray<T>(in);
    }
    const dim4 idims    = In.dims();
    const dim4 istrides = In.strides();

    // Ensure all JIT nodes are evaled
    In.eval();

    fly_array out;
    if (rdims[0] == 0 && rdims[1] == 1 && rdims[2] == 2 && rdims[3] == 3) {
        out = getHandle(In);
    } else if (rdims[0] == 0) {
        dim4 odims    = dim4(1, 1, 1, 1);
        dim4 ostrides = dim4(1, 1, 1, 1);
        for (int i = 0; i < 4; i++) {
            odims[i]    = idims[rdims[i]];
            ostrides[i] = istrides[rdims[i]];
        }
        Array<T> Out = In;
        // Use modDims instead of setDataDims to only modify the ArrayInfo
        Out.modDims(odims);
        Out.modStrides(ostrides);
        out = getHandle(Out);
    } else {
        Array<T> Out = reorder<T>(In, rdims);
        out          = getHandle(Out);
    }
    return out;
}

fly_err fly_reorder(fly_array *out, const fly_array in, const fly::dim4 &rdims) {
    try {
        const ArrayInfo &info = getInfo(in);
        fly_dtype type         = info.getType();

        if (info.elements() == 0) { return fly_retain_array(out, in); }

        DIM_ASSERT(1, info.elements() > 0);

        // Check that dimensions are not repeated
        // allDims is to check if all dimensions are there exactly once
        // If all dimensions are present, the allDims will be -1, -1, -1, -1
        // after the loop
        // Example:
        // rdims = {2, 0, 3, 1}
        // i = 0 => 2 found and cond is true so alldims[2] = -1
        // i = 1 => 0 found and cond is true so alldims[0] = -1
        // i = 2 => 3 found and cond is true so alldims[3] = -1
        // i = 3 => 1 found and cond is true so alldims[1] = -1
        // rdims = {2, 0, 3, 2} // Failure case
        // i = 3 => 2 found so cond is false (since alldims[2] = -1 when i = 0)
        // so failed.
        dim_t allDims[] = {0, 1, 2, 3};
        for (int i = 0; i < 4; i++) {
            DIM_ASSERT(i + 2, rdims[i] == allDims[rdims[i]]);
            allDims[rdims[i]] = -1;
        }

        fly_array output;

        switch (type) {
            case f32: output = reorder<float>(in, rdims); break;
            case c32: output = reorder<cfloat>(in, rdims); break;
            case f64: output = reorder<double>(in, rdims); break;
            case c64: output = reorder<cdouble>(in, rdims); break;
            case b8: output = reorder<char>(in, rdims); break;
            case s32: output = reorder<int>(in, rdims); break;
            case u32: output = reorder<uint>(in, rdims); break;
            case u8: output = reorder<uchar>(in, rdims); break;
            case s64: output = reorder<intl>(in, rdims); break;
            case u64: output = reorder<uintl>(in, rdims); break;
            case s16: output = reorder<short>(in, rdims); break;
            case u16: output = reorder<ushort>(in, rdims); break;
            case f16: output = reorder<half>(in, rdims); break;
            default: TYPE_ERROR(1, type);
        }
        swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_reorder(fly_array *out, const fly_array in, const unsigned x,
                  const unsigned y, const unsigned z, const unsigned w) {
    fly::dim4 rdims(x, y, z, w);
    return fly_reorder(out, in, rdims);
}
