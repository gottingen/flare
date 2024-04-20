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
#include <common/half.hpp>
#include <handle.hpp>
#include <join.hpp>
#include <fly/data.h>

#include <algorithm>
#include <climits>
#include <vector>

using fly::dim4;
using flare::common::half;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;
using std::swap;
using std::vector;

template<typename T>
static inline fly_array join(const int dim, const fly_array first,
                            const fly_array second) {
    return getHandle(join<T>(dim, getArray<T>(first), getArray<T>(second)));
}

template<typename T>
static inline fly_array join_many(const int dim, const unsigned n_arrays,
                                 const fly_array *inputs) {
    vector<Array<T>> inputs_;
    inputs_.reserve(n_arrays);

    dim_t dim_size{0};
    for (unsigned i{0}; i < n_arrays; ++i) {
        const Array<T> &iArray = getArray<T>(inputs[i]);
        if (!iArray.isEmpty()) {
            inputs_.push_back(iArray);
            dim_size += iArray.dims().dims[dim];
        }
    }

    // All dimensions except join dimension must be equal
    // calculate odims size
    fly::dim4 odims{inputs_[0].dims()};
    odims.dims[dim] = dim_size;

    Array<T> out{createEmptyArray<T>(odims)};
    join<T>(out, dim, inputs_);
    return getHandle(out);
}

fly_err fly_join(fly_array *out, const int dim, const fly_array first,
               const fly_array second) {
    try {
        const ArrayInfo &finfo{getInfo(first)};
        const ArrayInfo &sinfo{getInfo(second)};
        const dim4 &fdims{finfo.dims()};
        const dim4 &sdims{sinfo.dims()};

        ARG_ASSERT(1, dim >= 0 && dim < 4);
        ARG_ASSERT(2, finfo.getType() == sinfo.getType());
        if (sinfo.elements() == 0) { return fly_retain_array(out, first); }
        if (finfo.elements() == 0) { return fly_retain_array(out, second); }
        DIM_ASSERT(2, finfo.elements() > 0);
        DIM_ASSERT(3, sinfo.elements() > 0);

        // All dimensions except join dimension must be equal
        for (int i{0}; i < FLY_MAX_DIMS; i++) {
            if (i != dim) { DIM_ASSERT(2, fdims.dims[i] == sdims.dims[i]); }
        }

        fly_array output;

        switch (finfo.getType()) {
            case f32: output = join<float>(dim, first, second); break;
            case c32: output = join<cfloat>(dim, first, second); break;
            case f64: output = join<double>(dim, first, second); break;
            case c64: output = join<cdouble>(dim, first, second); break;
            case b8: output = join<char>(dim, first, second); break;
            case s32: output = join<int>(dim, first, second); break;
            case u32: output = join<uint>(dim, first, second); break;
            case s64: output = join<intl>(dim, first, second); break;
            case u64: output = join<uintl>(dim, first, second); break;
            case s16: output = join<short>(dim, first, second); break;
            case u16: output = join<ushort>(dim, first, second); break;
            case u8: output = join<uchar>(dim, first, second); break;
            case f16: output = join<half>(dim, first, second); break;
            default: TYPE_ERROR(1, finfo.getType());
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_join_many(fly_array *out, const int dim, const unsigned n_arrays,
                    const fly_array *inputs) {
    try {
        ARG_ASSERT(3, inputs != nullptr);

        if (n_arrays == 1) {
            fly_array ret{nullptr};
            FLY_CHECK(fly_retain_array(&ret, *inputs));
            std::swap(*out, ret);
            return FLY_SUCCESS;
        }

        ARG_ASSERT(1, dim >= 0 && dim < FLY_MAX_DIMS);
        ARG_ASSERT(2, n_arrays > 0);

        const fly_array *inputIt{inputs};
        const fly_array *inputEnd{inputs + n_arrays};
        while ((inputIt != inputEnd) && (getInfo(*inputIt).elements() == 0)) {
            ++inputIt;
        }
        if (inputIt == inputEnd) {
            // All arrays have 0 elements
            fly_array ret = nullptr;
            FLY_CHECK(fly_retain_array(&ret, *inputs));
            std::swap(*out, ret);
            return FLY_SUCCESS;
        }

        // inputIt points to first non empty array
        const fly_dtype assertType{getInfo(*inputIt).getType()};
        const dim4 &assertDims{getInfo(*inputIt).dims()};

        // Check all remaining arrays on assertType and assertDims
        while (++inputIt != inputEnd) {
            const ArrayInfo &info = getInfo(*inputIt);
            if (info.elements() > 0) {
                ARG_ASSERT(3, assertType == info.getType());
                const dim4 &infoDims{getInfo(*inputIt).dims()};
                // All dimensions except join dimension must be equal
                for (int i{0}; i < FLY_MAX_DIMS; i++) {
                    if (i != dim) {
                        DIM_ASSERT(3, assertDims.dims[i] == infoDims.dims[i]);
                    }
                }
            }
        }
        fly_array output;

        switch (assertType) {
            case f32: output = join_many<float>(dim, n_arrays, inputs); break;
            case c32: output = join_many<cfloat>(dim, n_arrays, inputs); break;
            case f64: output = join_many<double>(dim, n_arrays, inputs); break;
            case c64: output = join_many<cdouble>(dim, n_arrays, inputs); break;
            case b8: output = join_many<char>(dim, n_arrays, inputs); break;
            case s32: output = join_many<int>(dim, n_arrays, inputs); break;
            case u32: output = join_many<uint>(dim, n_arrays, inputs); break;
            case s64: output = join_many<intl>(dim, n_arrays, inputs); break;
            case u64: output = join_many<uintl>(dim, n_arrays, inputs); break;
            case s16: output = join_many<short>(dim, n_arrays, inputs); break;
            case u16: output = join_many<ushort>(dim, n_arrays, inputs); break;
            case u8: output = join_many<uchar>(dim, n_arrays, inputs); break;
            case f16: output = join_many<half>(dim, n_arrays, inputs); break;
            default: TYPE_ERROR(1, assertType);
        }
        swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
