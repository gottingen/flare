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
#include <common/err_common.hpp>
#include <handle.hpp>
#include <regions.hpp>
#include <types.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/image.h>

using fly::dim4;
using detail::uint;
using detail::ushort;

template<typename T>
static fly_array regions(fly_array const &in, fly_connectivity connectivity) {
    return getHandle<T>(regions<T>(getArray<char>(in), connectivity));
}

fly_err fly_regions(fly_array *out, const fly_array in,
                  const fly_connectivity connectivity, const fly_dtype type) {
    try {
        ARG_ASSERT(2, (connectivity == FLY_CONNECTIVITY_4 ||
                       connectivity == FLY_CONNECTIVITY_8_4));

        const ArrayInfo &info = getInfo(in);
        fly::dim4 dims         = info.dims();

        dim_t in_ndims = dims.ndims();
        DIM_ASSERT(1, (in_ndims == 2));

        fly_dtype in_type = info.getType();
        if (in_type != b8) { TYPE_ERROR(1, in_type); }

        fly_array output;
        switch (type) {
            case f32: output = regions<float>(in, connectivity); break;
            case f64: output = regions<double>(in, connectivity); break;
            case s32: output = regions<int>(in, connectivity); break;
            case u32: output = regions<uint>(in, connectivity); break;
            case s16: output = regions<short>(in, connectivity); break;
            case u16: output = regions<ushort>(in, connectivity); break;
            default: TYPE_ERROR(0, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
