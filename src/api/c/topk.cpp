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

#include <fly/data.h>
#include <fly/statistics.h>

#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <handle.hpp>
#include <topk.hpp>

using flare::common::half;
using detail::createEmptyArray;
using detail::uint;

namespace {

template<typename T>
fly_err topk(fly_array *v, fly_array *i, const fly_array in, const int k,
            const int dim, const fly_topk_function order) {
    auto vals = createEmptyArray<T>(fly::dim4());
    auto idxs = createEmptyArray<unsigned>(fly::dim4());

    topk(vals, idxs, getArray<T>(in), k, dim, order);

    *v = getHandle<T>(vals);
    *i = getHandle<unsigned>(idxs);
    return FLY_SUCCESS;
}
}  //  namespace

fly_err fly_topk(fly_array *values, fly_array *indices, const fly_array in,
               const int k, const int dim, const fly_topk_function order) {
    try {
        fly::topkFunction ord = (order == FLY_TOPK_DEFAULT ? FLY_TOPK_MAX : order);

        const ArrayInfo &inInfo = getInfo(in);

        ARG_ASSERT(2, (inInfo.ndims() > 0));

        if (inInfo.elements() == 1) {
            dim_t dims[1]   = {1};
            fly_err errValue = fly_constant(indices, 0, 1, dims, u32);
            return errValue == FLY_SUCCESS ? fly_retain_array(values, in)
                                          : errValue;
        }

        int rdim           = dim;
        const auto &inDims = inInfo.dims();

        if (rdim == -1) {
            for (dim_t d = 0; d < 4; d++) {
                if (inDims[d] > 1) {
                    rdim = d;
                    break;
                }
            }
        }

        ARG_ASSERT(2, (inInfo.dims()[rdim] >= k));
        ARG_ASSERT(
            4, (k > 0) && (k <= 256));  // TODO(umar): Remove this limitation

        if (rdim != 0) {
            FLY_ERROR("topk is supported along dimenion 0 only.",
                     FLY_ERR_NOT_SUPPORTED);
        }

        fly_dtype type = inInfo.getType();

        switch (type) {
            // TODO(umar): FIX RETURN VALUES HERE
            case f32: topk<float>(values, indices, in, k, rdim, ord); break;
            case f64: topk<double>(values, indices, in, k, rdim, ord); break;
            case u32: topk<uint>(values, indices, in, k, rdim, ord); break;
            case s32: topk<int>(values, indices, in, k, rdim, ord); break;
            case f16: topk<half>(values, indices, in, k, rdim, ord); break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return FLY_SUCCESS;
}
