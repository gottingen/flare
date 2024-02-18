/*******************************************************
 * Copyright (c) 2018, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

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
