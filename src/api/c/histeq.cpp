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

#include <arith.hpp>
#include <backend.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <common/moddims.hpp>
#include <copy.hpp>
#include <handle.hpp>
#include <lookup.hpp>
#include <reduce.hpp>
#include <scan.hpp>
#include <fly/data.h>
#include <fly/defines.h>
#include <fly/image.h>
#include <fly/index.h>

using fly::dim4;
using flare::common::cast;
using flare::common::modDims;
using detail::arithOp;
using detail::Array;
using detail::createValueArray;
using detail::getScalar;
using detail::intl;
using detail::lookup;
using detail::reduce_all;
using detail::scan;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T, typename hType>
static fly_array hist_equal(const fly_array& in, const fly_array& hist) {
    const Array<T> input = getArray<T>(in);

    fly_array vInput = 0;
    FLY_CHECK(fly_flat(&vInput, in));

    Array<float> fHist = cast<float>(getArray<hType>(hist));

    const dim4& hDims = fHist.dims();
    dim_t grayLevels  = fHist.elements();

    Array<float> cdf = scan<fly_add_t, float, float>(fHist, 0);

    float minCdf = getScalar<float>(reduce_all<fly_min_t, float, float>(cdf));
    float maxCdf = getScalar<float>(reduce_all<fly_max_t, float, float>(cdf));
    float factor = static_cast<float>(grayLevels - 1) / (maxCdf - minCdf);

    // constant array of min value from cdf
    Array<float> minCnst = createValueArray<float>(hDims, minCdf);
    // constant array of factor variable
    Array<float> facCnst = createValueArray<float>(hDims, factor);
    // cdf(i) - min for all elements
    Array<float> diff = arithOp<float, fly_sub_t>(cdf, minCnst, hDims);
    // multiply factor with difference
    Array<float> normCdf = arithOp<float, fly_mul_t>(diff, facCnst, hDims);
    // index input array with normalized cdf array
    Array<float> idxArr = lookup<float, T>(normCdf, getArray<T>(vInput), 0);

    Array<T> result = cast<T>(idxArr);
    result          = modDims(result, input.dims());

    FLY_CHECK(fly_release_array(vInput));

    return getHandle<T>(result);
}

fly_err fly_hist_equal(fly_array* out, const fly_array in, const fly_array hist) {
    try {
        const ArrayInfo& dataInfo = getInfo(in);
        const ArrayInfo& histInfo = getInfo(hist);

        fly_dtype dataType = dataInfo.getType();
        fly::dim4 histDims = histInfo.dims();

        ARG_ASSERT(2, (histDims.ndims() == 1));

        fly_array output = 0;
        switch (dataType) {
            case f64: output = hist_equal<double, uint>(in, hist); break;
            case f32: output = hist_equal<float, uint>(in, hist); break;
            case s32: output = hist_equal<int, uint>(in, hist); break;
            case u32: output = hist_equal<uint, uint>(in, hist); break;
            case s16: output = hist_equal<short, uint>(in, hist); break;
            case u16: output = hist_equal<ushort, uint>(in, hist); break;
            case s64: output = hist_equal<intl, uint>(in, hist); break;
            case u64: output = hist_equal<uintl, uint>(in, hist); break;
            case u8: output = hist_equal<uchar, uint>(in, hist); break;
            default: TYPE_ERROR(1, dataType);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
