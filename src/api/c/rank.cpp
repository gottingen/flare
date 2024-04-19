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
#include <complex.hpp>
#include <copy.hpp>
#include <handle.hpp>
#include <logic.hpp>
#include <qr.hpp>
#include <reduce.hpp>
#include <fly/array.h>
#include <fly/defines.h>
#include <fly/lapack.h>

using fly::dim4;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using detail::createValueArray;
using detail::getScalar;
using detail::logicOp;
using detail::reduce;
using detail::reduce_all;
using detail::scalar;
using detail::uint;

template<typename T>
static inline uint rank(const fly_array in, double tol) {
    using BT          = typename fly::dtype_traits<T>::base_type;
    const Array<T> In = getArray<T>(in);

    Array<BT> R = createEmptyArray<BT>(dim4());

    // Scoping to get rid of q, r and t as they are not necessary
    {
        Array<T> q = createEmptyArray<T>(dim4());
        Array<T> r = createEmptyArray<T>(dim4());
        Array<T> t = createEmptyArray<T>(dim4());
        qr(q, r, t, In);
        using detail::abs;

        R = abs<BT, T>(r);
    }

    Array<BT> val  = createValueArray<BT>(R.dims(), scalar<BT>(tol));
    Array<char> gt = logicOp<BT, fly_gt_t>(R, val, val.dims());
    Array<char> at = reduce<fly_or_t, char, char>(gt, 1);
    return getScalar<uint>(reduce_all<fly_notzero_t, char, uint>(at));
}

fly_err fly_rank(uint* out, const fly_array in, const double tol) {
    try {
        const ArrayInfo& i_info = getInfo(in);

        if (i_info.ndims() > 2) {
            FLY_ERROR("solve can not be used in batch mode", FLY_ERR_BATCH);
        }

        fly_dtype type = i_info.getType();

        ARG_ASSERT(1, i_info.isFloating());  // Only floating and complex types
        ARG_ASSERT(0, out != nullptr);

        uint output = 0;
        if (i_info.ndims() != 0) {
            switch (type) {
                case f32: output = rank<float>(in, tol); break;
                case f64: output = rank<double>(in, tol); break;
                case c32: output = rank<cfloat>(in, tol); break;
                case c64: output = rank<cdouble>(in, tol); break;
                default: TYPE_ERROR(1, type);
            }
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
