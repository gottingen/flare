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
#include <convolve.hpp>
#include <handle.hpp>
#include <iir.hpp>
#include <fly/arith.h>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/signal.h>

#include <cstdio>

using fly::dim4;
using detail::cdouble;
using detail::cfloat;

fly_err fly_fir(fly_array* y, const fly_array b, const fly_array x) {
    try {
        fly_array out;
        FLY_CHECK(fly_convolve1(&out, x, b, FLY_CONV_EXPAND, FLY_CONV_AUTO));

        dim4 xdims    = getInfo(x).dims();
        fly_seq seqs[] = {fly_span, fly_span, fly_span, fly_span};
        seqs[0].begin = 0.;
        seqs[0].end   = static_cast<double>(xdims[0]) - 1.;
        seqs[0].step  = 1.;
        fly_array res;
        FLY_CHECK(fly_index(&res, out, 4, seqs));
        FLY_CHECK(fly_release_array(out));
        std::swap(*y, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}

template<typename T>
inline static fly_array iir(const fly_array b, const fly_array a,
                           const fly_array x) {
    return getHandle(iir<T>(getArray<T>(b), getArray<T>(a), getArray<T>(x)));
}

fly_err fly_iir(fly_array* y, const fly_array b, const fly_array a,
              const fly_array x) {
    try {
        const ArrayInfo& ainfo = getInfo(a);
        const ArrayInfo& binfo = getInfo(b);
        const ArrayInfo& xinfo = getInfo(x);

        fly_dtype xtype = xinfo.getType();

        ARG_ASSERT(1, ainfo.getType() == xtype);
        ARG_ASSERT(2, binfo.getType() == xtype);
        ARG_ASSERT(1, binfo.ndims() == ainfo.ndims());

        dim4 adims = ainfo.dims();
        dim4 bdims = binfo.dims();
        dim4 xdims = xinfo.dims();

        if (xinfo.ndims() == 0) { return fly_retain_array(y, x); }

        if (xinfo.ndims() > 1) {
            if (binfo.ndims() > 1) {
                for (int i = 1; i < 3; i++) {
                    ARG_ASSERT(1, bdims[i] == xdims[i]);
                }
            }
        }

        // If only a0 is available, just normalize b and perform fir
        if (adims[0] == 1) {
            fly_array bnorm = 0;
            FLY_CHECK(fly_div(&bnorm, b, a, true));
            FLY_CHECK(fly_fir(y, bnorm, x));
            FLY_CHECK(fly_release_array(bnorm));
            return FLY_SUCCESS;
        }

        fly_array res;
        switch (xtype) {
            case f32: res = iir<float>(b, a, x); break;
            case f64: res = iir<double>(b, a, x); break;
            case c32: res = iir<cfloat>(b, a, x); break;
            case c64: res = iir<cdouble>(b, a, x); break;
            default: TYPE_ERROR(1, xtype);
        }

        std::swap(*y, res);
    }
    CATCHALL;
    return FLY_SUCCESS;
}
