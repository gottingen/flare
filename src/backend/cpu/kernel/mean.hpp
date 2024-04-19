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

#pragma once
#include <Array.hpp>
#include <common/Transform.hpp>

namespace flare {
namespace cpu {
namespace kernel {

template<typename Ti, typename To, typename Tw>
struct MeanOp {
    common::Transform<Ti, To, fly_add_t> transform;
    To runningMean;
    Tw runningCount;
    MeanOp(Ti mean, Tw count)
        : transform(), runningMean(transform(mean)), runningCount(count) {}

    /// Prevents the optimzation of the mean calculation by some compiler flags
    /// specifically -march=native.
    [[gnu::optimize("01")]] void operator()(Ti _newMean, Tw newCount) {
        To newMean = transform(_newMean);
        if ((newCount != 0) || (runningCount != 0)) {
            Tw runningScale = runningCount;
            Tw newScale     = newCount;
            runningCount += newCount;
            runningScale = runningScale / runningCount;
            newScale     = newScale / runningCount;
            runningMean  = (runningScale * runningMean) + (newScale * newMean);
        }
    }
};

template<typename T, typename Tw, int D>
struct mean_weighted_dim {
    void operator()(Param<T> output, const dim_t outOffset,
                    const CParam<T> input, const dim_t inOffset,
                    const CParam<Tw> weight, const dim_t wtOffset,
                    const int dim) {
        const fly::dim4 odims    = output.dims();
        const fly::dim4 ostrides = output.strides();
        const fly::dim4 istrides = input.strides();
        const fly::dim4 wstrides = weight.strides();
        const int D1            = D - 1;
        for (dim_t i = 0; i < odims[D1]; i++) {
            mean_weighted_dim<T, Tw, D1>()(output, outOffset + i * ostrides[D1],
                                           input, inOffset + i * istrides[D1],
                                           weight, wtOffset + i * wstrides[D1],
                                           dim);
        }
    }
};

template<typename T, typename Tw>
struct mean_weighted_dim<T, Tw, 0> {
    void operator()(Param<T> output, const dim_t outOffset,
                    const CParam<T> input, const dim_t inOffset,
                    const CParam<Tw> weight, const dim_t wtOffset,
                    const int dim) {
        const fly::dim4 idims    = input.dims();
        const fly::dim4 istrides = input.strides();
        const fly::dim4 wstrides = weight.strides();

        T const* const in  = input.get();
        Tw const* const wt = weight.get();
        T* out             = output.get();

        dim_t istride = istrides[dim];
        dim_t wstride = wstrides[dim];
        MeanOp<compute_t<T>, compute_t<T>, compute_t<Tw>> Op(0, 0);
        for (dim_t i = 0; i < idims[dim]; i++) {
            Op(compute_t<T>(in[inOffset + i * istride]),
               compute_t<Tw>(wt[wtOffset + i * wstride]));
        }

        out[outOffset] = Op.runningMean;
    }
};

template<typename Ti, typename Tw, typename To, int D>
struct mean_dim {
    void operator()(Param<To> output, const dim_t outOffset,
                    const CParam<Ti> input, const dim_t inOffset,
                    const int dim) {
        const fly::dim4 odims    = output.dims();
        const fly::dim4 ostrides = output.strides();
        const fly::dim4 istrides = input.strides();
        const int D1            = D - 1;
        for (dim_t i = 0; i < odims[D1]; i++) {
            mean_dim<Ti, Tw, To, D1>()(output, outOffset + i * ostrides[D1],
                                       input, inOffset + i * istrides[D1], dim);
        }
    }
};

template<typename Ti, typename Tw, typename To>
struct mean_dim<Ti, Tw, To, 0> {
    void operator()(Param<To> output, const dim_t outOffset,
                    const CParam<Ti> input, const dim_t inOffset,
                    const int dim) {
        const fly::dim4 idims    = input.dims();
        const fly::dim4 istrides = input.strides();

        Ti const* const in = input.get();
        To* out            = output.get();

        dim_t istride = istrides[dim];
        dim_t end     = inOffset + idims[dim] * istride;
        MeanOp<compute_t<Ti>, compute_t<To>, compute_t<Tw>> Op(0, 0);
        for (dim_t i = inOffset; i < end; i += istride) {
            Op(compute_t<Ti>(in[i]), 1);
        }

        out[outOffset] = Op.runningMean;
    }
};

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
