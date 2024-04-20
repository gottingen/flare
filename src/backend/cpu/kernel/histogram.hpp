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
#include <Param.hpp>
#include <types.hpp>

namespace flare {
namespace cpu {
namespace kernel {

template<typename T, bool IsLinear>
void histogram(Param<uint> out, CParam<T> in, const unsigned nbins,
               const double minval, const double maxval) {
    dim4 const outDims  = out.dims();
    float const step    = (maxval - minval) / (float)nbins;
    dim4 const inDims   = in.dims();
    dim4 const iStrides = in.strides();
    dim4 const oStrides = out.strides();
    dim_t const nElems  = inDims[0] * inDims[1];

    auto minValT = compute_t<T>(minval);
    for (dim_t b3 = 0; b3 < outDims[3]; b3++) {
        uint* outData   = out.get() + b3 * oStrides[3];
        const T* inData = in.get() + b3 * iStrides[3];
        for (dim_t b2 = 0; b2 < outDims[2]; b2++) {
            for (dim_t i = 0; i < nElems; i++) {
                int idx =
                    IsLinear
                        ? i
                        : ((i % inDims[0]) + (i / inDims[0]) * iStrides[1]);
                int bin = (int)((compute_t<T>(inData[idx]) - minValT) / step);
                bin     = std::max(bin, 0);
                bin     = std::min(bin, (int)(nbins - 1));
                outData[bin]++;
            }
            inData += iStrides[2];
            outData += oStrides[2];
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare
