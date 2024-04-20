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

#include <susan.hpp>

#include <Array.hpp>
#include <err_cuda.hpp>
#include <kernel/susan.hpp>
#include <fly/features.h>

#include <algorithm>

using fly::features;

namespace flare {
namespace cuda {

template<typename T>
unsigned susan(Array<float> &x_out, Array<float> &y_out, Array<float> &resp_out,
               const Array<T> &in, const unsigned radius, const float diff_thr,
               const float geom_thr, const float feature_ratio,
               const unsigned edge) {
    dim4 idims = in.dims();

    const unsigned corner_lim = in.elements() * feature_ratio;
    auto x_corners            = memAlloc<float>(corner_lim);
    auto y_corners            = memAlloc<float>(corner_lim);
    auto resp_corners         = memAlloc<float>(corner_lim);

    auto resp              = memAlloc<T>(in.elements());
    unsigned corners_found = 0;

    kernel::susan_responses<T>(resp.get(), in.get(), idims[0], idims[1], radius,
                               diff_thr, geom_thr, edge);

    kernel::nonMaximal<T>(x_corners.get(), y_corners.get(), resp_corners.get(),
                          &corners_found, idims[0], idims[1], resp.get(), edge,
                          corner_lim);

    const unsigned corners_out = std::min(corners_found, corner_lim);
    if (corners_out == 0) {
        x_out    = createEmptyArray<float>(dim4());
        y_out    = createEmptyArray<float>(dim4());
        resp_out = createEmptyArray<float>(dim4());
        return 0;
    } else {
        x_out = createDeviceDataArray<float>(
            dim4(corners_out), static_cast<void *>(x_corners.get()));
        y_out = createDeviceDataArray<float>(
            dim4(corners_out), static_cast<void *>(y_corners.get()));
        resp_out = createDeviceDataArray<float>(
            dim4(corners_out), static_cast<void *>(resp_corners.get()));
        x_corners.release();
        y_corners.release();
        resp_corners.release();
        return corners_out;
    }
}

#define INSTANTIATE(T)                                                        \
    template unsigned susan<T>(                                               \
        Array<float> & x_out, Array<float> & y_out, Array<float> & score_out, \
        const Array<T> &in, const unsigned radius, const float diff_thr,      \
        const float geom_thr, const float feature_ratio, const unsigned edge);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cuda
}  // namespace flare
