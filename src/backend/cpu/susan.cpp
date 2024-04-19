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

#include <Array.hpp>
#include <kernel/susan.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <fly/features.h>
#include <cmath>
#include <memory>

using fly::features;
using std::shared_ptr;

namespace flare {
namespace cpu {

template<typename T>
unsigned susan(Array<float> &x_out, Array<float> &y_out, Array<float> &resp_out,
               const Array<T> &in, const unsigned radius, const float diff_thr,
               const float geom_thr, const float feature_ratio,
               const unsigned edge) {
    dim4 idims                = in.dims();
    const unsigned corner_lim = in.elements() * feature_ratio;

    auto x_corners    = createEmptyArray<float>(dim4(corner_lim));
    auto y_corners    = createEmptyArray<float>(dim4(corner_lim));
    auto resp_corners = createEmptyArray<float>(dim4(corner_lim));
    auto response     = createEmptyArray<T>(dim4(in.elements()));
    auto corners_found =
        std::shared_ptr<unsigned>(memAlloc<unsigned>(1).release(), memFree);
    corners_found.get()[0] = 0;

    getQueue().enqueue(kernel::susan_responses<T>, response, in, idims[0],
                       idims[1], radius, diff_thr, geom_thr, edge);
    getQueue().enqueue(kernel::non_maximal<T>, x_corners, y_corners,
                       resp_corners, corners_found, idims[0], idims[1],
                       response, edge, corner_lim);
    getQueue().sync();

    const unsigned corners_out = min((corners_found.get())[0], corner_lim);
    if (corners_out == 0) {
        x_out    = createEmptyArray<float>(dim4());
        y_out    = createEmptyArray<float>(dim4());
        resp_out = createEmptyArray<float>(dim4());
        return 0;
    } else {
        x_out    = x_corners;
        y_out    = y_corners;
        resp_out = resp_corners;
        x_out.resetDims(dim4(corners_out));
        y_out.resetDims(dim4(corners_out));
        resp_out.resetDims(dim4(corners_out));
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

}  // namespace cpu
}  // namespace flare
