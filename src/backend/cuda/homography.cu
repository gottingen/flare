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
#include <arith.hpp>
#include <err_cuda.hpp>
#include <kernel/homography.hpp>
#include <fly/dim4.hpp>
#include <algorithm>

#include <limits>

using fly::dim4;

namespace flare {
namespace cuda {

#define RANSACConfidence 0.99f
#define LMEDSConfidence 0.99f
#define LMEDSOutlierRatio 0.4f

template<typename T>
int homography(Array<T> &bestH, const Array<float> &x_src,
               const Array<float> &y_src, const Array<float> &x_dst,
               const Array<float> &y_dst, const Array<float> &initial,
               const fly_homography_type htype, const float inlier_thr,
               const unsigned iterations) {
    const fly::dim4 idims    = x_src.dims();
    const unsigned nsamples = idims[0];

    unsigned iter    = iterations;
    Array<float> err = createEmptyArray<float>(dim4());
    if (htype == FLY_HOMOGRAPHY_LMEDS) {
        iter = ::std::min(
            iter, (unsigned)(log(1.f - LMEDSConfidence) /
                             log(1.f - pow(1.f - LMEDSOutlierRatio, 4.f))));
        err = createValueArray<float>(fly::dim4(nsamples, iter),
                                      std::numeric_limits<float>::max());
    }

    fly::dim4 rdims(4, iter);
    Array<float> fctr = createValueArray<float>(rdims, (float)nsamples);
    Array<float> rnd  = arithOp<float, fly_mul_t>(initial, fctr, rdims);

    Array<T> tmpH = createValueArray<T>(fly::dim4(9, iter), (T)0);

    return kernel::computeH<T>(bestH, tmpH, err, x_src, y_src, x_dst, y_dst,
                               rnd, iter, nsamples, inlier_thr, htype);
}

#define INSTANTIATE(T)                                                      \
    template int homography<T>(                                             \
        Array<T> & H, const Array<float> &x_src, const Array<float> &y_src, \
        const Array<float> &x_dst, const Array<float> &y_dst,               \
        const Array<float> &initial, const fly_homography_type htype,        \
        const float inlier_thr, const unsigned iterations);

INSTANTIATE(float)
INSTANTIATE(double)

}  // namespace cuda
}  // namespace flare
