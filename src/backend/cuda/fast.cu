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

#include <fast.hpp>

#include <LookupTable1D.hpp>
#include <kernel/fast.hpp>
#include <kernel/fast_lut.hpp>
#include <fly/dim4.hpp>

#include <mutex>

using fly::dim4;
using fly::features;

namespace flare {
namespace cuda {

template<typename T>
unsigned fast(Array<float> &x_out, Array<float> &y_out, Array<float> &score_out,
              const Array<T> &in, const float thr, const unsigned arc_length,
              const bool non_max, const float feature_ratio,
              const unsigned edge) {
    unsigned nfeat;
    float *d_x_out;
    float *d_y_out;
    float *d_score_out;

    // TODO(pradeep) Figure out a better way to create lut Array only once
    const Array<unsigned char> lut = createHostDataArray(
        fly::dim4(sizeof(FAST_LUT) / sizeof(unsigned char)), FAST_LUT);

    LookupTable1D<unsigned char> fastLUT(lut);

    kernel::fast<T>(&nfeat, &d_x_out, &d_y_out, &d_score_out, in, thr,
                    arc_length, non_max, feature_ratio, edge, fastLUT);

    if (nfeat > 0) {
        const dim4 out_dims(nfeat);

        x_out     = createDeviceDataArray<float>(out_dims, d_x_out);
        y_out     = createDeviceDataArray<float>(out_dims, d_y_out);
        score_out = createDeviceDataArray<float>(out_dims, d_score_out);
    }
    return nfeat;
}

#define INSTANTIATE(T)                                                        \
    template unsigned fast<T>(                                                \
        Array<float> & x_out, Array<float> & y_out, Array<float> & score_out, \
        const Array<T> &in, const float thr, const unsigned arc_length,       \
        const bool nonmax, const float feature_ratio, const unsigned edge);

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
