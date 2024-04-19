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
#include <err_opencl.hpp>
#include <kernel/nearest_neighbour.hpp>
#include <math.hpp>
#include <topk.hpp>
#include <transpose.hpp>
#include <fly/dim4.hpp>

using fly::dim4;
using cl::Device;

namespace flare {
namespace opencl {

template<typename T, typename To, fly_match_type dist_type>
void nearest_neighbour_(Array<uint>& idx, Array<To>& dist,
                        const Array<T>& query, const Array<T>& train,
                        const uint dist_dim, const uint n_dist) {
    uint sample_dim   = (dist_dim == 0) ? 1 : 0;
    const dim4& qDims = query.dims();
    const dim4& tDims = train.dims();

    const dim4 outDims(n_dist, qDims[sample_dim]);
    const dim4 distDims(tDims[sample_dim], qDims[sample_dim]);

    Array<To> tmp_dists = createEmptyArray<To>(distDims);

    idx  = createEmptyArray<uint>(outDims);
    dist = createEmptyArray<To>(outDims);

    Array<T> queryT = dist_dim == 0 ? transpose(query, false) : query;
    Array<T> trainT = dist_dim == 0 ? transpose(train, false) : train;

    kernel::allDistances<T, To>(tmp_dists, queryT, trainT, 1, dist_type);

    topk(dist, idx, tmp_dists, n_dist, 0, FLY_TOPK_MIN);
}

template<typename T, typename To>
void nearest_neighbour(Array<uint>& idx, Array<To>& dist, const Array<T>& query,
                       const Array<T>& train, const uint dist_dim,
                       const uint n_dist, const fly_match_type dist_type) {
    switch (dist_type) {
        case FLY_SAD:
            nearest_neighbour_<T, To, FLY_SAD>(idx, dist, query, train, dist_dim,
                                              n_dist);
            break;
        case FLY_SSD:
            nearest_neighbour_<T, To, FLY_SSD>(idx, dist, query, train, dist_dim,
                                              n_dist);
            break;
        case FLY_SHD:
            nearest_neighbour_<T, To, FLY_SHD>(idx, dist, query, train, dist_dim,
                                              n_dist);
            break;
        default: FLY_ERROR("Unsupported dist_type", FLY_ERR_NOT_CONFIGURED);
    }
}

#define INSTANTIATE(T, To)                                             \
    template void nearest_neighbour<T, To>(                            \
        Array<uint> & idx, Array<To> & dist, const Array<T>& query,    \
        const Array<T>& train, const uint dist_dim, const uint n_dist, \
        const fly_match_type dist_type);

INSTANTIATE(float, float)
INSTANTIATE(double, double)
INSTANTIATE(int, int)
INSTANTIATE(uint, uint)
INSTANTIATE(intl, intl)
INSTANTIATE(uintl, uintl)
INSTANTIATE(short, int)
INSTANTIATE(ushort, uint)
INSTANTIATE(uchar, uint)

INSTANTIATE(uintl, uint)  // For Hamming

}  // namespace opencl
}  // namespace flare
