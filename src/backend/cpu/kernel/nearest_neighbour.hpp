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

namespace flare {
namespace cpu {
namespace kernel {

#if defined(_WIN32) || defined(_MSC_VER)

#include <intrin.h>
#define __builtin_popcount __popcnt
#define __builtin_popcountll __popcnt64

#endif

template<typename T, typename To, fly_match_type dist_type>
struct dist_op {
    To operator()(T v1, T v2) {
        return v1 - v2;  // Garbage distance
    }
};

template<typename T, typename To>
struct dist_op<T, To, FLY_SAD> {
    To operator()(T v1, T v2) { return std::abs((double)v1 - (double)v2); }
};

template<typename T, typename To>
struct dist_op<T, To, FLY_SSD> {
    To operator()(T v1, T v2) { return (v1 - v2) * (v1 - v2); }
};

template<typename To>
struct dist_op<uint, To, FLY_SHD> {
    To operator()(uint v1, uint v2) { return __builtin_popcount(v1 ^ v2); }
};

template<typename To>
struct dist_op<uintl, To, FLY_SHD> {
    To operator()(uintl v1, uintl v2) { return __builtin_popcountll(v1 ^ v2); }
};

template<typename To>
struct dist_op<uchar, To, FLY_SHD> {
    To operator()(uchar v1, uchar v2) { return __builtin_popcount(v1 ^ v2); }
};

template<typename To>
struct dist_op<ushort, To, FLY_SHD> {
    To operator()(ushort v1, ushort v2) { return __builtin_popcount(v1 ^ v2); }
};

template<typename T, typename To, fly_match_type dist_type>
void nearest_neighbour(Param<To> dists, CParam<T> query, CParam<T> train,
                       const uint dist_dim) {
    uint sample_dim  = (dist_dim == 0) ? 1 : 0;
    const dim4 qDims = query.dims();
    const dim4 tDims = train.dims();

    const unsigned distLength = qDims[dist_dim];
    const unsigned nQuery     = qDims[sample_dim];
    const unsigned nTrain     = tDims[sample_dim];

    const T* qPtr = query.get();
    const T* tPtr = train.get();
    To* dPtr      = dists.get();

    dist_op<T, To, dist_type> op;

    for (unsigned i = 0; i < nQuery; i++) {
        for (unsigned j = 0; j < nTrain; j++) {
            To local_dist = 0;
            for (unsigned k = 0; k < distLength; k++) {
                size_t qIdx, tIdx;
                if (sample_dim == 0) {
                    qIdx = k * qDims[0] + i;
                    tIdx = k * tDims[0] + j;
                } else {
                    qIdx = i * qDims[0] + k;
                    tIdx = j * tDims[0] + k;
                }

                local_dist += op(qPtr[qIdx], tPtr[tIdx]);
            }

            dPtr[i * nTrain + j] = local_dist;
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace flare