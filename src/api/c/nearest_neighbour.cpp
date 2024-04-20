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
#include <handle.hpp>
#include <nearest_neighbour.hpp>
#include <fly/defines.h>
#include <fly/dim4.hpp>
#include <fly/vision.h>

using fly::dim4;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename Ti, typename To>
static void nearest_neighbour(fly_array* idx, fly_array* dist,
                              const fly_array query, const fly_array train,
                              const dim_t dist_dim, const uint n_dist,
                              const fly_match_type dist_type) {
    Array<uint> oIdxArray = createEmptyArray<uint>(fly::dim4());
    Array<To> oDistArray  = createEmptyArray<To>(fly::dim4());

    nearest_neighbour<Ti, To>(oIdxArray, oDistArray, getArray<Ti>(query),
                              getArray<Ti>(train), dist_dim, n_dist, dist_type);

    *idx  = getHandle<uint>(oIdxArray);
    *dist = getHandle<To>(oDistArray);
}

fly_err fly_nearest_neighbour(fly_array* idx, fly_array* dist, const fly_array query,
                            const fly_array train, const dim_t dist_dim,
                            const uint n_dist, const fly_match_type dist_type) {
    try {
        const ArrayInfo& qInfo = getInfo(query);
        const ArrayInfo& tInfo = getInfo(train);
        fly_dtype qType         = qInfo.getType();
        fly_dtype tType         = tInfo.getType();
        fly::dim4 qDims         = qInfo.dims();
        fly::dim4 tDims         = tInfo.dims();

        uint train_samples = (dist_dim == 0) ? 1 : 0;

        DIM_ASSERT(2, qDims[dist_dim] == tDims[dist_dim]);
        DIM_ASSERT(2, qDims[2] == 1 && qDims[3] == 1);
        DIM_ASSERT(3, tDims[2] == 1 && tDims[3] == 1);
        DIM_ASSERT(4, (dist_dim == 0 || dist_dim == 1));
        DIM_ASSERT(5, n_dist > 0 && n_dist <= (uint)tDims[train_samples]);
        ARG_ASSERT(5, n_dist > 0 && n_dist <= 256);
        ARG_ASSERT(6, dist_type == FLY_SAD || dist_type == FLY_SSD ||
                          dist_type == FLY_SHD);
        TYPE_ASSERT(qType == tType);

        // For Hamming, only u8, u16, u32 and u64 allowed.
        fly_array oIdx;
        fly_array oDist;

        if (dist_type == FLY_SHD) {
            TYPE_ASSERT(qType == u8 || qType == u16 || qType == u32 ||
                        qType == u64);
            switch (qType) {
                case u8:
                    nearest_neighbour<uchar, uint>(&oIdx, &oDist, query, train,
                                                   dist_dim, n_dist, FLY_SHD);
                    break;
                case u16:
                    nearest_neighbour<ushort, uint>(&oIdx, &oDist, query, train,
                                                    dist_dim, n_dist, FLY_SHD);
                    break;
                case u32:
                    nearest_neighbour<uint, uint>(&oIdx, &oDist, query, train,
                                                  dist_dim, n_dist, FLY_SHD);
                    break;
                case u64:
                    nearest_neighbour<uintl, uint>(&oIdx, &oDist, query, train,
                                                   dist_dim, n_dist, FLY_SHD);
                    break;
                default: TYPE_ERROR(1, qType);
            }
        } else {
            switch (qType) {
                case f32:
                    nearest_neighbour<float, float>(&oIdx, &oDist, query, train,
                                                    dist_dim, n_dist,
                                                    dist_type);
                    break;
                case f64:
                    nearest_neighbour<double, double>(&oIdx, &oDist, query,
                                                      train, dist_dim, n_dist,
                                                      dist_type);
                    break;
                case s32:
                    nearest_neighbour<int, int>(&oIdx, &oDist, query, train,
                                                dist_dim, n_dist, dist_type);
                    break;
                case u32:
                    nearest_neighbour<uint, uint>(&oIdx, &oDist, query, train,
                                                  dist_dim, n_dist, dist_type);
                    break;
                case s64:
                    nearest_neighbour<intl, intl>(&oIdx, &oDist, query, train,
                                                  dist_dim, n_dist, dist_type);
                    break;
                case u64:
                    nearest_neighbour<uintl, uintl>(&oIdx, &oDist, query, train,
                                                    dist_dim, n_dist,
                                                    dist_type);
                    break;
                case s16:
                    nearest_neighbour<short, int>(&oIdx, &oDist, query, train,
                                                  dist_dim, n_dist, dist_type);
                    break;
                case u16:
                    nearest_neighbour<ushort, uint>(&oIdx, &oDist, query, train,
                                                    dist_dim, n_dist,
                                                    dist_type);
                    break;
                case u8:
                    nearest_neighbour<uchar, uint>(&oIdx, &oDist, query, train,
                                                   dist_dim, n_dist, dist_type);
                    break;
                default: TYPE_ERROR(1, qType);
            }
        }
        std::swap(*idx, oIdx);
        std::swap(*dist, oDist);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
