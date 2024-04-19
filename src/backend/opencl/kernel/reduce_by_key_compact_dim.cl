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

kernel void compact_dim(global int *reduced_block_sizes, global Tk *oKeys,
                        KParam oKInfo, global To *oVals, KParam oVInfo,
                        const global Tk *iKeys, KParam iKInfo,
                        const global To *iVals, KParam iVInfo,
                        const int nBlocksZ) {
    local int dim_ordering[4];
    const uint lid  = get_local_id(0);
    const uint bid  = get_group_id(0);
    const uint gidx = get_global_id(0);

    const int bidy = get_group_id(1);
    const int bidz = get_group_id(2) % nBlocksZ;
    const int bidw = get_group_id(2) / nBlocksZ;

    if (lid == 0) {
        int d           = 1;
        dim_ordering[0] = DIM;
        for (int i = 0; i < 4; ++i) {
            if (i != DIM) dim_ordering[d++] = i;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    Tk k;
    To v;

    // reduced_block_sizes should have inclusive sum of block sizes
    int nwrite =
        (bid == 0) ? reduced_block_sizes[0]
                   : (reduced_block_sizes[bid] - reduced_block_sizes[bid - 1]);
    int writeloc = (bid == 0) ? 0 : reduced_block_sizes[bid - 1];

    const int tid = bidw * iVInfo.strides[dim_ordering[3]] +
                    bidz * iVInfo.strides[dim_ordering[2]] +
                    bidy * iVInfo.strides[dim_ordering[1]] +
                    gidx * iVInfo.strides[DIM];
    k = iKeys[gidx];
    v = iVals[tid];

    if (lid < nwrite) {
        oKeys[writeloc + lid] = k;
        const int bOffset     = bidw * oVInfo.strides[dim_ordering[3]] +
                            bidz * oVInfo.strides[dim_ordering[2]] +
                            bidy * oVInfo.strides[dim_ordering[1]];
        oVals[bOffset + (writeloc + lid) * oVInfo.strides[DIM]] = v;
    }
}
