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

kernel void compact(global int *reduced_block_sizes, global Tk *oKeys,
                    KParam oKInfo, global To *oVals, KParam oVInfo,
                    const global Tk *iKeys, KParam iKInfo,
                    const global To *iVals, KParam iVInfo, const int nBlocksZ) {
    const uint lid = get_local_id(0);
    const uint bid = get_group_id(0);
    const uint gid = get_global_id(0);

    const int bidy = get_group_id(1);
    const int bidz = get_group_id(2) % nBlocksZ;
    const int bidw = get_group_id(2) / nBlocksZ;

    Tk k;
    To v;

    const int bOffset = bidw * oVInfo.strides[3] + bidz * oVInfo.strides[2] +
                        bidy * oVInfo.strides[1];

    // reduced_block_sizes should have inclusive sum of block sizes
    int nwrite =
        (bid == 0) ? reduced_block_sizes[0]
                   : (reduced_block_sizes[bid] - reduced_block_sizes[bid - 1]);
    int writeloc = (bid == 0) ? 0 : reduced_block_sizes[bid - 1];

    k = iKeys[gid];
    v = iVals[bOffset + gid];

    if (lid < nwrite) {
        oKeys[writeloc + lid]           = k;
        oVals[bOffset + writeloc + lid] = v;
    }
}
