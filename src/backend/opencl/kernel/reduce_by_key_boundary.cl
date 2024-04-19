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

kernel void final_boundary_reduce(global int *reduced_block_sizes,
                                  global Tk *oKeys, KParam oKInfo,
                                  global To *oVals, KParam oVInfo,
                                  const int n) {
    const uint lid = get_local_id(0);
    const uint bid = get_group_id(0);
    const uint gid = get_global_id(0);

    if (gid == ((bid + 1) * get_local_size(0)) - 1 &&
        bid < get_num_groups(0) - 1) {
        Tk k0 = oKeys[gid];
        Tk k1 = oKeys[gid + 1];
        if (k0 == k1) {
            To v0                    = oVals[gid];
            To v1                    = oVals[gid + 1];
            oVals[gid + 1]           = binOp(v0, v1);
            reduced_block_sizes[bid] = get_local_size(0) - 1;
        } else {
            reduced_block_sizes[bid] = get_local_size(0);
        }
    }

    // if last block, set block size to difference between n and block boundary
    if (lid == 0 && bid == get_num_groups(0) - 1) {
        reduced_block_sizes[bid] = n - (bid * get_local_size(0));
    }
}
