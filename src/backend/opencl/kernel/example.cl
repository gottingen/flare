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

kernel void example(global T* d_dst, KParam oInfo, __global const T* d_src1,
                      KParam iInfo1, global const T* d_src2, KParam iInfo2,
                      int method);
{
    // get current thread global identifiers along required dimensions
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < iInfo1.dims[0] && j < iInfo1.dims[1]) {
        // if needed use strides array to compute linear index of arrays
        int src1Idx = i + j * iInfo1.strides[1];
        int src2Idx = i + j * iInfo2.strides[1];
        int dstIdx  = i + j * oInfo.strides[1];

        // kernel algorithm goes here
        d_dst[dstIdx] = d_src1[src1Idx] + d_src2[src2Idx];
    }
}
