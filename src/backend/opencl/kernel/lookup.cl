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

int trimIndex(int idx, const int len) {
    int ret_val = idx;
    if (ret_val < 0) {
        int offset = (abs(ret_val) - 1) % len;
        ret_val    = offset;
    } else if (ret_val >= len) {
        int offset = abs(ret_val) % len;
        ret_val    = len - offset - 1;
    }
    return ret_val;
}

kernel void lookupND(global in_t *out, KParam oInfo, global const in_t *in,
                     KParam iInfo, global const idx_t *indices, KParam idxInfo,
                     int nBBS0, int nBBS1) {
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    int gz = get_group_id(0) / nBBS0;
    int gw = get_group_id(1) / nBBS1;

    int gx = get_local_size(0) * (get_group_id(0) - gz * nBBS0) + lx;
    int gy = get_local_size(1) * (get_group_id(1) - gw * nBBS1) + ly;

    global const idx_t *idxPtr = indices;

    int i = iInfo.strides[0] *
            (DIM == 0 ? trimIndex((int)idxPtr[gx], iInfo.dims[0]) : gx);
    int j = iInfo.strides[1] *
            (DIM == 1 ? trimIndex((int)idxPtr[gy], iInfo.dims[1]) : gy);
    int k = iInfo.strides[2] *
            (DIM == 2 ? trimIndex((int)idxPtr[gz], iInfo.dims[2]) : gz);
    int l = iInfo.strides[3] *
            (DIM == 3 ? trimIndex((int)idxPtr[gw], iInfo.dims[3]) : gw);

    global const in_t *inPtr = in + (i + j + k + l) + iInfo.offset;
    global in_t *outPtr =
        out + (gx * oInfo.strides[0] + gy * oInfo.strides[1] +
               gz * oInfo.strides[2] + gw * oInfo.strides[3] + oInfo.offset);

    if (gx < oInfo.dims[0] && gy < oInfo.dims[1] && gz < oInfo.dims[2] &&
        gw < oInfo.dims[3]) {
        outPtr[0] = inPtr[0];
    }
}
