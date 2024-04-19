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

void diff_this(global T* out, __global const T* in, const int oMem,
               const int iMem0, const int iMem1, const int iMem2) {
    if (isDiff2 == 0) {
        out[oMem] = in[iMem1] - in[iMem0];
    } else {
        out[oMem] = in[iMem2] - in[iMem1] - in[iMem1] + in[iMem0];
    }
}

kernel void diff_kernel(global T* out, __global const T* in,
                          const KParam op, const KParam ip, const int oElem,
                          const int blocksPerMatX, const int blocksPerMatY) {
    const int idz = get_group_id(0) / blocksPerMatX;
    const int idw = get_group_id(1) / blocksPerMatY;

    const int blockIdx_x = get_group_id(0) - idz * blocksPerMatX;
    const int blockIdx_y = get_group_id(1) - idw * blocksPerMatY;

    const int idx = get_local_id(0) + blockIdx_x * get_local_size(0);
    const int idy = get_local_id(1) + blockIdx_y * get_local_size(1);

    if (idx >= op.dims[0] || idy >= op.dims[1] || idz >= op.dims[2] ||
        idw >= op.dims[3])
        return;

    int iMem0 =
        idw * ip.strides[3] + idz * ip.strides[2] + idy * ip.strides[1] + idx;
    int iMem1 = iMem0 + ip.strides[DIM];
    int iMem2 = iMem1 + ip.strides[DIM];

    int oMem =
        idw * op.strides[3] + idz * op.strides[2] + idy * op.strides[1] + idx;

    iMem2 *= isDiff2;

    diff_this(out, in + ip.offset, oMem, iMem0, iMem1, iMem2);
}
