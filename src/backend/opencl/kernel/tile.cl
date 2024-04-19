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

kernel void tile(global T *out, global const T *in, const KParam op,
                 const KParam ip, const int blocksPerMatX,
                 const int blocksPerMatY) {
    const int oz = get_group_id(0) / blocksPerMatX;
    const int ow = get_group_id(1) / blocksPerMatY;

    const int blockIdx_x = get_group_id(0) - oz * blocksPerMatX;
    const int blockIdx_y = get_group_id(1) - ow * blocksPerMatY;

    const int xx = get_local_id(0) + blockIdx_x * get_local_size(0);
    const int yy = get_local_id(1) + blockIdx_y * get_local_size(1);

    if (xx >= op.dims[0] || yy >= op.dims[1] || oz >= op.dims[2] ||
        ow >= op.dims[3])
        return;

    const int iz  = oz % ip.dims[2];
    const int iw  = ow % ip.dims[3];
    const int izw = iw * ip.strides[3] + iz * ip.strides[2];
    const int ozw = ow * op.strides[3] + oz * op.strides[2];

    const int incy = blocksPerMatY * get_local_size(1);
    const int incx = blocksPerMatX * get_local_size(0);

    for (int oy = yy; oy < op.dims[1]; oy += incy) {
        const int iy = oy % ip.dims[1];
        for (int ox = xx; ox < op.dims[0]; ox += incx) {
            const int ix = ox % ip.dims[0];

            int iMem = izw + iy * ip.strides[1] + ix;
            int oMem = ozw + oy * op.strides[1] + ox;

            out[oMem] = in[ip.offset + iMem];
        }
    }
}
