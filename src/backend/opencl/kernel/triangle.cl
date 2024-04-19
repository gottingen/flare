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

kernel void triangle(global T *rptr, KParam rinfo, const global T *iptr,
                     KParam iinfo, const int groups_x, const int groups_y) {
    const int oz = get_group_id(0) / groups_x;
    const int ow = get_group_id(1) / groups_y;

    const int groupId_0 = get_group_id(0) - oz * groups_x;
    const int groupId_1 = get_group_id(1) - ow * groups_y;

    const int xx = get_local_id(0) + groupId_0 * get_local_size(0);
    const int yy = get_local_id(1) + groupId_1 * get_local_size(1);

    const int incy = groups_y * get_local_size(1);
    const int incx = groups_x * get_local_size(0);

    global T *d_r       = rptr;
    const global T *d_i = iptr + iinfo.offset;

    if (oz < rinfo.dims[2] && ow < rinfo.dims[3]) {
        d_i = d_i + oz * iinfo.strides[2] + ow * iinfo.strides[3];
        d_r = d_r + oz * rinfo.strides[2] + ow * rinfo.strides[3];

        for (int oy = yy; oy < rinfo.dims[1]; oy += incy) {
            const global T *Yd_i = d_i + oy * iinfo.strides[1];
            global T *Yd_r       = d_r + oy * rinfo.strides[1];

            for (int ox = xx; ox < rinfo.dims[0]; ox += incx) {
                bool cond         = is_upper ? (oy >= ox) : (oy <= ox);
                bool do_unit_diag = is_unit_diag && (oy == ox);
                if (cond) {
                    Yd_r[ox] = do_unit_diag ? (T)(ONE) : Yd_i[ox];
                } else {
                    Yd_r[ox] = (T)(ZERO);
                }
            }
        }
    }
}
