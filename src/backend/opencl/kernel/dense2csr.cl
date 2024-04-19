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

#if IS_CPLX
#define IS_ZERO(val) ((val.x == 0) && (val.y == 0))
#else
#define IS_ZERO(val) (val == 0)
#endif

kernel void dense2Csr(global T *svalptr, global int *scolptr,
                      global const T *dvalptr, const KParam valinfo,
                      global const int *dcolptr, const KParam colinfo,
                      global const int *rowptr) {
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    if (gidx >= valinfo.dims[0]) return;
    if (gidy >= valinfo.dims[1]) return;

    int rowoff = rowptr[gidx];
    svalptr += rowoff;
    scolptr += rowoff;

    dvalptr += valinfo.offset;
    dcolptr += colinfo.offset;

    int idx = gidx + gidy * valinfo.strides[1];
    T val   = dvalptr[gidx + gidy * valinfo.strides[1]];
    if (IS_ZERO(val)) return;

    int oloc          = dcolptr[gidx + gidy * colinfo.strides[1]];
    svalptr[oloc - 1] = val;
    scolptr[oloc - 1] = gidy;
}
