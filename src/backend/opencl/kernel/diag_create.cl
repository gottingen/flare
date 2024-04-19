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

kernel void diagCreateKernel(global T *oData, KParam oInfo,
                               const global T *iData, KParam iInfo, int num,
                               int groups_x) {
    unsigned idz       = get_group_id(0) / groups_x;
    unsigned groupId_x = get_group_id(0) - idz * groups_x;

    unsigned idx = get_local_id(0) + groupId_x * get_local_size(0);
    unsigned idy = get_global_id(1);

    if (idx >= oInfo.dims[0] || idy >= oInfo.dims[1] || idz >= oInfo.dims[2])
        return;

    global T *optr =
        oData + idz * oInfo.strides[2] + idy * oInfo.strides[1] + idx;
    const global T *iptr =
        iData + idz * iInfo.strides[1] + ((num > 0) ? idx : idy) + iInfo.offset;

    T val = (idx == (idy - num)) ? *iptr : (T)(ZERO);
    *optr = val;
}
