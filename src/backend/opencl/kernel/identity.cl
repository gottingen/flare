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

kernel void identity_kernel(global T *oData, KParam oInfo, int groups_x,
                            int groups_y) {
    unsigned idz = get_group_id(0) / groups_x;
    unsigned idw = get_group_id(1) / groups_y;

    unsigned groupId_x = get_group_id(0) - idz * groups_x;
    unsigned groupId_y = get_group_id(1) - idw * groups_y;

    unsigned idx = get_local_id(0) + groupId_x * get_local_size(0);
    unsigned idy = get_local_id(1) + groupId_y * get_local_size(1);

    if (idx >= oInfo.dims[0] || idy >= oInfo.dims[1] || idz >= oInfo.dims[2] ||
        idw >= oInfo.dims[3])
        return;

    global T *ptr = oData + idz * oInfo.strides[2] + idw * oInfo.strides[3];
    T val         = (idx == idy) ? (T)(ONE) : (T)(ZERO);
    ptr[idx + idy * oInfo.strides[1]] = val;
}
