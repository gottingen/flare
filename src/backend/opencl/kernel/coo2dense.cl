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

kernel void coo2Dense(global T *oPtr, const KParam output, global const T *vPtr,
                      const KParam values, global const int *rPtr,
                      const KParam rowIdx, global const int *cPtr,
                      const KParam colIdx) {
    const int id = get_group_id(0) * get_local_size(0) * reps + get_local_id(0);

    if (id >= values.dims[0]) return;

    const int dimSize = get_local_size(0);

    for (int i = get_local_id(0); i < reps * dimSize; i += dimSize) {
        if (i >= values.dims[0]) return;

        T v   = vPtr[i];
        int r = rPtr[i];
        int c = cPtr[i];

        int offset = r + c * output.strides[1];

        oPtr[offset] = v;
    }
}
