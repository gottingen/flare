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

kernel void sparseArithCSR(global T *oPtr, const KParam out,
                           global const T *values, global const int *rowIdx,
                           global const int *colIdx, const int nNZ,
                           global const T *rPtr, const KParam rhs,
                           const int reverse) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);

    if (row >= out.dims[0]) return;

    const int rowStartIdx = rowIdx[row];
    const int rowEndIdx   = rowIdx[row + 1];

    // Repeat loop until all values in the row are computed
    for (int idx = rowStartIdx + get_local_id(0); idx < rowEndIdx;
         idx += get_local_size(0)) {
        const int col = colIdx[idx];

        if (row >= out.dims[0] || col >= out.dims[1]) continue;  // Bad indices

        // Get Values
        const T val  = values[idx];
        const T rval = rPtr[col * rhs.strides[1] + row];

        const int offset = col * out.strides[1] + row;
        if (reverse)
            oPtr[offset] = OP(rval, val);
        else
            oPtr[offset] = OP(val, rval);
    }
}

kernel void sparseArithCSR2(global T *values, global int *rowIdx,
                            global int *colIdx, const int nNZ,
                            global const T *rPtr, const KParam rhs,
                            const int reverse) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);

    if (row >= rhs.dims[0]) return;

    const int rowStartIdx = rowIdx[row];
    const int rowEndIdx   = rowIdx[row + 1];

    // Repeat loop until all values in the row are computed
    for (int idx = rowStartIdx + get_local_id(0); idx < rowEndIdx;
         idx += get_local_size(0)) {
        const int col = colIdx[idx];

        if (row >= rhs.dims[0] || col >= rhs.dims[1]) continue;  // Bad indices

        // Get Values
        const T val  = values[idx];
        const T rval = rPtr[col * rhs.strides[1] + row];

        if (reverse)
            values[idx] = OP(rval, val);
        else
            values[idx] = OP(val, rval);
    }
}
