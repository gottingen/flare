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

kernel void csr2Coo(global int *orowidx, global int *ocolidx,
                    global const int *irowidx, global const int *icolidx,
                    const int M) {
    int lid = get_local_id(0);
    for (int rowId = get_group_id(0); rowId < M; rowId += get_num_groups(0)) {
        int colStart = irowidx[rowId];
        int colEnd   = irowidx[rowId + 1];
        for (int colId = colStart + lid; colId < colEnd;
             colId += get_local_size(0)) {
            orowidx[colId] = rowId;
            ocolidx[colId] = icolidx[colId];
        }
    }
}

kernel void swapIndex(global T *ovalues, global int *oindex,
                      global const T *ivalues, global const int *iindex,
                      global const int *swapIdx, const int nNZ) {
    int id = get_global_id(0);
    if (id >= nNZ) return;

    int idx = swapIdx[id];

    ovalues[id] = ivalues[idx];
    oindex[id]  = iindex[idx];
}

kernel void csrReduce(global int *orowIdx, global const int *irowIdx,
                      const int M, const int nNZ) {
    int id = get_global_id(0);

    if (id >= nNZ) return;

    // Read COO row indices
    int iRId  = irowIdx[id];
    int iRId1 = 0;
    if (id > 0) iRId1 = irowIdx[id - 1];

    // If id is 0, then mark the edge cases of csrRow[0] and csrRow[M]
    if (id == 0) {
        orowIdx[id] = 0;
        orowIdx[M]  = nNZ;
    } else if (iRId1 != iRId) {
        // If iRId1 and iRId are not same, that means the row has incremented
        // For example, if iRId is 5 and iRId1 is 4, that means row 4 has
        // ended and row 5 has begun at index id.
        // We use the for-loop because there can be any number of empty rows
        // between iRId1 and iRId, all of which should be marked by id
        for (int i = iRId1 + 1; i <= iRId; i++) orowIdx[i] = id;
    }

    // The last X rows are corner cases if they dont have any values
    if (id < M) {
        if (id > irowIdx[nNZ - 1] && orowIdx[id] == 0) { orowIdx[id] = nNZ; }
    }
}
