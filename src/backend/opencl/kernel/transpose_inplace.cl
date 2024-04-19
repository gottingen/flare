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
#if DOCONJUGATE
T doOp(T in) {
    T out = {in.x, -in.y};
    return out;
}
#else
#define doOp(in) in
#endif

kernel void transpose_inplace(global T *iData, const KParam in,
                                const int blocksPerMatX,
                                const int blocksPerMatY) {
    local T shrdMem_s[TILE_DIM * (TILE_DIM + 1)];
    local T shrdMem_d[TILE_DIM * (TILE_DIM + 1)];

    const int shrdStride = TILE_DIM + 1;

    // create variables to hold output dimensions
    const int iDim0 = in.dims[0];
    const int iDim1 = in.dims[1];

    // calculate strides
    const int iStride1 = in.strides[1];

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // batch based block Id
    const int batchId_x  = get_group_id(0) / blocksPerMatX;
    const int blockIdx_x = (get_group_id(0) - batchId_x * blocksPerMatX);

    const int batchId_y  = get_group_id(1) / blocksPerMatY;
    const int blockIdx_y = (get_group_id(1) - batchId_y * blocksPerMatY);

    const int x0 = TILE_DIM * blockIdx_x;
    const int y0 = TILE_DIM * blockIdx_y;

    global T *iptr = iData + batchId_x * in.strides[2] +
                       batchId_y * in.strides[3] + in.offset;

    if (blockIdx_y > blockIdx_x) {
        // calculate global indices
        int gx = lx + x0;
        int gy = ly + y0;
        int dx = lx + y0;
        int dy = ly + x0;

        // Copy to shared memory
        for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
            int gy_ = gy + repeat;
            if (IS32MULTIPLE || (gx < iDim0 && gy_ < iDim1))
                shrdMem_s[(ly + repeat) * shrdStride + lx] =
                    iptr[gy_ * iStride1 + gx];

            int dy_ = dy + repeat;
            if (IS32MULTIPLE || (dx < iDim0 && dy_ < iDim1))
                shrdMem_d[(ly + repeat) * shrdStride + lx] =
                    iptr[dy_ * iStride1 + dx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Copy from shared memory to global memory
        for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
            int dy_ = dy + repeat;
            if (IS32MULTIPLE || (dx < iDim0 && dy_ < iDim1))
                iptr[dy_ * iStride1 + dx] =
                    doOp(shrdMem_s[(ly + repeat) + (shrdStride * lx)]);

            int gy_ = gy + repeat;
            if (IS32MULTIPLE || (gx < iDim0 && gy_ < iDim1))
                iptr[gy_ * iStride1 + gx] =
                    doOp(shrdMem_d[(ly + repeat) + (shrdStride * lx)]);
        }

    } else if (blockIdx_y == blockIdx_x) {
        // calculate global indices
        int gx = lx + x0;
        int gy = ly + y0;

        // Copy to shared memory
        for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
            int gy_ = gy + repeat;
            if (IS32MULTIPLE || (gx < iDim0 && gy_ < iDim1))
                shrdMem_s[(ly + repeat) * shrdStride + lx] =
                    iptr[gy_ * iStride1 + gx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Copy from shared memory to global memory
        for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
            int gy_ = gy + repeat;
            if (IS32MULTIPLE || (gx < iDim0 && gy_ < iDim1))
                iptr[gy_ * iStride1 + gx] =
                    doOp(shrdMem_s[(ly + repeat) + (shrdStride * lx)]);
        }
    }
}
