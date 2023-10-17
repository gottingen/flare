// Copyright 2023 The Elastic-AI Authors.
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
#ifndef FLARE_RUNTIME_CUDA_ALGORITHM_TRANSPOSE_H_
#define FLARE_RUNTIME_CUDA_ALGORITHM_TRANSPOSE_H_

#include <flare/runtime/cuda/cuda_error.h>

namespace flare::rt {

// ----------------------------------------------------------------------------
// row-wise matrix transpose
// ----------------------------------------------------------------------------
//
template <typename T>
__global__ void cuda_transpose(
  const T* d_in,
  T* d_out,
  size_t rows,
  size_t cols
) {
  __shared__ T tile[32][32];
  size_t x = blockIdx.x * 32 + threadIdx.x;
  size_t y = blockIdx.y * 32 + threadIdx.y;

  for(size_t i = 0; i < 32; i += 8) {
    if(x < cols && (y + i) < rows) {
      tile[threadIdx.y + i][threadIdx.x] = d_in[(y + i) * cols + x];
    }
  }

  __syncthreads();

  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;

  for(size_t i = 0; i < 32; i += 8) {
    if(x < rows && (y + i) < cols) {
      d_out[(y + i) * rows + x] = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}

}  // namespace flare::rt
#endif  // FLARE_RUNTIME_CUDA_ALGORITHM_TRANSPOSE_H_
