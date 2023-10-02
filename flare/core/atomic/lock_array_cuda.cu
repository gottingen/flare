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


#include <cinttypes>
#include <flare/core/atomic/lock_array.h>
#include <sstream>
#include <string>

namespace flare {

namespace {

__global__ void init_lock_arrays_cuda_kernel() {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < CUDA_SPACE_ATOMIC_MASK + 1) {
    detail::CUDA_SPACE_ATOMIC_LOCKS_DEVICE[i] = 0;
    detail::CUDA_SPACE_ATOMIC_LOCKS_NODE[i] = 0;
  }
}

}  // namespace

namespace detail {

int32_t* CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
int32_t* CUDA_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;

// Putting this into anonymous namespace so we don't have multiple defined symbols
// When linking in more than one copy of the object file
namespace {

void check_error_and_throw_cuda(cudaError e, const std::string msg) {
  if (e != cudaSuccess) {
    std::ostringstream out;
    out << "Desul::Error: " << msg << " error(" << cudaGetErrorName(e)
        << "): " << cudaGetErrorString(e);
    throw std::runtime_error(out.str());
  }
}

}  // namespace

// define functions
template <typename T>
void init_lock_arrays_cuda() {
  if (CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h != nullptr) return;
  auto error_malloc1 = cudaMalloc(&CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h,
                                  sizeof(int32_t) * (CUDA_SPACE_ATOMIC_MASK + 1));
  check_error_and_throw_cuda(error_malloc1,
                             "init_lock_arrays_cuda: cudaMalloc device locks");

  auto error_malloc2 = cudaMallocHost(&CUDA_SPACE_ATOMIC_LOCKS_NODE_h,
                                      sizeof(int32_t) * (CUDA_SPACE_ATOMIC_MASK + 1));
  check_error_and_throw_cuda(error_malloc2,
                             "init_lock_arrays_cuda: cudaMalloc host locks");

  auto error_sync1 = cudaDeviceSynchronize();
  copy_cuda_lock_arrays_to_device();
  check_error_and_throw_cuda(error_sync1, "init_lock_arrays_cuda: post mallocs");
  init_lock_arrays_cuda_kernel<<<(CUDA_SPACE_ATOMIC_MASK + 1 + 255) / 256, 256>>>();
  auto error_sync2 = cudaDeviceSynchronize();
  check_error_and_throw_cuda(error_sync2, "init_lock_arrays_cuda: post init kernel");
}

template <typename T>
void finalize_lock_arrays_cuda() {
  if (CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h == nullptr) return;
  cudaFree(CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h);
  cudaFreeHost(CUDA_SPACE_ATOMIC_LOCKS_NODE_h);
  CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h = nullptr;
  CUDA_SPACE_ATOMIC_LOCKS_NODE_h = nullptr;
}

// Instantiate functions
template void init_lock_arrays_cuda<int>();
template void finalize_lock_arrays_cuda<int>();

}  // namespace detail

}  // namespace flare
