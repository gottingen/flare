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
// This program demonstrates how to perform parallel sort with CUDA.

#include <flare/runtime/cuda/cudaflow.h>
#include <flare/runtime/cuda/algorithm/sort.h>

int main(int argc, char* argv[]) {
  
  if(argc != 2) {
    std::cerr << "usage: ./cuda_sort N\n";
    std::exit(EXIT_FAILURE);
  }

  unsigned N = std::atoi(argv[1]);

  // gpu data
  auto d_keys = flare::rt::cuda_malloc_shared<int>(N);

  // cpu data
  std::vector<int> h_keys(N);

  for(unsigned i=0; i<N; i++) {
    int k = rand() % 10000;
    d_keys[i] = k;
    h_keys[i] = k;
  }
  
  // --------------------------------------------------------------------------
  // Standard GPU sort
  // --------------------------------------------------------------------------

  auto p = flare::rt::cudaDefaultExecutionPolicy{};
  
  auto beg = std::chrono::steady_clock::now();
  flare::rt::cudaStream s;
  auto bufsz = flare::rt::cuda_sort_buffer_size<decltype(p), int>(N);
  flare::rt::cudaDeviceVector<std::byte> buf(bufsz);
  flare::rt::cuda_sort(p, d_keys, d_keys+N, flare::rt::cuda_less<int>{}, buf.data());
  s.synchronize();
  auto end = std::chrono::steady_clock::now();

  std::cout << "GPU sort: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(end-beg).count()
            << " us\n";
  
  // --------------------------------------------------------------------------
  // CPU sort
  // --------------------------------------------------------------------------
  beg = std::chrono::steady_clock::now();
  std::sort(h_keys.begin(), h_keys.end());
  end = std::chrono::steady_clock::now();
  
  std::cout << "CPU sort: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(end-beg).count()
            << " us\n";

  // --------------------------------------------------------------------------
  // verify the result
  // --------------------------------------------------------------------------
  
  for(unsigned i=0; i<N; i++) {
    if(d_keys[i] != h_keys[i]) {
      throw std::runtime_error("incorrect result");
    }
  }

  std::cout << "correct result\n";
};
