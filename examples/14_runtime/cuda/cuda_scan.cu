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
// This program demonstrate how to perform a parallel scan
// using cudaFlow.

#include <flare/runtime/cuda/cudaflow.h>
#include <flare/runtime/cuda/algorithm/scan.h>

int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./cuda_scan N\n";
    std::exit(EXIT_FAILURE);
  }

  int N = std::atoi(argv[1]);

  auto data1 = flare::rt::cuda_malloc_shared<int>(N);
  auto data2 = flare::rt::cuda_malloc_shared<int>(N);
  auto scan1 = flare::rt::cuda_malloc_shared<int>(N);
  auto scan2 = flare::rt::cuda_malloc_shared<int>(N);

  // --------------------------------------------------------------------------
  // inclusive/exclusive scan
  // --------------------------------------------------------------------------

  // initialize the data
  std::iota(data1, data1 + N, 0);
  std::iota(data2, data2 + N, 0);
  
  flare::rt::cudaStream stream;
  flare::rt::cudaDefaultExecutionPolicy policy(stream);

  // declare the buffer
  void* buff;
  cudaMalloc(&buff, policy.scan_bufsz<int>(N));
  
  // create inclusive and exclusive scan tasks
  flare::rt::cuda_inclusive_scan(policy, data1, data1+N, scan1, flare::rt::cuda_plus<int>{}, buff);
  flare::rt::cuda_exclusive_scan(policy, data2, data2+N, scan2, flare::rt::cuda_plus<int>{}, buff);

  stream.synchronize();
  
  // inspect 
  for(int i=1; i<N; i++) {
    if(scan1[i] != scan1[i-1] + data1[i]) {
      throw std::runtime_error("incorrect inclusive scan result");
    }
    if(scan2[i] != scan2[i-1] + data2[i-1]) {
      throw std::runtime_error("incorrect exclusive scan result");
    }
  }

  std::cout << "scan done\n";
  
  // --------------------------------------------------------------------------
  // transform inclusive/exclusive scan
  // --------------------------------------------------------------------------
  
  // initialize the data
  std::iota(data1, data1 + N, 0);
  std::iota(data2, data2 + N, 0);
  
  // transform inclusive scan
  flare::rt::cuda_transform_inclusive_scan(policy,
    data1, data1+N, scan1, flare::rt::cuda_plus<int>{},
    [] __device__ (int a) { return a*10; },
    buff
  );

  // transform exclusive scan
  flare::rt::cuda_transform_exclusive_scan(policy,
    data2, data2+N, scan2, flare::rt::cuda_plus<int>{},
    [] __device__ (int a) { return a*11; },
    buff
  );
  
  stream.synchronize();
  
  // inspect 
  for(int i=1; i<N; i++) {
    if(scan1[i] != scan1[i-1] + data1[i] * 10) {
      throw std::runtime_error("incorrect transform inclusive scan result");
    }
    if(scan2[i] != scan2[i-1] + data2[i-1] * 11) {
      throw std::runtime_error("incorrect transform exclusive scan result");
    }
  }

  std::cout << "transform scan done - all results are correct\n";
  
  // deallocate the data
  cudaFree(data1);
  cudaFree(data2);
  cudaFree(scan1);
  cudaFree(scan2);
  cudaFree(buff);

  return 0;
}


