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
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <flare/runtime/taskflow.h>
#include <flare/runtime/cuda/cudaflow.h>
#include <flare/runtime/cuda/algorithm/scan.h>

// ----------------------------------------------------------------------------
// cuda_scan
// ----------------------------------------------------------------------------

template <typename T>
void cuda_scan() {

  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor;
  
  for(int n=0; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {

    taskflow.emplace([n](){
  
      auto data1 = flare::rt::cuda_malloc_shared<int>(n);
      auto data2 = flare::rt::cuda_malloc_shared<int>(n);
      auto scan1 = flare::rt::cuda_malloc_shared<int>(n);
      auto scan2 = flare::rt::cuda_malloc_shared<int>(n);

      // --------------------------------------------------------------------------
      // inclusive/exclusive scan
      // --------------------------------------------------------------------------

      // initialize the data
      std::iota(data1, data1 + n, 0);
      std::iota(data2, data2 + n, 0);
      
      flare::rt::cudaStream stream;
      flare::rt::cudaDefaultExecutionPolicy policy(stream);

      // declare the buffer
      void* buff;
      cudaMalloc(&buff, policy.scan_bufsz<int>(n));
      
      // create inclusive and exclusive scan tasks
      flare::rt::cuda_inclusive_scan(policy, data1, data1+n, scan1, flare::rt::cuda_plus<int>{}, buff);
      flare::rt::cuda_exclusive_scan(policy, data2, data2+n, scan2, flare::rt::cuda_plus<int>{}, buff);

      stream.synchronize();
      
      // inspect 
      for(int i=1; i<n; i++) {
        REQUIRE(scan1[i] == (scan1[i-1] + data1[i]));
        REQUIRE(scan2[i] == (scan2[i-1] + data2[i-1]));
      }
  
      // deallocate the data
      REQUIRE(cudaFree(data1) == cudaSuccess);
      REQUIRE(cudaFree(data2) == cudaSuccess);
      REQUIRE(cudaFree(scan1) == cudaSuccess);
      REQUIRE(cudaFree(scan2) == cudaSuccess);
      REQUIRE(cudaFree(buff)  == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_scan.int" * doctest::timeout(300)) {
  cuda_scan<int>();
}

// ----------------------------------------------------------------------------
// transform_scan
// ----------------------------------------------------------------------------

template <typename T>
void cuda_transform_scan() {

  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor;
  
  for(int n=0; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {
  
    taskflow.emplace([n](){

      auto data1 = flare::rt::cuda_malloc_shared<int>(n);
      auto data2 = flare::rt::cuda_malloc_shared<int>(n);
      auto scan1 = flare::rt::cuda_malloc_shared<int>(n);
      auto scan2 = flare::rt::cuda_malloc_shared<int>(n);

      // --------------------------------------------------------------------------
      // inclusive/exclusive scan
      // --------------------------------------------------------------------------

      flare::rt::cudaStream stream;
      flare::rt::cudaDefaultExecutionPolicy policy(stream);

      // declare the buffer
      void* buff;
      cudaMalloc(&buff, policy.scan_bufsz<int>(n));
      
      // initialize the data
      std::iota(data1, data1 + n, 0);
      std::iota(data2, data2 + n, 0);
      
      // transform inclusive scan
      flare::rt::cuda_transform_inclusive_scan(policy,
        data1, data1+n, scan1, flare::rt::cuda_plus<int>{},
        [] __device__ (int a) { return a*10; },
        buff
      );

      // transform exclusive scan
      flare::rt::cuda_transform_exclusive_scan(policy,
        data2, data2+n, scan2, flare::rt::cuda_plus<int>{},
        [] __device__ (int a) { return a*11; },
        buff
      );
      
      stream.synchronize();
  
      // inspect 
      for(int i=1; i<n; i++) {
        REQUIRE(scan1[i] == scan1[i-1] + data1[i] * 10);
        REQUIRE(scan2[i] == scan2[i-1] + data2[i-1] * 11);
      }
  
      // deallocate the data
      REQUIRE(cudaFree(data1) == cudaSuccess);
      REQUIRE(cudaFree(data2) == cudaSuccess);
      REQUIRE(cudaFree(scan1) == cudaSuccess);
      REQUIRE(cudaFree(scan2) == cudaSuccess);
      REQUIRE(cudaFree(buff)  == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_transform_scan.int" * doctest::timeout(300)) {
  cuda_transform_scan<int>();
}
