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
#include <flare/runtime/cuda/algorithm/reduce.h>

// ----------------------------------------------------------------------------
// cuda_reduce_bufsz
// ----------------------------------------------------------------------------

TEST_CASE("cuda_reduce.BufferSize") {

  using P = flare::rt::cudaExecutionPolicy<32, 3>;
  
  // within one block
  for(unsigned i=0; i<=P::nv; i++) {
    REQUIRE(P::reduce_bufsz<int>(i) == 0);
  }

  // two blocks
  for(unsigned i=P::nv+1; i<=2*P::nv; i++) {
    REQUIRE(P::reduce_bufsz<int>(i) == 2*sizeof(int));
  }
  
  // three blocks
  for(unsigned i=2*P::nv+1; i<=3*P::nv; i++) {
    REQUIRE(P::reduce_bufsz<int>(i) == 3*sizeof(int));
  }

  REQUIRE(
    P::reduce_bufsz<int>(P::nv*P::nv) == P::nv*sizeof(int)
  );

  REQUIRE(
    P::reduce_bufsz<int>(P::nv*P::nv+1) == (P::nv + 3)*sizeof(int)
  );

  REQUIRE(
    P::reduce_bufsz<int>(P::nv*P::nv*2) == (2*P::nv + 2)*sizeof(int)
  );
  
}

// ----------------------------------------------------------------------------
// cuda_reduce
// ----------------------------------------------------------------------------

template <typename T>
void cuda_reduce() {

  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor;
  
  for(int n=0; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {
    taskflow.emplace([n](){
      flare::rt::cudaStream stream;
      flare::rt::cudaDefaultExecutionPolicy policy(stream);
      
      unsigned bufsz = policy.reduce_bufsz<T>(n);

      T gold {1000};

      auto gpu = flare::rt::cuda_malloc_shared<T>(n);
      auto res = flare::rt::cuda_malloc_shared<T>(1);
      auto buf = flare::rt::cuda_malloc_shared<T>(bufsz);
      for(int i=0; i<n; i++) {
        gpu[i] = i;
        gold += i;
      }
      *res = T{1000};  // initial value
      
      // reduce
      flare::rt::cuda_reduce(policy,
        gpu, gpu + n, res, [] __device__ (T a, T b) { return a + b; }, buf
      );
      stream.synchronize();

      REQUIRE(*res == gold);
      
      // uninitialized reduce
      flare::rt::cuda_uninitialized_reduce(policy,
        gpu, gpu + n, res, [] __device__ (T a, T b) { return a + b; }, buf
      );
      stream.synchronize();
      
      if(n == 0) {
        REQUIRE(*res == 1000);
      }
      else {
        REQUIRE(*res == gold - 1000);
      }

      REQUIRE(cudaFree(gpu) == cudaSuccess);
      REQUIRE(cudaFree(res) == cudaSuccess);
      REQUIRE(cudaFree(buf) == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_reduce.int" * doctest::timeout(300)) {
  cuda_reduce<int>();
}

// ----------------------------------------------------------------------------
// cuda_transform_reduce
// ----------------------------------------------------------------------------

template <typename T>
void cuda_transform_reduce() {

  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor;
  
  for(int n=0; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {
    taskflow.emplace([n](){
      flare::rt::cudaStream stream;
      flare::rt::cudaDefaultExecutionPolicy policy(stream);
      
      unsigned bufsz = policy.reduce_bufsz<T>(n);

      T gold {1000};

      auto gpu = flare::rt::cuda_malloc_shared<T>(n);
      auto res = flare::rt::cuda_malloc_shared<T>(1);
      auto buf = flare::rt::cuda_malloc_shared<T>(bufsz);
      for(int i=0; i<n; i++) {
        gpu[i] = i;
        gold += (-i);
      }
      *res = T{1000};  // initial value
      
      // reduce
      flare::rt::cuda_transform_reduce(policy,
        gpu, gpu + n, res, 
        [] __device__ (T a, T b) { return a + b; }, 
        [] __device__ (T a)      { return -a; }, 
        buf
      );
      stream.synchronize();

      REQUIRE(*res == gold);
      
      // uninitialized reduce
      flare::rt::cuda_uninitialized_transform_reduce(policy,
        gpu, gpu + n, res, 
        [] __device__ (T a, T b) { return a + b; }, 
        [] __device__ (T a)      { return -a; },
        buf
      );
      stream.synchronize();
      
      if(n == 0) {
        REQUIRE(*res == 1000);
      }
      else {
        REQUIRE(*res == gold - 1000);
      }

      REQUIRE(cudaFree(gpu) == cudaSuccess);
      REQUIRE(cudaFree(res) == cudaSuccess);
      REQUIRE(cudaFree(buf) == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_transform_reduce.int" * doctest::timeout(300)) {
  cuda_transform_reduce<int>();
}







