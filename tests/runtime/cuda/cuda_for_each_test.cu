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
#include <flare/runtime/cuda/algorithm/for_each.h>

constexpr float eps = 0.0001f;

template <typename T>
void run_and_wait(T& cf) {
  flare::rt::cudaStream stream;
  cf.run(stream);
  stream.synchronize();
}

// ----------------------------------------------------------------------------
// for_each
// ----------------------------------------------------------------------------

template <typename T>
void cuda_for_each() {

  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor;
  
  for(int n=0; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {

    taskflow.emplace([n](){
      flare::rt::cudaStream stream;
      flare::rt::cudaDefaultExecutionPolicy policy(stream);

      auto g_data = flare::rt::cuda_malloc_shared<T>(n);
      for(int i=0; i<n; i++) {
        g_data[i] = 0;
      }

      flare::rt::cuda_for_each(policy,
        g_data, g_data + n, [] __device__ (T& val) { val = 12222; }
      );

      stream.synchronize();

      for(int i=0; i<n; i++) {
        REQUIRE(std::fabs(g_data[i] - (T)12222) < eps);
      }

      flare::rt::cuda_free(g_data);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_for_each.int" * doctest::timeout(300)) {
  cuda_for_each<int>();
}

TEST_CASE("cuda_for_each.float" * doctest::timeout(300)) {
  cuda_for_each<float>();
}

TEST_CASE("cuda_for_each.double" * doctest::timeout(300)) {
  cuda_for_each<double>();
}

// ----------------------------------------------------------------------------
// for_each
// ----------------------------------------------------------------------------

template <typename T, typename F>
void cudaflow_for_each() {
    
  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor;

  for(int n=1; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {
    
    taskflow.emplace([n](){

      auto cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
      
      T* gpu = nullptr;
      REQUIRE(cudaMalloc(&gpu, n*sizeof(T)) == cudaSuccess);

      F cf;
      auto d2h = cf.copy(cpu, gpu, n);
      auto h2d = cf.copy(gpu, cpu, n);
      auto kernel = cf.for_each(
        gpu, gpu+n, [] __device__ (T& val) { val = 65536; }
      );
      h2d.precede(kernel);
      d2h.succeed(kernel);

      run_and_wait(cf);

      for(int i=0; i<n; i++) {
        REQUIRE(std::fabs(cpu[i] - (T)65536) < eps);
      }

      // update the kernel
      cf.for_each(kernel,
        gpu, gpu+n, [] __device__ (T& val) { val = 100; }
      );

      run_and_wait(cf);

      for(int i=0; i<n; i++) {
        REQUIRE(std::fabs(cpu[i] - (T)100) < eps);
      }

      std::free(cpu);
      REQUIRE(cudaFree(gpu) == cudaSuccess); 
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cudaFlow.for_each.int" * doctest::timeout(300)) {
  cudaflow_for_each<int, flare::rt::cudaFlow>();
}

TEST_CASE("cudaFlow.for_each.float" * doctest::timeout(300)) {
  cudaflow_for_each<float, flare::rt::cudaFlow>();
}

TEST_CASE("cudaFlow.for_each.double" * doctest::timeout(300)) {
  cudaflow_for_each<double, flare::rt::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.for_each.int" * doctest::timeout(300)) {
  cudaflow_for_each<int, flare::rt::cudaFlowCapturer>();
}

TEST_CASE("cudaFlowCapturer.for_each.float" * doctest::timeout(300)) {
  cudaflow_for_each<float, flare::rt::cudaFlowCapturer>();
}

TEST_CASE("cudaFlowCapturer.for_each.double" * doctest::timeout(300)) {
  cudaflow_for_each<double, flare::rt::cudaFlowCapturer>();
}
