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
#include <flare/runtime/cuda/algorithm/transform.h>

constexpr float eps = 0.0001f;

template <typename T>
void run_and_wait(T& cf) {
  flare::rt::cudaStream stream;
  cf.run(stream);
  stream.synchronize();
}

// ----------------------------------------------------------------------------
// cuda transform
// ----------------------------------------------------------------------------

template <typename T>
void cuda_transform() {

  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor;
  
  for(int n=1; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {

    taskflow.emplace([n](){
      
      flare::rt::cudaStream stream;
      flare::rt::cudaDefaultExecutionPolicy policy(stream);

      T v1 = ::rand() % 100;
      T v2 = ::rand() % 100;

      T* dx = flare::rt::cuda_malloc_shared<T>(n);
      T* dy = flare::rt::cuda_malloc_shared<T>(n);

      for(int i=0; i<n; i++) {
        dx[i] = v1;
        dy[i] = v2;
      }
      
      // transform
      flare::rt::cuda_transform(policy, dx, dx+n, dy,
        [] __device__ (T x) { return x + 2;  }
      );
      stream.synchronize();

      // verify the result 
      for (int i = 0; i < n; i++) {
        REQUIRE(std::fabs(dx[i] - v1) < eps);
        REQUIRE(std::fabs(dy[i] - (dx[i] + 2)) < eps);
      }

      // transform again
      flare::rt::cuda_transform(policy, dy, dy+n, dx,
        [] __device__ (T y) { return y - 4; }
      );
      stream.synchronize();
      
      // verify the result 
      for (int i = 0; i < n; i++) {
        REQUIRE(std::fabs(dx[i] - (v1 - 2)) < eps);
        REQUIRE(std::fabs(dy[i] - (v1 + 2)) < eps);
      }

      // free memory
      REQUIRE(cudaFree(dx) == cudaSuccess);
      REQUIRE(cudaFree(dy) == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_transform.int" * doctest::timeout(300)) {
  cuda_transform<int>();
}

TEST_CASE("cuda_transform.float" * doctest::timeout(300)) {
  cuda_transform<float>();
}

TEST_CASE("cuda_transform.double" * doctest::timeout(300)) {
  cuda_transform<double>();
}

// ----------------------------------------------------------------------------
// cudaflow transform
// ----------------------------------------------------------------------------

template <typename T, typename F>
void cudaflow_transform() {

  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor;
  
  for(int n=1; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {

    taskflow.emplace([n](){

      T v1 = ::rand() % 100;
      T v2 = ::rand() % 100;

      std::vector<T> hx, hy;

      T* dx {nullptr};
      T* dy {nullptr};
      
      // allocate x
      hx.resize(n, v1);
      REQUIRE(cudaMalloc(&dx, n*sizeof(T)) == cudaSuccess);

      // allocate y
      hy.resize(n, v2);
      REQUIRE(cudaMalloc(&dy, n*sizeof(T)) == cudaSuccess);
      
      // axpy
      F cf;
      auto h2d_x = cf.copy(dx, hx.data(), n).name("h2d_x");
      auto h2d_y = cf.copy(dy, hy.data(), n).name("h2d_y");
      auto d2h_x = cf.copy(hx.data(), dx, n).name("d2h_x");
      auto d2h_y = cf.copy(hy.data(), dy, n).name("d2h_y");
      auto kernel = cf.transform(dx, dx+n, dy, 
        [] __device__ (T x) { return x + 2;  }
      );
      kernel.succeed(h2d_x, h2d_y)
            .precede(d2h_x, d2h_y);

      run_and_wait(cf);

      // verify the result 
      for (int i = 0; i < n; i++) {
        REQUIRE(std::fabs(hx[i] - v1) < eps);
        REQUIRE(std::fabs(hy[i] - (hx[i] + 2)) < eps);
      }

      // update the kernel and run the cf again
      cf.transform(kernel, dy, dy+n, dx,
        [] __device__ (T y) { return y - 4; }
      );
      
      run_and_wait(cf); 
      
      // verify the result 
      for (int i = 0; i < n; i++) {
        REQUIRE(std::fabs(hx[i] - (v1 - 2)) < eps);
        REQUIRE(std::fabs(hy[i] - (v1 + 2)) < eps);
      }

      // free memory
      REQUIRE(cudaFree(dx) == cudaSuccess);
      REQUIRE(cudaFree(dy) == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cudaFlow.transform.int" * doctest::timeout(300)) {
  cudaflow_transform<int, flare::rt::cudaFlow>();
}

TEST_CASE("cudaFlow.transform.float" * doctest::timeout(300)) {
  cudaflow_transform<float, flare::rt::cudaFlow>();
}

TEST_CASE("cudaFlow.transform.double" * doctest::timeout(300)) {
  cudaflow_transform<double, flare::rt::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.transform.int" * doctest::timeout(300)) {
  cudaflow_transform<int, flare::rt::cudaFlowCapturer>();
}

TEST_CASE("cudaFlowCapturer.transform.float" * doctest::timeout(300)) {
  cudaflow_transform<float, flare::rt::cudaFlowCapturer>();
}

TEST_CASE("cudaFlowCapturer.transform.double" * doctest::timeout(300)) {
  cudaflow_transform<double, flare::rt::cudaFlowCapturer>();
}

// ----------------------------------------------------------------------------
// cuda transform2
// ----------------------------------------------------------------------------

template <typename T>
void cuda_transform2() {

  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor;
  
  for(int n=1; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {

    taskflow.emplace([n](){
      
      flare::rt::cudaStream stream;
      flare::rt::cudaDefaultExecutionPolicy policy(stream);

      T v1 = ::rand() % 100;
      T v2 = ::rand() % 100;
      T v3 = ::rand() % 1000;

      T* dx = flare::rt::cuda_malloc_shared<T>(n);
      T* dy = flare::rt::cuda_malloc_shared<T>(n);
      T* dz = flare::rt::cuda_malloc_shared<T>(n);

      for(int i=0; i<n; i++) {
        dx[i] = v1;
        dy[i] = v2;
        dz[i] = v3;
      }
      
      // transform
      flare::rt::cuda_transform(policy, dx, dx+n, dy, dz,
        [] __device__ (T x, T y) { return x + y;  }
      );
      stream.synchronize();

      // verify the result 
      for (int i = 0; i < n; i++) {
        REQUIRE(std::fabs(dx[i] - v1) < eps);
        REQUIRE(std::fabs(dy[i] - v2) < eps);
        REQUIRE(std::fabs(dz[i] - dx[i] - dy[i]) < eps);
      }

      // free memory
      REQUIRE(cudaFree(dx) == cudaSuccess);
      REQUIRE(cudaFree(dy) == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_transform2.int" * doctest::timeout(300)) {
  cuda_transform2<int>();
}

TEST_CASE("cuda_transform2.float" * doctest::timeout(300)) {
  cuda_transform2<float>();
}

TEST_CASE("cuda_transform2.double" * doctest::timeout(300)) {
  cuda_transform2<double>();
}

// ----------------------------------------------------------------------------
// cudaflow transform2
// ----------------------------------------------------------------------------

template <typename T, typename F>
void cudaflow_transform2() {

  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor;
  
  for(int n=1; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {

    taskflow.emplace([n](){

      T v1 = ::rand() % 100;
      T v2 = ::rand() % 100;
      T v3 = ::rand() % 100;

      std::vector<T> hx, hy, hz;

      T* dx {nullptr};
      T* dy {nullptr};
      T* dz {nullptr};
      
      // allocate x
      hx.resize(n, v1);
      REQUIRE(cudaMalloc(&dx, n*sizeof(T)) == cudaSuccess);

      // allocate y
      hy.resize(n, v2);
      REQUIRE(cudaMalloc(&dy, n*sizeof(T)) == cudaSuccess);
      
      // allocate z
      hz.resize(n, v3);
      REQUIRE(cudaMalloc(&dz, n*sizeof(T)) == cudaSuccess);
      
      // axpy
      F cf;
      auto h2d_x = cf.copy(dx, hx.data(), n).name("h2d_x");
      auto h2d_y = cf.copy(dy, hy.data(), n).name("h2d_y");
      auto h2d_z = cf.copy(dz, hz.data(), n).name("h2d_z");
      auto d2h_x = cf.copy(hx.data(), dx, n).name("d2h_x");
      auto d2h_y = cf.copy(hy.data(), dy, n).name("d2h_y");
      auto d2h_z = cf.copy(hz.data(), dz, n).name("d2h_z");
      auto kernel = cf.transform(dx, dx+n, dy, dz,
        [] __device__ (T x, T y) { return x + y;  }
      );
      kernel.succeed(h2d_x, h2d_y, h2d_z)
            .precede(d2h_x, d2h_y, d2h_z);

      run_and_wait(cf);

      // verify the result 
      for (int i = 0; i < n; i++) {
        REQUIRE(std::fabs(hx[i] - v1) < eps);
        REQUIRE(std::fabs(hy[i] - v2) < eps);
        REQUIRE(std::fabs(hz[i] - v1 - v2) < eps);
      }

      // update the kernel and run the cf again
      // dz = v1 + v2
      // dx = v1
      // dy = v2
      cf.transform(kernel, dz, dz+n, dx, dy,
        [] __device__ (T z, T x) { return z + x + T(10); }
      );
      
      run_and_wait(cf); 
      
      // verify the result 
      for (int i = 0; i < n; i++) {
        REQUIRE(std::fabs(hy[i] - (v1 + v2 + v1 + T(10))) < eps);
      }

      // free memory
      REQUIRE(cudaFree(dx) == cudaSuccess);
      REQUIRE(cudaFree(dy) == cudaSuccess);
      REQUIRE(cudaFree(dz) == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cudaFlow.transform2.int" * doctest::timeout(300)) {
  cudaflow_transform2<int, flare::rt::cudaFlow>();
}

TEST_CASE("cudaFlow.transform2.float" * doctest::timeout(300)) {
  cudaflow_transform2<float, flare::rt::cudaFlow>();
}

TEST_CASE("cudaFlow.transform2.double" * doctest::timeout(300)) {
  cudaflow_transform2<double, flare::rt::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.transform2.int" * doctest::timeout(300)) {
  cudaflow_transform2<int, flare::rt::cudaFlowCapturer>();
}

TEST_CASE("cudaFlowCapturer.transform2.float" * doctest::timeout(300)) {
  cudaflow_transform2<float, flare::rt::cudaFlowCapturer>();
}

TEST_CASE("cudaFlowCapturer.transform2.double" * doctest::timeout(300)) {
  cudaflow_transform2<double, flare::rt::cudaFlowCapturer>();
}
