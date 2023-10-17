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
#include <flare/runtime/cuda/algorithm/find.h>

// ----------------------------------------------------------------------------
// cuda_min_max_element
// ----------------------------------------------------------------------------

template <typename T>
void cuda_min_max_element() {

  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor;
  
  for(int n=1; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {

    taskflow.emplace([n](){

      flare::rt::cudaStream stream;
      flare::rt::cudaDefaultExecutionPolicy policy(stream);
  
      // gpu data
      auto gdata = flare::rt::cuda_malloc_shared<T>(n);
      auto min_i = flare::rt::cuda_malloc_shared<unsigned>(1);
      auto max_i = flare::rt::cuda_malloc_shared<unsigned>(1);

      // buffer
      void* buff;
      cudaMalloc(&buff, policy.min_element_bufsz<T>(n));

      for(int i=0; i<n; i++) {
        gdata[i] = rand() % 1000 - 500;
      }

      // --------------------------------------------------------------------------
      // GPU find
      // --------------------------------------------------------------------------
      flare::rt::cudaStream s;
      flare::rt::cudaDefaultExecutionPolicy p(s);

      flare::rt::cuda_min_element(
        p, gdata, gdata+n, min_i, []__device__(T a, T b) { return a < b; }, buff
      );
      
      flare::rt::cuda_max_element(
        p, gdata, gdata+n, max_i, []__device__(T a, T b) { return a < b; }, buff
      );
      s.synchronize();
      
      auto min_v = *std::min_element(gdata, gdata+n, [](T a, T b) { return a < b; });
      auto max_v = *std::max_element(gdata, gdata+n, [](T a, T b) { return a < b; });

      REQUIRE(gdata[*min_i] == min_v);
      REQUIRE(gdata[*max_i] == max_v);
      
      // change the comparator
      flare::rt::cuda_min_element(
        p, gdata, gdata+n, min_i, []__device__(T a, T b) { return a > b; }, buff
      );
      
      flare::rt::cuda_max_element(
        p, gdata, gdata+n, max_i, []__device__(T a, T b) { return a > b; }, buff
      );
      s.synchronize();
      
      min_v = *std::min_element(gdata, gdata+n, [](T a, T b) { return a > b; });
      max_v = *std::max_element(gdata, gdata+n, [](T a, T b) { return a > b; });

      REQUIRE(gdata[*min_i] == min_v);
      REQUIRE(gdata[*max_i] == max_v);

      // change the comparator
      flare::rt::cuda_min_element(
        p, gdata, gdata+n, min_i, []__device__(T a, T b) { return -a > -b; }, buff
      );
      
      flare::rt::cuda_max_element(
        p, gdata, gdata+n, max_i, []__device__(T a, T b) { return -a > -b; }, buff
      );
      s.synchronize();
      
      min_v = *std::min_element(gdata, gdata+n, [](T a, T b) { return -a > -b; });
      max_v = *std::max_element(gdata, gdata+n, [](T a, T b) { return -a > -b; });

      REQUIRE(gdata[*min_i] == min_v);
      REQUIRE(gdata[*max_i] == max_v);
      
      // change the comparator
      flare::rt::cuda_min_element(
        p, gdata, gdata+n, min_i, 
        []__device__(T a, T b) { return std::abs(a) < std::abs(b); }, 
        buff
      );
      
      flare::rt::cuda_max_element(
        p, gdata, gdata+n, max_i, 
        []__device__(T a, T b) { return std::abs(a) < std::abs(b); }, 
        buff
      );
      s.synchronize();
      
      min_v = *std::min_element(
        gdata, gdata+n, [](T a, T b) { return std::abs(a) < std::abs(b); }
      );

      max_v = *std::max_element(
        gdata, gdata+n, [](T a, T b) { return std::abs(a) < std::abs(b); }
      );

      REQUIRE(std::abs(gdata[*min_i]) == std::abs(min_v));
      REQUIRE(std::abs(gdata[*max_i]) == std::abs(max_v));
      
      // deallocate the memory
      REQUIRE(cudaFree(gdata) == cudaSuccess);
      REQUIRE(cudaFree(min_i) == cudaSuccess);
      REQUIRE(cudaFree(max_i) == cudaSuccess);
      REQUIRE(cudaFree(buff)  == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_min_max_element.int" * doctest::timeout(300)) {
  cuda_min_max_element<int>();
}

TEST_CASE("cuda_min_max_element.float" * doctest::timeout(300)) {
  cuda_min_max_element<float>();
}

