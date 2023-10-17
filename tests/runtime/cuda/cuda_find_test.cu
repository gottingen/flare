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
// cuda_find_if
// ----------------------------------------------------------------------------

template <typename T>
void cuda_find_if() {

  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor;
  
  for(int n=0; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {

    taskflow.emplace([n](){

      flare::rt::cudaStream stream;
      flare::rt::cudaDefaultExecutionPolicy policy(stream);
  
      // gpu data
      auto gdata = flare::rt::cuda_malloc_shared<T>(n);
      auto gfind = flare::rt::cuda_malloc_shared<unsigned>(1);

      // cpu data
      auto hdata = std::vector<T>(n);

      // initialize the data
      for(int i=0; i<n; i++) {
        T k = rand()% 100;
        gdata[i] = k;
        hdata[i] = k;
      }

      // --------------------------------------------------------------------------
      // GPU find
      // --------------------------------------------------------------------------
      flare::rt::cudaStream s;
      flare::rt::cudaDefaultExecutionPolicy p(s);
      flare::rt::cuda_find_if(
        p, gdata, gdata+n, gfind, []__device__(T v) { return v == (T)50; }
      );
      s.synchronize();
      
      // --------------------------------------------------------------------------
      // CPU find
      // --------------------------------------------------------------------------
      auto hiter = std::find_if(
        hdata.begin(), hdata.end(), [=](T v) { return v == (T)50; }
      );
      
      // --------------------------------------------------------------------------
      // verify the result
      // --------------------------------------------------------------------------
      unsigned hfind = std::distance(hdata.begin(), hiter);
      REQUIRE(*gfind == hfind);

      REQUIRE(cudaFree(gdata) == cudaSuccess);
      REQUIRE(cudaFree(gfind) == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_find_if.int" * doctest::timeout(300)) {
  cuda_find_if<int>();
}

TEST_CASE("cuda_find_if.float" * doctest::timeout(300)) {
  cuda_find_if<float>();
}
