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
#include <flare/runtime/cuda/algorithm/merge.h>

// ----------------------------------------------------------------------------
// cuda_merge
// ----------------------------------------------------------------------------

template <typename T>
void cuda_merge() {

  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor;
  
  for(int n1=0; n1<=123456; n1 = n1*2 + 1) {
    for(int n2=0; n2<=123456; n2 = n2*2 + 1) {
    
      taskflow.emplace([n1, n2](){

        // gpu data
        auto da = flare::rt::cuda_malloc_shared<T>(n1);
        auto db = flare::rt::cuda_malloc_shared<T>(n2);
        auto dc = flare::rt::cuda_malloc_shared<T>(n1 + n2);

        // host data
        std::vector<T> ha(n1), hb(n2), hc(n1 + n2);

        for(int i=0; i<n1; i++) {
          da[i] = ha[i] = rand()%100;
        }
        for(int i=0; i<n2; i++) {
          db[i] = hb[i] = rand()%100;
        }
        
        std::sort(da, da+n1);
        std::sort(db, db+n2);
        std::sort(ha.begin(), ha.end());
        std::sort(hb.begin(), hb.end());

        // --------------------------------------------------------------------------
        // GPU merge
        // --------------------------------------------------------------------------

        flare::rt::cudaStream stream;
        flare::rt::cudaDefaultExecutionPolicy policy(stream);

        // allocate the buffer
        void* buf;
        REQUIRE(cudaMalloc(&buf, policy.merge_bufsz(n1, n2)) == cudaSuccess);

        flare::rt::cuda_merge(policy,
          da, da+n1, db, db+n2, dc, flare::rt::cuda_less<T>{}, buf
        );
        stream.synchronize();

        // --------------------------------------------------------------------------
        // CPU merge
        // --------------------------------------------------------------------------
        std::merge(ha.begin(), ha.end(), hb.begin(), hb.end(), hc.begin());

        // --------------------------------------------------------------------------
        // verify the result
        // --------------------------------------------------------------------------

        for(int i=0; i<n1+n2; i++) {
          REQUIRE(dc[i] == hc[i]);
        }
        
        // --------------------------------------------------------------------------
        // deallocate the memory
        // --------------------------------------------------------------------------
        REQUIRE(cudaFree(da) == cudaSuccess);
        REQUIRE(cudaFree(db) == cudaSuccess);
        REQUIRE(cudaFree(dc) == cudaSuccess);
        REQUIRE(cudaFree(buf) == cudaSuccess);
      });
    }
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_merge.int" * doctest::timeout(300)) {
  cuda_merge<int>();
}

TEST_CASE("cuda_merge.float" * doctest::timeout(300)) {
  cuda_merge<float>();
}

// ----------------------------------------------------------------------------
// cuda_merge_by_key
// ----------------------------------------------------------------------------

template <typename T>
void cuda_merge_by_key() {

  flare::rt::Taskflow taskflow;
  flare::rt::Executor executor;
  
  for(int n1=0; n1<=123456; n1 = n1*2 + 1) {
    for(int n2=0; n2<=123456; n2 = n2*2 + 1) {
    
      taskflow.emplace([n1, n2](){

        // gpu data
        auto da_k = flare::rt::cuda_malloc_shared<T>(n1);
        auto da_v = flare::rt::cuda_malloc_shared<T>(n1);
        auto db_k = flare::rt::cuda_malloc_shared<T>(n2);
        auto db_v = flare::rt::cuda_malloc_shared<T>(n2);
        auto dc_k = flare::rt::cuda_malloc_shared<T>(n1 + n2);
        auto dc_v = flare::rt::cuda_malloc_shared<T>(n1 + n2);

        std::unordered_map<T, T> map;

        for(int i=0; i<n1; i++) {
          da_k[i] = 1 + 2*i;
          da_v[i] = rand();
          map[da_k[i]] = da_v[i];
        }

        for(int i=0; i<n2; i++) {
          db_k[i] = 2*i;
          db_v[i] = rand();
          map[db_k[i]] = db_v[i];
        }

        REQUIRE(map.size() == n1 + n2);
        
        flare::rt::cudaStream stream;
        flare::rt::cudaDefaultExecutionPolicy policy(stream);

        // allocate the buffer
        void* buf;
        REQUIRE(cudaMalloc(&buf, policy.merge_bufsz(n1, n2)) == cudaSuccess);

        flare::rt::cuda_merge_by_key(
          policy, 
          da_k, da_k+n1, da_v,
          db_k, db_k+n2, db_v,
          dc_k, dc_v,
          flare::rt::cuda_less<T>{},
          buf
        );
        stream.synchronize();

        // --------------------------------------------------------------------------
        // verify the result
        // --------------------------------------------------------------------------

        REQUIRE(std::is_sorted(dc_k, dc_k+n1+n2));

        for(int i=0; i<n1+n2; i++) {
          REQUIRE(map.find(dc_k[i]) != map.end());
          REQUIRE(dc_v[i] == map[dc_k[i]]);
        }
        
        // --------------------------------------------------------------------------
        // deallocate the memory
        // --------------------------------------------------------------------------
        REQUIRE(cudaFree(da_k) == cudaSuccess);
        REQUIRE(cudaFree(da_v) == cudaSuccess);
        REQUIRE(cudaFree(db_k) == cudaSuccess);
        REQUIRE(cudaFree(db_v) == cudaSuccess);
        REQUIRE(cudaFree(dc_k) == cudaSuccess);
        REQUIRE(cudaFree(dc_v) == cudaSuccess);
        REQUIRE(cudaFree(buf) == cudaSuccess);
      });
    }
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_merge_by_key.int" * doctest::timeout(300)) {
  cuda_merge_by_key<int>();
}

TEST_CASE("cuda_merge_by_key.float" * doctest::timeout(300)) {
  cuda_merge_by_key<float>();
}



