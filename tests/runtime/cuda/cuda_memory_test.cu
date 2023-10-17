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

// ----------------------------------------------------------------------------
// USM Allocator
// ----------------------------------------------------------------------------

TEST_CASE("cudaUSMAllocator" * doctest::timeout(300)) {

  flare::rt::cudaStream stream;

  std::vector<int, flare::rt::cudaUSMAllocator<int>> vec;
  std::vector<int, flare::rt::cudaUSMAllocator<int>> rhs;

  REQUIRE(vec.size() == 0);

  vec.resize(100, 10);
  REQUIRE(vec.size() == 100);
  for(auto c : vec) {
    REQUIRE(c == 10);
  }

  rhs = std::move(vec);

  REQUIRE(vec.size() == 0);
  REQUIRE(rhs.size() == 100);
  for(auto c : rhs) {
    REQUIRE(c == 10);
  }

  for(int i=0; i<65536; i++) {
    vec.push_back(-i);
  }
  for(int i=0; i<65536; i++) {
    REQUIRE(vec[i] == -i);
  }

  rhs = vec;
  
  for(int i=0; i<65536; i++) {
    REQUIRE(vec[i] == rhs[i]);
  }

  flare::rt::cudaDefaultExecutionPolicy p(stream);
  
  flare::rt::cuda_for_each(p, vec.data(), vec.data() + vec.size(), [] __device__ (int& v) {
    v = -177;
  });
  stream.synchronize();

  rhs = vec;
  for(size_t i=0; i<vec.size(); i++) {
    REQUIRE(vec[i] == -177);
    REQUIRE(rhs[i] == vec[i]);
  }

  vec.clear();
  REQUIRE(vec.size() == 0);
}

// ----------------------------------------------------------------------------
// Device Allocator
// ----------------------------------------------------------------------------

TEST_CASE("cudaDeviceAllocator" * doctest::timeout(300)) {


  size_t N = 10000;
  
  std::vector<flare::rt::NoInit<int>, flare::rt::cudaDeviceAllocator<flare::rt::NoInit<int>>> vec;
  std::vector<flare::rt::NoInit<int>, flare::rt::cudaDeviceAllocator<flare::rt::NoInit<int>>> rhs(N);

  REQUIRE(vec.size() == 0);
  REQUIRE(rhs.size() == 10000);
  
  //flare::rt::cudaStream stream;
  //flare::rt::cudaDefaultExecutionPolicy policy(stream);
  //
  //flare::rt::cuda_for_each(policy, rhs.data(), rhs.data() + N, [] __device__ (flare::rt::NoInit<int>& v) {
  //  v = -177;
  //});
  //stream.synchronize();
}












