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
// This program demonstrates how to performs a parallel transform
// using cudaFlow.

#include <flare/runtime/cuda/cudaflow.h>
#include <flare/runtime/cuda/algorithm/transform.h>

int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./cuda_transform num_items\n";
    std::exit(EXIT_FAILURE);
  }

  size_t N = std::atoi(argv[1]);

  auto input  = flare::rt::cuda_malloc_shared<int>(N);
  auto output = flare::rt::cuda_malloc_shared<int>(N);
  
  // initialize the data
  for(size_t i=0; i<N; i++) {
    input [i] = -1;
    output[i] = 1;
  }
  
  // perform parallel transform
  flare::rt::cudaFlow cudaflow;
  flare::rt::cudaStream stream;
  
  // output[i] = input[i] + 11
  cudaflow.transform(
    input, input + N, output, [] __device__ (int a) { return a + 11; }
  );

  cudaflow.run(stream);
  stream.synchronize();

  // inspect the result
  for(size_t i=0; i<N; i++) {
    if(output[i] != 10) {
      throw std::runtime_error("incorrect result");
    }
  }

  std::cout << "correct result\n";

  return 0;
}
