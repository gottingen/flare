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

// This program demonstrates how to rebind a cudaFlowCapturer task
// to another GPU operation.

#include <flare/runtime/cuda/cudaflow.h>
#include <flare/runtime/cuda/algorithm/for_each.h>

int main() {

  size_t N = 10000;

  auto data = flare::rt::cuda_malloc_shared<int>(N);
  
  flare::rt::cudaFlowCapturer cudaflow;
  flare::rt::cudaStream stream;

  // set data to -1
  for(size_t i=0; i<N; i++) {
    data[i] = -1;
  }
  
  // clear data with 0
  std::cout << "clearing data with 0 ...\n";

  flare::rt::cudaTask task = cudaflow.memset(data, 0, N*sizeof(int));
  cudaflow.run(stream);
  stream.synchronize();

  for(size_t i=0; i<N; i++) {
    if(data[i] != 0) {
      std::cout << data[i] << '\n';
      throw std::runtime_error("unexpected result after fill");
    }
  }
  std::cout << "correct result after fill\n";

  // Rebind the task to for-each task setting each element to 100.
  // You can rebind a capture task to any other task type.
  std::cout << "rebind to for_each task setting each element to 100 ...\n";

  cudaflow.for_each(
    task, data, data+N, [] __device__ (int& i){ i = 100; }
  );
  cudaflow.run(stream);
  stream.synchronize();
  
  for(size_t i=0; i<N; i++) {
    if(data[i] != 100) {
      throw std::runtime_error("unexpected result after for_each");
    }
  }
  std::cout << "correct result after updating for_each\n";

  return 0;
}



