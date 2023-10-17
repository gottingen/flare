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

#include <flare/runtime/cuda/cudaflow.h>
#include <flare/runtime/cuda/algorithm/merge.h>

int main(int argc, char* argv[]) {
  
  if(argc != 2) {
    std::cerr << "usage: ./cuda_merge N\n";
    std::exit(EXIT_FAILURE);
  }

  unsigned N = std::atoi(argv[1]);
  
  // gpu data
  auto da = flare::rt::cuda_malloc_shared<int>(N);
  auto db = flare::rt::cuda_malloc_shared<int>(N);
  auto dc = flare::rt::cuda_malloc_shared<int>(N + N);

  // host data
  std::vector<int> ha(N), hb(N), hc(N + N);

  for(unsigned i=0; i<N; i++) {
    da[i] = ha[i] = rand()%100;
    db[i] = hb[i] = rand()%100;
  }
  
  std::sort(da, da+N);
  std::sort(db, db+N);
  std::sort(ha.begin(), ha.end());
  std::sort(hb.begin(), hb.end());

  // --------------------------------------------------------------------------
  // GPU merge
  // --------------------------------------------------------------------------

  flare::rt::cudaStream stream;
  flare::rt::cudaDefaultExecutionPolicy policy(stream);

  // allocate the buffer
  void* buf;
  cudaMalloc(&buf, policy.merge_bufsz(N, N));

  auto beg = std::chrono::steady_clock::now();
  flare::rt::cuda_merge(policy,
    da, da+N, db, db+N, dc, flare::rt::cuda_less<int>{}, buf
  );
  stream.synchronize();
  auto end = std::chrono::steady_clock::now();

  std::cout << "GPU merge: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(end-beg).count()
            << " us\n";
  
  // --------------------------------------------------------------------------
  // CPU merge
  // --------------------------------------------------------------------------
  beg = std::chrono::steady_clock::now();
  std::merge(ha.begin(), ha.end(), hb.begin(), hb.end(), hc.begin());
  end = std::chrono::steady_clock::now();
  
  std::cout << "CPU merge: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(end-beg).count()
            << " us\n";

  // --------------------------------------------------------------------------
  // verify the result
  // --------------------------------------------------------------------------

  for(size_t i=0; i<N; i++) {
    if(dc[i] != hc[i]) {
      throw std::runtime_error("incorrect result");
    }
  }

  std::cout << "correct result\n";
  
  // --------------------------------------------------------------------------
  // deallocate the memory
  // --------------------------------------------------------------------------
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  cudaFree(buf);

  return 0;
};
