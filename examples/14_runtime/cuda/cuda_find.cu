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
#include <flare/runtime/cuda/algorithm/find.h>

int main(int argc, char *argv[]) {

    if (argc != 2) {
        std::cerr << "usage: ./cuda_find N\n";
        std::exit(EXIT_FAILURE);
    }

    unsigned N = std::atoi(argv[1]);

    // gpu data
    auto gdata = flare::rt::cuda_malloc_shared<int>(N);
    auto gfind = flare::rt::cuda_malloc_shared<unsigned>(1);

    // cpu data
    auto hdata = std::vector<int>(N);

    size_t tgpu{0}, tcpu{0};

    // initialize the data
    for (unsigned i = 0; i < N; i++) {
        auto k = rand();
        gdata[i] = k;
        hdata[i] = k;
    }

    // --------------------------------------------------------------------------
    // GPU find
    // --------------------------------------------------------------------------
    auto beg = std::chrono::steady_clock::now();
    flare::rt::cudaStream s;
    flare::rt::cudaDefaultExecutionPolicy p(s);
    flare::rt::cuda_find_if(
            p, gdata, gdata + N, gfind, []__device__(int v) { return v == 100; }
    );
    s.synchronize();
    auto end = std::chrono::steady_clock::now();
    tgpu += std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count();

    // --------------------------------------------------------------------------
    // CPU find
    // --------------------------------------------------------------------------
    beg = std::chrono::steady_clock::now();
    auto hiter = std::find_if(
            hdata.begin(), hdata.end(), [=](int v) { return v == 100; }
    );
    end = std::chrono::steady_clock::now();
    tcpu += std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count();

    // --------------------------------------------------------------------------
    // verify the result
    // --------------------------------------------------------------------------
    if (unsigned hfind = std::distance(hdata.begin(), hiter); *gfind != hfind) {
        printf("gdata[%u]=%d, hdata[%u]=%d\n",
               *gfind, gdata[*gfind], hfind, hdata[hfind]
        );
        throw std::runtime_error("incorrect result");
    }

    // output the time
    std::cout << "GPU time: " << tgpu << '\n'
              << "CPU time: " << tcpu << std::endl;

    // delete the memory
    flare::rt::cuda_free(gdata);
    flare::rt::cuda_free(gfind);

    return 0;
}
