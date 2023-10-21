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

// Kernel: saxpy
__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

// Function: main
int main() {

    const unsigned N = 1 << 20;

    std::vector<float> hx(N, 1.0f), hy(N, 2.0f);

    auto dx = flare::rt::cuda_malloc_device<float>(N);
    auto dy = flare::rt::cuda_malloc_device<float>(N);

    flare::rt::cudaFlowCapturer cf;

    auto h2d_x = cf.copy(dx, hx.data(), N).name("h2d_x");
    auto h2d_y = cf.copy(dy, hy.data(), N).name("h2d_y");
    auto d2h_x = cf.copy(hx.data(), dx, N).name("d2h_x");
    auto d2h_y = cf.copy(hy.data(), dy, N).name("d2h_y");
    auto kernel = cf.kernel((N + 255) / 256, 256, 0, saxpy, N, 2.0f, dx, dy)
            .name("saxpy");
    kernel.succeed(h2d_x, h2d_y)
            .precede(d2h_x, d2h_y);

    // execute the cudaflow capturer
    std::cout << "running cudaflow capturer ...\n";
    flare::rt::cudaStream stream;
    cf.run(stream);
    stream.synchronize();

    // inspect the result
    float max_error = 0.0f;
    for (size_t i = 0; i < N; i++) {
        max_error = std::max(max_error, abs(hx[i] - 1.0f));
        max_error = std::max(max_error, abs(hy[i] - 4.0f));
    }
    std::cout << "saxpy finished with max error: " << max_error << '\n';

    // free memory
    flare::rt::cuda_free(dx);
    flare::rt::cuda_free(dy);

    // dump the cudaFlow graph
    cf.dump(std::cout);

    // dump the native CUDA graph
    cf.dump_native_graph(std::cout);

    return 0;
}

