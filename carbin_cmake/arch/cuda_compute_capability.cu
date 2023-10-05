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

#include <iostream>
#include <cuda_runtime_api.h>

int main() {
    cudaDeviceProp device_properties;
    const cudaError_t error = cudaGetDeviceProperties(&device_properties,
            /*device*/ 0);
    if (error != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << '\n';
        return error;
    }
    unsigned int const compute_capability =
            device_properties.major * 10 + device_properties.minor;
#ifdef SM_ONLY
    std::cout << compute_capability;
#else
    switch (compute_capability) {
        // clang-format off
        case 30: std::cout << "Set -Dflare_ARCH_KEPLER30=ON ." << std::endl; break;
        case 32: std::cout << "Set -Dflare_ARCH_KEPLER32=ON ." << std::endl; break;
        case 35: std::cout << "Set -Dflare_ARCH_KEPLER35=ON ." << std::endl; break;
        case 37: std::cout << "Set -Dflare_ARCH_KEPLER37=ON ." << std::endl; break;
        case 50: std::cout << "Set -Dflare_ARCH_MAXWELL50=ON ." << std::endl; break;
        case 52: std::cout << "Set -Dflare_ARCH_MAXWELL52=ON ." << std::endl; break;
        case 53: std::cout << "Set -Dflare_ARCH_MAXWELL53=ON ." << std::endl; break;
        case 60: std::cout << "Set -Dflare_ARCH_PASCAL60=ON ." << std::endl; break;
        case 61: std::cout << "Set -Dflare_ARCH_PASCAL61=ON ." << std::endl; break;
        case 70: std::cout << "Set -Dflare_ARCH_VOLTA70=ON ." << std::endl; break;
        case 72: std::cout << "Set -Dflare_ARCH_VOLTA72=ON ." << std::endl; break;
        case 75: std::cout << "Set -Dflare_ARCH_TURING75=ON ." << std::endl; break;
        case 80: std::cout << "Set -Dflare_ARCH_AMPERE80=ON ." << std::endl; break;
        case 86: std::cout << "Set -Dflare_ARCH_AMPERE86=ON ." << std::endl; break;
        case 89: std::cout << "Set -Dflare_ARCH_ADA89=ON ." << std::endl; break;
        case 90: std::cout << "Set -Dflare_ARCH_HOPPER90=ON ." << std::endl; break;
        default:
            std::cout << "Compute capability " << compute_capability
                      << " is not supported" << std::endl;
            // clang-format on
    }
#endif
    return 0;
}
