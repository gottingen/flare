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
#include <flare/runtime/taskflow.h>
#include <flare/runtime/cuda/cudaflow.h>

int main() {

    // CUDA version
    std::cout << "========================================\n"
              << "CUDA version: "
              << flare::rt::cuda_get_runtime_version() << '\n'
              << "CUDA driver version: "
              << flare::rt::cuda_get_driver_version() << '\n';

    // Number of CUDA devices
    auto num_cuda_devices = flare::rt::cuda_get_num_devices();

    std::cout << "There are " << num_cuda_devices << " CUDA devices.\n";

    // Iterate each device and dump its property
    std::cout << "\nquerying device properties ...\n";
    for (size_t i = 0; i < num_cuda_devices; ++i) {
        std::cout << "CUDA device #" << i << '\n';
        flare::rt::cuda_dump_device_property(std::cout, flare::rt::cuda_get_device_property(i));
    }

    // we can also query each device property attribute by attribute
    std::cout << "\nquerying device attributes ...\n";
    for (size_t i = 0; i < num_cuda_devices; ++i) {
        std::cout << "CUDA device #" << i << '\n';
        std::cout << "Compute capability   : "
                  << flare::rt::cuda_get_device_compute_capability_major(i) << '.'
                  << flare::rt::cuda_get_device_compute_capability_minor(i) << '\n';
        std::cout << "max threads per block: "
                  << flare::rt::cuda_get_device_max_threads_per_block(i) << '\n'
                  << "max x-dim   per block: "
                  << flare::rt::cuda_get_device_max_x_dim_per_block(i) << '\n'
                  << "max y-dim   per block: "
                  << flare::rt::cuda_get_device_max_y_dim_per_block(i) << '\n'
                  << "max z-dim   per block: "
                  << flare::rt::cuda_get_device_max_z_dim_per_block(i) << '\n'
                  << "max x-dim   per grid : "
                  << flare::rt::cuda_get_device_max_x_dim_per_grid(i) << '\n'
                  << "max y-dim   per grid : "
                  << flare::rt::cuda_get_device_max_y_dim_per_grid(i) << '\n'
                  << "max z-dim   per grid : "
                  << flare::rt::cuda_get_device_max_z_dim_per_grid(i) << '\n'
                  << "warp size            : "
                  << flare::rt::cuda_get_device_warp_size(i) << '\n'
                  << "unified addressing?  : "
                  << flare::rt::cuda_get_device_unified_addressing(i) << '\n';
    }

    return 0;
}



