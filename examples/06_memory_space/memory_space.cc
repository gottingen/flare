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

#include <flare/core.h>
#include <cstdio>

// The type of a two-dimensional N x 3 array of double.
// It lives in flare' default memory space.
using tensor_type = flare::Tensor<double *[3]>;

// The "HostMirror" type corresponding to tensor_type above is also a
// two-dimensional N x 3 array of double.  However, it lives in the
// host memory space corresponding to tensor_type's memory space.  For
// example, if tensor_type lives in CUDA device memory, host_tensor_type
// lives in host (CPU) memory.  Furthermore, declaring host_tensor_type
// as the host mirror of tensor_type means that host_tensor_type has the
// same layout as tensor_type.  This makes it easier to copy between the
// two Tensors.
// Advanced issues: If a memory space is accessible from the host without
// performance penalties then it is its own host_mirror_space. This is
// the case for HostSpace, CudaUVMSpace and CudaHostPinnedSpace.

using host_tensor_type = tensor_type::HostMirror;

struct ReduceFunctor {
    tensor_type a;

    ReduceFunctor(tensor_type a_) : a(a_) {}

    using value_type = int;  // Specify type for reduction value, lsum

    FLARE_INLINE_FUNCTION
    void operator()(int i, int &lsum) const {
        lsum += a(i, 0) - a(i, 1) + a(i, 2);
    }
};

int main() {
    flare::initialize();

    {
        tensor_type a("A", 10);
        // If tensor_type and host_mirror_type live in the same memory space,
        // a "mirror tensor" is just an alias, and deep_copy does nothing.
        // Otherwise, a mirror tensor of a device Tensor lives in host memory,
        // and deep_copy does a deep copy.
        host_tensor_type h_a = flare::create_mirror_tensor(a);

        // The Tensor h_a lives in host (CPU) memory, so it's legal to fill
        // the tensor sequentially using ordinary code, like this.
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 3; j++) {
                h_a(i, j) = i * 10 + j;
            }
        }
        flare::deep_copy(a, h_a);  // Copy from host to device.

        int sum = 0;
        flare::parallel_reduce(10, ReduceFunctor(a), sum);
        printf("Result is %i\n", sum);
    }

    flare::finalize();
}
