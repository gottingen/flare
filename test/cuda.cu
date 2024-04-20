// Copyright 2023 The EA Authors.
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

#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <fly/array.h>
#include <fly/device.h>

using fly::fly_alloc;
using fly::fly_free;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
TEST(Memory, AfAllocDeviceCUDA) {
    void *ptr;
    ASSERT_SUCCESS(fly_alloc_device(&ptr, sizeof(float)));

    /// Tests to see if the pointer returned can be used by cuda functions
    float gold_val = 5;
    float *gold    = NULL;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&gold, sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(gold, &gold_val, sizeof(float),
                                      cudaMemcpyHostToDevice));

    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(ptr, gold, sizeof(float), cudaMemcpyDeviceToDevice));

    float host;
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(&host, ptr, sizeof(float), cudaMemcpyDeviceToHost));
    ASSERT_SUCCESS(fly_free_device(ptr));

    ASSERT_EQ(5, host);
}
#pragma GCC diagnostic pop

TEST(Memory, AfAllocDeviceV2CUDA) {
    void *ptr;
    ASSERT_SUCCESS(fly_alloc_device(&ptr, sizeof(float)));

    /// Tests to see if the pointer returned can be used by cuda functions
    float gold_val = 5;
    float *gold    = NULL;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&gold, sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(gold, &gold_val, sizeof(float),
                                      cudaMemcpyHostToDevice));

    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(ptr, gold, sizeof(float), cudaMemcpyDeviceToDevice));

    float host;
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(&host, ptr, sizeof(float), cudaMemcpyDeviceToHost));
    ASSERT_SUCCESS(fly_free_device(ptr));

    ASSERT_EQ(5, host);
}

TEST(Memory, SNIPPET_AllocCUDA) {
    //! [ex_alloc_v2_cuda]

    void *ptr = fly_alloc(sizeof(float));

    float *dptr     = static_cast<float *>(ptr);
    float host_data = 5.0f;

    cudaError_t error = cudaSuccess;
    error = cudaMemcpy(dptr, &host_data, sizeof(float), cudaMemcpyHostToDevice);
    fly_free(ptr);

    //! [ex_alloc_v2_cuda]
    ASSERT_EQ(cudaSuccess, error);
}
