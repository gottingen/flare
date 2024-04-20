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

#include <flare.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <fly/dim4.hpp>
#include <fly/traits.hpp>
#include <iostream>
#include <string>
#include <vector>

using fly::array;
using fly::deviceGC;
using fly::deviceMemInfo;
using std::cout;
using std::endl;
using std::string;
using std::vector;

const size_t step_bytes = 1024;

// This test should be by itself as it leaks memory intentionally
TEST(Memory, lock) {
    size_t alloc_bytes, alloc_buffers;
    size_t lock_bytes, lock_buffers;

    cleanSlate();  // Clean up everything done so far

    const dim_t num = step_bytes / sizeof(float);

    vector<float> in(num);

    fly_array arr = 0;
    ASSERT_SUCCESS(fly_create_array(&arr, &in[0], 1, &num, f32));

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, step_bytes);
    ASSERT_EQ(lock_bytes, step_bytes);

    // arr1 gets released by end of the following code block
    {
        array a(arr);
        a.lock();

        // No new memory should be allocated
        deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

        ASSERT_EQ(alloc_buffers, 1u);
        ASSERT_EQ(lock_buffers, 1u);
        ASSERT_EQ(alloc_bytes, step_bytes);
        ASSERT_EQ(lock_bytes, step_bytes);
    }

    // Making sure all unlocked buffers are freed
    deviceGC();

    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    ASSERT_EQ(alloc_buffers, 1u);
    ASSERT_EQ(lock_buffers, 1u);
    ASSERT_EQ(alloc_bytes, step_bytes);
    ASSERT_EQ(lock_bytes, step_bytes);
}
