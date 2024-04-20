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

TEST(Memory, recover) {
    cleanSlate();  // Clean up everything done so far

    try {
        array vec[100];

        // Trying to allocate 1 Terrabyte of memory and trash the memory manager
        // should crash memory manager
        for (int i = 0; i < 1000; i++) {
            vec[i] = randu(1024, 1024, 256);  // Allocating 1GB
        }

        FAIL();
    } catch (exception &ae) {
        ASSERT_EQ(ae.err(), FLY_ERR_NO_MEM);

        const int num   = 1000 * 1000;
        const float val = 1.0;

        array a    = constant(val, num);  // This should work as expected
        float *h_a = a.host<float>();
        for (int i = 0; i < 1000 * 1000; i++) { ASSERT_EQ(h_a[i], val); }
        freeHost(h_a);
    }
}
