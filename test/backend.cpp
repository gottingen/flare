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
#include <fly/data.h>
#include <fly/dim4.hpp>
#include <fly/traits.hpp>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include <fly/device.h>

using fly::dtype_traits;
using fly::getAvailableBackends;
using fly::setBackend;
using std::string;
using std::vector;

const char* getActiveBackendString(fly_backend active) {
    switch (active) {
        case FLY_BACKEND_CPU: return "FLY_BACKEND_CPU";
        case FLY_BACKEND_CUDA: return "FLY_BACKEND_CUDA";
        default: return "FLY_BACKEND_DEFAULT";
    }
}

void testFunction(fly_backend expected) {
    fly_backend activeBackend = (fly_backend)0;
    fly_get_active_backend(&activeBackend);

    ASSERT_EQ(expected, activeBackend);

    fly_array outArray = 0;
    dim_t dims[]      = {32, 32};
    EXPECT_EQ(FLY_SUCCESS, fly_randu(&outArray, 2, dims, f32));

    // Verify backends returned by array and by function are the same
    fly_backend arrayBackend = (fly_backend)0;
    fly_get_backend_id(&arrayBackend, outArray);
    EXPECT_EQ(arrayBackend, activeBackend);

    // cleanup
    if (outArray != 0) { ASSERT_SUCCESS(fly_release_array(outArray)); }
}

void backendTest() {
    int backends = getAvailableBackends();

    ASSERT_NE(backends, 0);

    bool cpu    = backends & FLY_BACKEND_CPU;
    bool cuda   = backends & FLY_BACKEND_CUDA;

    if (cpu) {
        setBackend(FLY_BACKEND_CPU);
        testFunction(FLY_BACKEND_CPU);
    }

    if (cuda) {
        setBackend(FLY_BACKEND_CUDA);
        testFunction(FLY_BACKEND_CUDA);
    }

}

TEST(BACKEND_TEST, Basic) { backendTest(); }

using fly::getActiveBackend;

void test_backend(std::atomic<int>& counter, int ntests,
                  fly::Backend default_backend, fly::Backend test_backend) {
    auto ta_backend = getActiveBackend();
    ASSERT_EQ(default_backend, ta_backend);

    // Wait until all threads reach this point
    counter++;
    while (counter < ntests) {}

    setBackend(test_backend);

    // Wait until all threads reach this point
    counter++;
    while (counter < 2 * ntests) {}

    ta_backend = getActiveBackend();
    ASSERT_EQ(test_backend, ta_backend);
}

TEST(Backend, Threads) {
    using std::thread;
    std::atomic<int> count(0);

    setBackend(FLY_BACKEND_DEFAULT);
    auto default_backend = getActiveBackend();

    int numbk = fly::getBackendCount();

    thread a, b, c;
    if (fly::getAvailableBackends() & FLY_BACKEND_CPU) {
        a = thread([&]() {
            test_backend(count, numbk, default_backend, FLY_BACKEND_CPU);
        });
    }

    if (fly::getAvailableBackends() & FLY_BACKEND_CUDA) {
        c = thread([&]() {
            test_backend(count, numbk, default_backend, FLY_BACKEND_CUDA);
        });
    }

    if (a.joinable()) a.join();
    if (b.joinable()) b.join();
    if (c.joinable()) c.join();
}
