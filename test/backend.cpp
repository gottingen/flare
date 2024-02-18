/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

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
        case FLY_BACKEND_OPENCL: return "FLY_BACKEND_OPENCL";
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
    bool opencl = backends & FLY_BACKEND_OPENCL;

    if (cpu) {
        setBackend(FLY_BACKEND_CPU);
        testFunction(FLY_BACKEND_CPU);
    }

    if (cuda) {
        setBackend(FLY_BACKEND_CUDA);
        testFunction(FLY_BACKEND_CUDA);
    }

    if (opencl) {
        setBackend(FLY_BACKEND_OPENCL);
        testFunction(FLY_BACKEND_OPENCL);
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

    if (fly::getAvailableBackends() & FLY_BACKEND_OPENCL) {
        b = thread([&]() {
            test_backend(count, numbk, default_backend, FLY_BACKEND_OPENCL);
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
