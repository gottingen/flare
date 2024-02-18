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
#include <string>
#include <vector>

#include <fly/device.h>

using fly::dim4;
using fly::dtype_traits;
using fly::getDevice;
using fly::info;
using fly::setDevice;
using std::string;
using std::vector;

template<typename T>
void testFunction() {
    info();

    fly_array outArray = 0;
    dim4 dims(32, 32, 1, 1);
    ASSERT_SUCCESS(fly_randu(&outArray, dims.ndims(), dims.get(),
                            (fly_dtype)dtype_traits<T>::fly_type));
    // cleanup
    if (outArray != 0) { ASSERT_SUCCESS(fly_release_array(outArray)); }
}

void infoTest() {
    int nDevices = 0;
    ASSERT_SUCCESS(fly_get_device_count(&nDevices));
    ASSERT_EQ(true, nDevices > 0);

    const char* ENV = getenv("FLY_MULTI_GPU_TESTS");
    if (ENV && ENV[0] == '0') {
        testFunction<float>();
    } else {
        int oldDevice = getDevice();
        testFunction<float>();
        for (int d = 0; d < nDevices; d++) {
            setDevice(d);
            testFunction<float>();
        }
        setDevice(oldDevice);
    }
}

TEST(Info, All) { infoTest(); }
