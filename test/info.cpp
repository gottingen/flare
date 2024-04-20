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
