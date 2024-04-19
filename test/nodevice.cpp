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

// Include functions that provide information about the system and shouldn't
// throw exceptions during runtime.

#include <flare.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>

TEST(NoDevice, Info) { ASSERT_SUCCESS(fly_info()); }

TEST(NoDevice, InfoCxx) { fly::info(); }

TEST(NoDevice, InfoString) {
    char* str;
    ASSERT_SUCCESS(fly_info_string(&str, true));
    ASSERT_SUCCESS(fly_free_host((void*)str));
}

TEST(NoDevice, GetDeviceCount) {
    int device = 0;
    ASSERT_SUCCESS(fly_get_device_count(&device));
}

TEST(NoDevice, GetDeviceCountCxx) { fly::getDeviceCount(); }

TEST(NoDevice, GetSizeOf) {
    size_t size;
    ASSERT_SUCCESS(fly_get_size_of(&size, f32));
    ASSERT_EQ(4, size);
}

TEST(NoDevice, GetSizeOfCxx) {
    size_t size = fly::getSizeOf(f32);
    ASSERT_EQ(4, size);
}

TEST(NoDevice, GetBackendCount) {
    unsigned int nbackends;
    ASSERT_SUCCESS(fly_get_backend_count(&nbackends));
}

TEST(NoDevice, GetBackendCountCxx) {
    unsigned int nbackends = fly::getBackendCount();
    UNUSED(nbackends);
}

TEST(NoDevice, GetVersion) {
    int major = 0, minor = 0, patch = 0;

    ASSERT_SUCCESS(fly_get_version(&major, &minor, &patch));

    ASSERT_EQ(FLY_VERSION_MAJOR, major);
    ASSERT_EQ(FLY_VERSION_MINOR, minor);
    ASSERT_EQ(FLY_VERSION_PATCH, patch);
}

TEST(NoDevice, GetRevision) {
    const char* revision = fly_get_revision();
    UNUSED(revision);
}
