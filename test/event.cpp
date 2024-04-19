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
#include <fly/event.h>

#include <memory>
#include <utility>

#include <iostream>

using fly::event;

TEST(EventTests, SimpleCreateRelease) {
    fly_event event;
    ASSERT_SUCCESS(fly_create_event(&event));
    ASSERT_SUCCESS(fly_delete_event(event));
}

TEST(EventTests, MarkEnqueueAndBlock) {
    fly_event event;
    ASSERT_SUCCESS(fly_create_event(&event));
    ASSERT_SUCCESS(fly_mark_event(event));
    ASSERT_SUCCESS(fly_enqueue_wait_event(event));
    ASSERT_SUCCESS(fly_block_event(event));
    ASSERT_SUCCESS(fly_delete_event(event));
}

TEST(EventTests, EventCreateAndMove) {
    fly_event eventHandle;
    ASSERT_SUCCESS(fly_create_event(&eventHandle));

    event e(eventHandle);
    e.mark();
    ASSERT_EQ(eventHandle, e.get());

    auto otherEvent = std::move(e);
    ASSERT_EQ(otherEvent.get(), eventHandle);

    event f;
    fly_event fE        = f.get();
    event anotherEvent = std::move(f);
    ASSERT_EQ(fE, anotherEvent.get());
    fly::sync();
}
