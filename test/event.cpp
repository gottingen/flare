/*******************************************************
 * Copyright (c) 2019, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

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
