/*******************************************************
 * Copyright (c) 2022, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <common/EventBase.hpp>
#include <fly/event.h>

#include <sycl/sycl.hpp>

namespace flare {
namespace oneapi {
class OneAPIEventPolicy {
   public:
    using EventType = sycl::event *;
    using QueueType = sycl::queue;
    using ErrorType = int;

    static ErrorType createAndMarkEvent(EventType *e) noexcept {
        *e = new sycl::event;
        return 0;
    }

    static ErrorType markEvent(EventType *e, QueueType stream) noexcept {
        **e = stream.ext_oneapi_submit_barrier();
        return 0;
    }

    static ErrorType waitForEvent(EventType *e, QueueType stream) noexcept {
        stream.ext_oneapi_submit_barrier({**e});
        return 0;
    }

    static ErrorType syncForEvent(EventType *e) noexcept {
        (*e)->wait();
        return 0;
    }

    static ErrorType destroyEvent(EventType *e) noexcept {
        delete *e;
        return 0;
    }
};

using Event = common::EventBase<OneAPIEventPolicy>;

/// \brief Creates a new event and marks it in the queue
Event makeEvent(sycl::queue &queue);

fly_event createEvent();

void markEventOnActiveQueue(fly_event eventHandle);

void enqueueWaitOnActiveQueue(fly_event eventHandle);

void block(fly_event eventHandle);

fly_event createAndMarkEvent();

}  // namespace oneapi
}  // namespace flare
