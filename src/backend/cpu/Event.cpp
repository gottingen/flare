/*******************************************************
 * Copyright (c) 2019, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Event.hpp>

#include <common/err_common.hpp>
#include <events.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <fly/event.h>
#include <memory>

using std::make_unique;

namespace flare {
namespace cpu {
/// \brief Creates a new event and marks it in the queue
Event makeEvent(cpu::queue& queue) {
    Event e;
    if (0 == e.create()) { e.mark(queue); }
    return e;
}

fly_event createEvent() {
    auto e = make_unique<Event>();
    // Ensure that the default queue is initialized
    getQueue();
    if (e->create() != 0) {
        FLY_ERROR("Could not create event", FLY_ERR_RUNTIME);
    }
    Event& ref = *e.release();
    return getHandle(ref);
}

void markEventOnActiveQueue(fly_event eventHandle) {
    Event& event = getEvent(eventHandle);
    // Use the currently-active queue
    if (event.mark(getQueue()) != 0) {
        FLY_ERROR("Could not mark event on active queue", FLY_ERR_RUNTIME);
    }
}

void enqueueWaitOnActiveQueue(fly_event eventHandle) {
    Event& event = getEvent(eventHandle);
    // Use the currently-active queue
    if (event.enqueueWait(getQueue()) != 0) {
        FLY_ERROR("Could not enqueue wait on active queue for event",
                 FLY_ERR_RUNTIME);
    }
}

void block(fly_event eventHandle) {
    Event& event = getEvent(eventHandle);
    if (event.block() != 0) {
        FLY_ERROR("Could not block on active queue for event", FLY_ERR_RUNTIME);
    }
}

fly_event createAndMarkEvent() {
    fly_event handle = createEvent();
    markEventOnActiveQueue(handle);
    return handle;
}

}  // namespace cpu
}  // namespace flare
