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
#include <fly/event.h>
#include <memory>

#include <memory>

using std::make_unique;
using std::unique_ptr;

namespace flare {
namespace opencl {
/// \brief Creates a new event and marks it in the queue
Event makeEvent(cl::CommandQueue& queue) {
    Event e;
    if (e.create() == CL_SUCCESS) { e.mark(queue()); }
    return e;
}

fly_event createEvent() {
    auto e = make_unique<Event>();
    // Ensure the default CL command queue is initialized
    getQueue()();
    if (e->create() != CL_SUCCESS) {
        FLY_ERROR("Could not create event", FLY_ERR_RUNTIME);
    }
    Event& ref = *e.release();
    return getHandle(ref);
}

void markEventOnActiveQueue(fly_event eventHandle) {
    Event& event = getEvent(eventHandle);
    // Use the currently-active stream
    if (event.mark(getQueue()()) != CL_SUCCESS) {
        FLY_ERROR("Could not mark event on active queue", FLY_ERR_RUNTIME);
    }
}

void enqueueWaitOnActiveQueue(fly_event eventHandle) {
    Event& event = getEvent(eventHandle);
    // Use the currently-active stream
    if (event.enqueueWait(getQueue()()) != CL_SUCCESS) {
        FLY_ERROR("Could not enqueue wait on active queue for event",
                 FLY_ERR_RUNTIME);
    }
}

void block(fly_event eventHandle) {
    Event& event = getEvent(eventHandle);
    if (event.block() != CL_SUCCESS) {
        FLY_ERROR("Could not block on active queue for event", FLY_ERR_RUNTIME);
    }
}

fly_event createAndMarkEvent() {
    fly_event handle = createEvent();
    markEventOnActiveQueue(handle);
    return handle;
}

}  // namespace opencl
}  // namespace flare
