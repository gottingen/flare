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
#include <cuda_runtime_api.h>
#include <events.hpp>
#include <platform.hpp>
#include <fly/event.h>

#include <memory>

namespace flare {
namespace cuda {
/// \brief Creates a new event and marks it in the queue
Event makeEvent(cudaStream_t queue) {
    Event e;
    if (e.create() == CUDA_SUCCESS) { e.mark(queue); }
    return e;
}

fly_event createEvent() {
    // Default CUDA stream needs to be initialized to use the CUDA driver
    // Ctx
    getActiveStream();
    auto e = std::make_unique<Event>();
    if (e->create() != CUDA_SUCCESS) {
        FLY_ERROR("Could not create event", FLY_ERR_RUNTIME);
    }
    return getHandle(*(e.release()));
}

void markEventOnActiveQueue(fly_event eventHandle) {
    Event& event = getEvent(eventHandle);
    // Use the currently-active stream
    cudaStream_t stream = getActiveStream();
    if (event.mark(stream) != CUDA_SUCCESS) {
        FLY_ERROR("Could not mark event on active stream", FLY_ERR_RUNTIME);
    }
}

void enqueueWaitOnActiveQueue(fly_event eventHandle) {
    Event& event = getEvent(eventHandle);
    // Use the currently-active stream
    cudaStream_t stream = getActiveStream();
    if (event.enqueueWait(stream) != CUDA_SUCCESS) {
        FLY_ERROR("Could not enqueue wait on active stream for event",
                 FLY_ERR_RUNTIME);
    }
}

void block(fly_event eventHandle) {
    Event& event = getEvent(eventHandle);
    if (event.block() != CUDA_SUCCESS) {
        FLY_ERROR("Could not block on active stream for event", FLY_ERR_RUNTIME);
    }
}

fly_event createAndMarkEvent() {
    fly_event handle = createEvent();
    markEventOnActiveQueue(handle);
    return handle;
}

}  // namespace cuda
}  // namespace flare
