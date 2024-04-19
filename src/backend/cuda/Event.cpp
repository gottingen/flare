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
