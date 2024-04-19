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

#include <err_oneapi.hpp>
#include <events.hpp>
#include <platform.hpp>
#include <fly/event.h>
#include <memory>

#include <memory>

using std::make_unique;
using std::unique_ptr;

namespace flare {
namespace oneapi {
/// \brief Creates a new event and marks it in the queue
Event makeEvent(sycl::queue& queue) {
    Event e;
    if (e.create() == 0) { e.mark(queue); }
    return e;
}

fly_event createEvent() {
    auto e = make_unique<Event>();
    // Ensure the default CL command queue is initialized
    getQueue();
    if (e->create() != 0) {
        FLY_ERROR("Could not create event", FLY_ERR_RUNTIME);
    }
    Event& ref = *e.release();
    return getHandle(ref);
}

void markEventOnActiveQueue(fly_event eventHandle) {
    Event& event = getEvent(eventHandle);
    // Use the currently-active stream
    if (event.mark(getQueue()) != 0) {
        FLY_ERROR("Could not mark event on active queue", FLY_ERR_RUNTIME);
    }
}

void enqueueWaitOnActiveQueue(fly_event eventHandle) {
    Event& event = getEvent(eventHandle);
    // Use the currently-active stream
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

}  // namespace oneapi
}  // namespace flare
