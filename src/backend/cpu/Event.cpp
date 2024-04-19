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
