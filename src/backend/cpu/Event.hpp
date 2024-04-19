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
#pragma once

#include <common/EventBase.hpp>
#include <queue.hpp>
#include <fly/event.h>

#include <type_traits>

namespace flare {
namespace cpu {

class CPUEventPolicy {
   public:
    using EventType = queue_event;
    using QueueType = std::add_lvalue_reference<queue>::type;
    using ErrorType = int;

    static int createAndMarkEvent(queue_event *e) noexcept {
        return e->create();
    }

    static int markEvent(queue_event *e, cpu::queue &stream) noexcept {
        return e->mark(stream);
    }

    static int waitForEvent(queue_event *e, cpu::queue &stream) noexcept {
        return e->wait(stream);
    }

    static int syncForEvent(queue_event *e) noexcept {
        e->sync();
        return 0;
    }

    static int destroyEvent(queue_event *e) noexcept { return 0; }
};

using Event = common::EventBase<CPUEventPolicy>;

/// \brief Creates a new event and marks it in the queue
Event makeEvent(cpu::queue &queue);

fly_event createEvent();

void markEventOnActiveQueue(fly_event eventHandle);

void enqueueWaitOnActiveQueue(fly_event eventHandle);

void block(fly_event eventHandle);

fly_event createAndMarkEvent();

}  // namespace cpu
}  // namespace flare
