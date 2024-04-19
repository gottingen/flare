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
