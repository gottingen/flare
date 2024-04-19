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

#include <cl2hpp.hpp>
#include <common/EventBase.hpp>
#include <fly/event.h>

namespace flare {
namespace opencl {
class OpenCLEventPolicy {
   public:
    using EventType = cl_event;
    using QueueType = cl_command_queue;
    using ErrorType = cl_int;

    static cl_int createAndMarkEvent(cl_event *e) noexcept {
        // Events are created when you mark them
        return CL_SUCCESS;
    }

    static cl_int markEvent(cl_event *e, cl_command_queue stream) noexcept {
        return clEnqueueMarkerWithWaitList(stream, 0, nullptr, e);
    }

    static cl_int waitForEvent(cl_event *e, cl_command_queue stream) noexcept {
        return clEnqueueMarkerWithWaitList(stream, 1, e, nullptr);
    }

    static cl_int syncForEvent(cl_event *e) noexcept {
        return clWaitForEvents(1, e);
    }

    static cl_int destroyEvent(cl_event *e) noexcept {
        return clReleaseEvent(*e);
    }
};

using Event = common::EventBase<OpenCLEventPolicy>;

/// \brief Creates a new event and marks it in the queue
Event makeEvent(cl::CommandQueue &queue);

fly_event createEvent();

void markEventOnActiveQueue(fly_event eventHandle);

void enqueueWaitOnActiveQueue(fly_event eventHandle);

void block(fly_event eventHandle);

fly_event createAndMarkEvent();

}  // namespace opencl
}  // namespace flare
