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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fly/event.h>

namespace flare {
namespace cuda {

class CUDARuntimeEventPolicy {
   public:
    using EventType = CUevent;
    using QueueType = CUstream;
    using ErrorType = CUresult;

    static ErrorType createAndMarkEvent(CUevent *e) noexcept {
        // Creating events with the CU_EVENT_BLOCKING_SYNC flag
        // severly impacts the speed if/when creating many arrays
        auto err = cuEventCreate(e, CU_EVENT_DISABLE_TIMING);
        return err;
    }

    static ErrorType markEvent(CUevent *e, QueueType &stream) noexcept {
        auto err = cuEventRecord(*e, stream);
        return err;
    }

    static ErrorType waitForEvent(CUevent *e, QueueType &stream) noexcept {
        auto err = cuStreamWaitEvent(stream, *e, 0);
        return err;
    }

    static ErrorType syncForEvent(CUevent *e) noexcept {
        return cuEventSynchronize(*e);
    }

    static ErrorType destroyEvent(CUevent *e) noexcept {
        auto err = cuEventDestroy(*e);
        return err;
    }
};

using Event = common::EventBase<CUDARuntimeEventPolicy>;

/// \brief Creates a new event and marks it in the stream
Event makeEvent(cudaStream_t queue);

fly_event createEvent();

void markEventOnActiveQueue(fly_event eventHandle);

void enqueueWaitOnActiveQueue(fly_event eventHandle);

void block(fly_event eventHandle);

fly_event createAndMarkEvent();

}  // namespace cuda
}  // namespace flare
