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

#include <events.hpp>

#include <Event.hpp>
#include <common/err_common.hpp>
#include <fly/device.h>
#include <fly/event.h>

using detail::block;
using detail::createEvent;
using detail::enqueueWaitOnActiveQueue;
using detail::Event;
using detail::markEventOnActiveQueue;

Event &getEvent(fly_event handle) {
    Event &event = *static_cast<Event *>(handle);
    return event;
}

fly_event getHandle(Event &event) { return static_cast<fly_event>(&event); }

fly_err fly_create_event(fly_event *handle) {
    try {
        FLY_CHECK(fly_init());
        *handle = createEvent();
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_delete_event(fly_event handle) {
    try {
        delete &getEvent(handle);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_mark_event(const fly_event handle) {
    try {
        markEventOnActiveQueue(handle);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_enqueue_wait_event(const fly_event handle) {
    try {
        enqueueWaitOnActiveQueue(handle);
    }
    CATCHALL;

    return FLY_SUCCESS;
}

fly_err fly_block_event(const fly_event handle) {
    try {
        block(handle);
    }
    CATCHALL;

    return FLY_SUCCESS;
}
