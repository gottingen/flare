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

#include <fly/event.h>
#include "error.hpp"

namespace fly {

event::event() : e_{} { FLY_THROW(fly_create_event(&e_)); }

event::event(fly_event e) : e_(e) {}

event::~event() {
    // No dtor throw
    if (e_) { fly_delete_event(e_); }
}

// NOLINTNEXTLINE(performance-noexcept-move-constructor) we can't change the API
event::event(event&& other) : e_(other.e_) { other.e_ = 0; }

// NOLINTNEXTLINE(performance-noexcept-move-constructor) we can't change the API
event& event::operator=(event&& other) {
    fly_delete_event(this->e_);
    this->e_ = other.e_;
    other.e_ = 0;
    return *this;
}

fly_event event::get() const { return e_; }

void event::mark() { FLY_THROW(fly_mark_event(e_)); }

void event::enqueue() { FLY_THROW(fly_enqueue_wait_event(e_)); }

void event::block() const { FLY_THROW(fly_block_event(e_)); }

}  // namespace fly
