/*******************************************************
 * Copyright (c) 2019, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

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
