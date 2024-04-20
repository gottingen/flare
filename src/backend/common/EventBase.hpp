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
#include <utility>

namespace flare {
namespace common {

template<typename NativeEventPolicy>
class EventBase {
    using QueueType = typename NativeEventPolicy::QueueType;
    using EventType = typename NativeEventPolicy::EventType;
    using ErrorType = typename NativeEventPolicy::ErrorType;
    EventType e_;

   public:
    /// Default constructor of the Event object. Does not create the event.
    constexpr EventBase() noexcept : e_() {}

    /// Deleted copy constructor
    ///
    /// The event object can only be moved.
    EventBase(EventBase &other) = delete;

    /// \brief Move constructor of the Event object. Resets the moved object to
    ///        an invalid event.
    EventBase(EventBase &&other) noexcept
        : e_(std::forward<EventType>(other.e_)) {
        other.e_ = 0;
    }

    /// \brief Event destructor. Calls the destroy event call on the native API
    ~EventBase() noexcept {
        // if (e_)
        NativeEventPolicy::destroyEvent(&e_);
    }

    /// \brief Creates the event object by calling the native create API
    ErrorType create() noexcept {
        return NativeEventPolicy::createAndMarkEvent(&e_);
    }

    /// \brief Adds the event on the queue. Once this point on the program
    ///        is executed, the event is marked complete.
    ///
    /// \returns the error code for the mark call
    ErrorType mark(QueueType queue) noexcept {
        return NativeEventPolicy::markEvent(&e_, queue);
    }

    /// \brief This is an asynchronous function which will block the
    ///        queue/stream from progressing before continuing forward. It will
    ///        not block the calling thread.
    ///
    /// \param queue The queue that will wait for the previous tasks to complete
    ///
    /// \returns the error code for the wait call
    ErrorType enqueueWait(QueueType queue) noexcept {
        return NativeEventPolicy::waitForEvent(&e_, queue);
    }

    /// \brief This function will block the calling thread until the event has
    ///        completed
    ErrorType block() noexcept { return NativeEventPolicy::syncForEvent(&e_); }

    /// \brief Returns true if the event is a valid event.
    constexpr operator bool() const { return e_; }

    EventBase &operator=(EventBase &other) = delete;

    EventBase &operator=(EventBase &&other) noexcept {
        e_       = std::move(other.e_);
        other.e_ = 0;
        return *this;
    }
};

}  // namespace common
}  // namespace flare
